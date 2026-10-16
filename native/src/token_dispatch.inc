// Included inside the backend kernel collection. The program, not this switch,
// defines the architecture, residual wiring, layer count and weight bindings.
using Value = std::variant<V, HC, Routes, ExpertRows, std::string>;
static std::string expert_name(std::string pattern, int expert) {
  auto at = pattern.find("{expert}");
  if (at == std::string::npos)
    throw std::runtime_error("expert binding missing placeholder");
  pattern.replace(at, 8, std::to_string(expert));
  return pattern;
}
V token(int id, int pos) {
  current_position = pos;
  if (pos < 0 || pos >= int(program.context) || id < 0 || id >= 151552)
    throw std::runtime_error("token/context bound");
  std::vector<std::unique_ptr<Value>> values(program.nodes.size());
  auto remaining = program.uses;
  for (size_t i = 0; i < program.nodes.size(); ++i) {
    const auto &n = program.nodes[i];
    auto input = [&](size_t j) -> Value & {
      return *values.at(n.inputs.at(j));
    };
    auto tensor = [&](size_t j) -> const V & { return std::get<V>(input(j)); };
    auto weight = [&]() -> const std::string & { return n.text("weight"); };
    Value result;
    if (n.op == "embedding") {
      auto &embedding = w(weight() + ".weight");
      auto width = n.integer("width");
      result =
          cast(resident::from_device(static_cast<uint16_t *>(embedding.data) +
                                         size_t(id) * width,
                                     width, ACL_BF16),
               ACL_FLOAT);
    } else if (n.op == "repeat") {
      V v = tensor(0);
      for (uint32_t j = 1; j < n.integer("count"); ++j)
        v = resident::concat(v, tensor(0));
      result = v;
    } else if (n.op == "mhc_pre" || n.op == "mhc_merge")
      result = pre(tensor(0), weight(), n.op == "mhc_merge", n.name);
    else if (n.op == "mhc_input")
      result = std::get<HC>(input(0)).input;
    else if (n.op == "mhc_post")
      result = post({}, tensor(1), std::get<HC>(input(0)));
    else if (n.op == "norm")
      result = norm(tensor(0), weight());
    else if (n.op == "optional_norm")
      result = bindings.count(weight() + ".weight") ? norm(tensor(0), weight())
                                                    : tensor(0);
    else if (n.op == "linear")
      result = linear(weight(), tensor(0));
    else if (n.op == "gather")
      result = gather(tensor(0));
    else if (n.op == "allreduce")
      result = sum(tensor(0));
    else if (n.op == "mome")
      result = mome(tensor(0), weight(), pos, n.name);
    else if (n.op == "slice") {
      auto start = n.integer("start"), width = n.integer("width");
      if (size_t(start) + width > tensor(0).size())
        throw std::runtime_error("slice bounds");
      result = resident::slice(tensor(0), start, width);
    } else if (n.op == "rope") {
      result = rope(tensor(0), pos);
    } else if (n.op == "concat") {
      result = resident::concat(tensor(0), tensor(1));
    } else if (n.op == "mla_query")
      result = absorb_query(tensor(0), weight(), pos);
    else if (n.op == "mla_value")
      result = expand_value(tensor(0), weight());
    else if (n.op == "sink_normalize")
      result = normalize_sinks(n.name, weight(), n.text("norm"));
    else if (n.op == "cache_append") {
      append_cache(tensor(0), tensor(1), n.name, pos);
      result = n.name;
    } else if (n.op == "latent_attention")
      result = attend(tensor(0), std::get<std::string>(input(1)));
    else if (n.op == "swiglu")
      result = activate(tensor(0), tensor(1));
    else if (n.op == "router")
      result = route(tensor(0), weight(), n.name);
    else if (n.op == "expert_linear") {
      const auto &r = std::get<Routes>(input(1));
      auto indices = local_route_ids(r);
      auto &b = bank(weight());
      ExpertRows out;
      out.rows.reserve(r.ids.size());
      for (size_t j = 0; j < r.ids.size(); ++j)
        out.rows.push_back(
            selected_linear(tensor(0), resident::slice(indices, j, 1), b));
      result = std::move(out);
    } else if (n.op == "expert_swiglu_down") {
      const auto &r = std::get<Routes>(input(2));
      auto indices = local_route_ids(r);
      auto &b = bank(weight());
      const auto &gate = std::get<ExpertRows>(input(0)),
                 &up = std::get<ExpertRows>(input(1));
      ExpertRows out;
      out.rows.reserve(r.ids.size());
      for (size_t j = 0; j < r.ids.size(); ++j)
        out.rows.push_back(
            selected_linear(resident::concat(gate.rows.at(j), up.rows.at(j)),
                            resident::slice(indices, j, 1), b, true));
      result = std::move(out);
    } else if (n.op == "expert_reduce") {
      const auto &r = std::get<Routes>(input(1));
      const auto &rows = std::get<ExpertRows>(input(0));
      auto owners = select(route_owner_table, r.ids, 256, 1);
      auto scores = binary(r.scores, owners, '*');
      V out(n.width);
      checked(aclrtMemset(out.data(), out.bytes(), 0, out.bytes()),
              "resident reduction zero");
      for (size_t j = 0; j < r.ids.size(); ++j)
        out = binary(
            out, binary(rows.rows.at(j), resident::slice(scores, j, 1), '*'),
            '+');
      result = out;
    } else if (n.op == "add") {
      result = rounded(binary(tensor(0), tensor(1), '+'));
    } else
      throw std::runtime_error("unregistered token kernel " + n.op);
    if (result.index() != n.kind)
      throw std::runtime_error("kernel result type " + n.name);
    if (auto v = std::get_if<V>(&result)) {
      if (v->size() != n.width)
        throw std::runtime_error("kernel result shape " + n.name);
      trace(n.name, *v);
    }
    values[i] = std::make_unique<Value>(std::move(result));
    for (auto j : n.inputs)
      if (--remaining[j] == 0)
        values[j].reset();
  }
  return std::get<V>(std::move(*values[program.output]));
}

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
      std::vector<uint16_t> b(width);
      checked(aclrtMemcpy(b.data(), width * 2,
                          static_cast<uint16_t *>(embedding.data) +
                              size_t(id) * width,
                          width * 2, ACL_MEMCPY_DEVICE_TO_HOST),
              "embedding row");
      result = unpacked(b);
    } else if (n.op == "repeat") {
      V v;
      for (uint32_t j = 0; j < n.integer("count"); ++j)
        v.insert(v.end(), tensor(0).begin(), tensor(0).end());
      result = std::move(v);
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
      result = V(tensor(0).begin() + start, tensor(0).begin() + start + width);
    } else if (n.op == "rope") {
      V v = tensor(0);
      rope(v, pos);
      result = std::move(v);
    } else if (n.op == "concat") {
      V v = tensor(0);
      v.insert(v.end(), tensor(1).begin(), tensor(1).end());
      result = std::move(v);
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
      ExpertRows out;
      out.rows.resize(r.ids.size());
      for (size_t j = 0; j < r.ids.size(); ++j)
        if (r.ids[j] / 32 == rank)
          out.rows[j] = linear(expert_name(weight(), r.ids[j]), tensor(0));
      result = std::move(out);
    } else if (n.op == "expert_swiglu_down") {
      const auto &r = std::get<Routes>(input(2));
      const auto &gate = std::get<ExpertRows>(input(0));
      const auto &up = std::get<ExpertRows>(input(1));
      ExpertRows out;
      out.rows.resize(r.ids.size());
      for (size_t j = 0; j < r.ids.size(); ++j)
        if (r.ids[j] / 32 == rank)
          out.rows[j] = expert_down(gate.rows.at(j), up.rows.at(j),
                                    expert_name(weight(), r.ids[j]));
      result = std::move(out);
    } else if (n.op == "expert_reduce") {
      const auto &r = std::get<Routes>(input(1));
      const auto &rows = std::get<ExpertRows>(input(0));
      V out(n.width, 0);
      for (size_t j = 0; j < r.ids.size(); ++j)
        if (r.ids[j] / 32 == rank)
          for (size_t k = 0; k < out.size(); ++k)
            out[k] += r.scores[j] * rows.rows.at(j).at(k);
      result = std::move(out);
    } else if (n.op == "add") {
      V out = tensor(0);
      for (size_t j = 0; j < out.size(); ++j)
        out[j] = roundbf(out[j] + tensor(1).at(j));
      result = std::move(out);
    } else
      throw std::runtime_error("unregistered token kernel " + n.op);
    if (result.index() != n.kind)
      throw std::runtime_error("kernel result type " + n.name);
    if (auto v = std::get_if<V>(&result)) {
      if (v->size() != n.width)
        throw std::runtime_error("kernel result shape " + n.name);
      for (float x : *v)
        if (!std::isfinite(x))
          throw std::runtime_error("nonfinite node " + n.name);
      trace(n.name, *v);
    }
    values[i] = std::make_unique<Value>(std::move(result));
    for (auto j : n.inputs)
      if (--remaining[j] == 0)
        values[j].reset();
  }
  return std::get<V>(std::move(*values[program.output]));
}

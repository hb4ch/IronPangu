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
V execute_token() {
  int pos = current_position;
  std::vector<std::unique_ptr<Value>> values(program.nodes.size());
  auto remaining = program.uses;
  for (size_t i = 0; i < program.nodes.size(); ++i) {
    const auto &n = program.nodes[i];
    inferfabric::profiling::Scope timing(n.op.c_str(), true);
    auto input = [&](size_t j) -> Value & {
      return *values.at(n.inputs.at(j));
    };
    auto tensor = [&](size_t j) -> const V & { return std::get<V>(input(j)); };
    auto weight = [&]() -> const std::string & { return n.text("weight"); };
    Value result;
    if (n.op == "embedding") {
      auto &embedding = w(weight() + ".weight");
      auto width = n.integer("width");
      DeviceTensor output({int64_t(width)}, ACL_BF16, 2);
      View source(embedding.data,
                  {int64_t(embedding.bytes / (2 * width)), int64_t(width)},
                  ACL_BF16);
      View destination(output.data, {1, int64_t(width)}, ACL_BF16);
      uint64_t bytes = 0;
      aclOpExecutor *executor = nullptr;
      checked(prepared::query("aclnnIndexSelect", false,
                              aclnnIndexSelectGetWorkspaceSize, source.t, 0,
                              token_index->tensor, destination.t, &bytes,
                              &executor),
              "embedding device index prepare");
      run(aclnnIndexSelect, bytes, executor);
      result = this->result(output);
    } else if (n.op == "repeat") {
      result = resident::repeat(tensor(0), n.integer("count"));
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
      auto batch = resident::repeat(cast(tensor(0), ACL_BF16), r.ids.size());
      result = grouped_linear(batch, r, bank(weight()));
    } else if (n.op == "expert_swiglu_down") {
      const auto &r = std::get<Routes>(input(2));
      const auto &gate = std::get<ExpertRows>(input(0)),
                 &up = std::get<ExpertRows>(input(1));
      std::vector<V> parts;
      for (size_t j = 0; j < r.ids.size(); ++j) {
        parts.push_back(gate.rows.at(j));
        parts.push_back(up.rows.at(j));
      }
      result = grouped_linear(resident::concat(parts), r, bank(weight()), true);
    } else if (n.op == "expert_reduce") {
      const auto &r = std::get<Routes>(input(1));
      const auto &rows = std::get<ExpertRows>(input(0));
      auto owners =
          select(route_owner_table, select(r.ids, r.order, 8, 1), 256, 1);
      auto scores = binary(select(r.scores, r.order, 8, 1), owners, '*');
      result = weighted_rows(rows, scores, n.width);
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

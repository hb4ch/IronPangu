// Included inside the backend kernel collection. The program, not this switch,
// defines the architecture, residual wiring, layer count and weight bindings.
using Value = std::variant<V, HC, Routes, ExpertRows, std::string, Quantized, RoutedQuantized, std::monostate>;
static std::string expert_name(std::string pattern, int expert) {
  auto at = pattern.find("{expert}");
  if (at == std::string::npos)
    throw std::runtime_error("expert binding missing placeholder");
  pattern.replace(at, 8, std::to_string(expert));
  return pattern;
}
V execute_token() {
  metadata_enqueued = false;
  int pos = current_position;
  std::vector<std::unique_ptr<Value>> values(program.nodes.size());
  auto remaining = program.uses;
  for (size_t i = 0; i < program.nodes.size(); ++i) {
    const auto &n = program.nodes[i];
    execution.instruction(int64_t(i),
        n.op=="copy" || n.op=="repeat" || n.op=="concat" || n.op=="expert_pack",
        n.op=="cache_c8_norm_rope_append" || n.op=="cache_quantize_append" || n.op=="sink_normalize" || n.op=="cache_append" || n.op=="cache_norm_rope_append" || n.op=="expert_gmm" || (n.op=="linear" && n.kernel=="paired_bf16"));
    inferfabric::profiling::Scope timing(n.op.c_str(), true);
    auto input = [&](size_t j) -> Value & {
      return *values.at(n.inputs.at(j));
    };
    auto tensor = [&](size_t j) -> const V & {
      const auto &v = std::get<V>(input(j));
      if (program.physical && v.dtype() != (program.nodes[n.inputs[j]].dtype == "bf16" ? ACL_BF16 : ACL_FLOAT))
        throw std::runtime_error("physical input storage mismatch: " + n.name);
      return v;
    };
    auto weight = [&]() -> const std::string & { return n.text("weight"); };
    Value result;
    if (n.op == "cast") {
      result = cast(tensor(0), n.dtype == "bf16" ? ACL_BF16 : ACL_FLOAT,true);
    } else if (n.op == "quantize") {
      result = quantize(tensor(0));
    } else if (n.op == "copy") {
      V copy(tensor(0).size(), tensor(0).dtype());
      resident::copy(copy.data(), tensor(0).data(), copy.bytes());
      result = std::move(copy);
    } else if (n.op == "add_f32") {
      result = binary(tensor(0), tensor(1), '+');
    } else if (n.op == "embedding") {
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
    } else if(n.op=="attention_metadata_zero") {
      execution.zero(attention_metadata->data,attention_metadata->bytes);result=std::monostate{};
    } else if(n.op=="attention_metadata") {
      prepare_attention_metadata(true);result=std::monostate{};
    } else if(n.op=="mhc_pre_project") {
      result=pre(tensor(0),weight(),false,n.name,true);
    } else if(n.op=="mhc_sinkhorn") {
      result=sinkhorn(std::get<HC>(input(0)));
    } else if(n.op=="router_projection") {
      result=project_route(tensor(0),weight(),n.name);
    } else if(n.op=="router_select") {
      result=select_route(tensor(0),program.nodes[n.inputs[0]].name);
    } else if(n.op=="route_quantize") {
      result=quantize_experts(tensor(0),std::get<Routes>(input(1)),n.width);
    } else if(n.op=="expert_act_quantize") {
      const auto &rows=std::get<ExpertRows>(input(0));
      result=quantize_experts(rows.packed,Routes{},n.width,true,&rows);
    } else if(n.op=="expert_pack") {
      result=pack_expert_rows(std::get<ExpertRows>(input(0)),std::get<ExpertRows>(input(1)));
    } else if(n.op=="expert_zero") {
      result=zero_expert_output(std::get<RoutedQuantized>(input(0)),n.width);
    } else if(n.op=="expert_gmm") {
      result=expert_gmm(std::get<RoutedQuantized>(input(0)),std::get<ExpertRows>(input(1)),bank(weight(),n.attrs.count("paired_weight")?n.text("paired_weight"):""));
    } else if(n.op=="expert_values") {
      result=std::get<ExpertRows>(input(0)).packed;
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
      result = (program.physical ? n.kernel == "norm" : bindings.count(weight() + ".weight")) ? norm(tensor(0), weight())
                                                    : tensor(0);
    else if (n.op == "linear" && program.physical && n.kernel=="int8")
      result = quantized_linear(weight(),std::get<Quantized>(input(0)));
    else if (n.op == "linear")
      result = n.attrs.count("paired_weight")
                   ? linear_pair(weight(), n.text("paired_weight"), tensor(0))
                   : linear(weight(), tensor(0), program.physical ? n.kernel : "");
    else if (n.op == "gather")
      result = gather(tensor(0));
    else if (n.op == "allreduce")
      result = sum(tensor(0), n.attrs.count("dtype") && n.text("dtype") == "bf16");
    else if (n.op == "mome" || n.op == "mome_inplace")
      result = mome(tensor(0), weight(), pos, n.name, n.op == "mome_inplace");
    else if (n.op == "slice") {
      auto start = n.integer("start"), width = n.integer("width");
      if (size_t(start) + width > tensor(0).size())
        throw std::runtime_error("slice bounds");
      result = resident::slice(tensor(0), start, width);
    } else if (n.op == "query_project") {
      result=query_project(tensor(0),weight(),n.text("input_layout")=="rank_local");
    } else if (n.op == "query_rotary_extract") {
      result=query_rotary_extract(tensor(0),n.text("input_layout")=="rank_local");
    } else if (n.op == "concat_columns") {
      result=concat_columns(tensor(0),tensor(1),n.integer("rows"));
    } else if (n.op == "rope_f32") {
      result=rope(tensor(0),pos,true);
    } else if (n.op == "rope") {
      result = rope(tensor(0), pos);
    } else if (n.op == "concat") {
      result = resident::concat(tensor(0), tensor(1));
    } else if (n.op == "mla_query")
      result = absorb_query(tensor(0), weight(), pos,
                            n.attrs.count("input_layout") &&
                                n.text("input_layout") == "rank_local");
    else if (n.op == "mla_value")
      result = expand_value(tensor(0), weight());
    else if (n.op == "sink_normalize")
      result = normalize_sinks(n.name, weight(), n.text("norm"));
    else if (n.op == "cache_append") {
      append_cache(tensor(0), tensor(1), n.name, pos);
      result = n.name;
    } else if(n.op=="cache_c8_norm_rope_append") {
      append_normalized_cache(tensor(0),tensor(1),weight(),n.name,pos,true);
      result=n.name;
    } else if(n.op=="packed_attention_metadata") {
      prepare_packed_attention_metadata();result=std::monostate{};
    } else if(n.op=="packed_latent_attention") {
      result=attend_packed(tensor(0),std::get<std::string>(input(1)));
    } else if(n.op=="cache_normalize") {
      result=normalize_cache_row(tensor(0),weight());
    } else if(n.op=="cache_quantize_append") {
      quantize_cache(tensor(0),tensor(1),n.name,n.text("cache_dtype")=="int8"?8:4);
      result=n.name;
    } else if(n.op=="cache_dequantize") {
      result=dequantize_cache(std::get<std::string>(input(0)));
    } else if (n.op == "cache_norm_rope_append") {
      if(program.physical) append_normalized_cache(tensor(0),tensor(1),weight(),n.name,pos);
      else append_normalized_cache(resident::concat(tensor(0),tensor(1)),tensor(2),weight(),n.name,pos);
      result = n.name;
    } else if (n.op == "latent_attention")
      result = attend(tensor(0), std::get<std::string>(input(1)),n.inputs.size()==4?&tensor(3):nullptr);
    else if (n.op == "swiglu")
      result = n.inputs.size() == 1 ? activate(tensor(0)) : activate(tensor(0), tensor(1));
    else if (n.op == "router")
      result = route(tensor(0), weight(), n.name);
    else if (n.op == "expert_linear") {
      const auto &r = std::get<Routes>(input(1));
      auto batch = cast(tensor(0), ACL_BF16);
      result = grouped_linear(batch, r,
                              bank(weight(), n.attrs.count("paired_weight")
                                                 ? n.text("paired_weight")
                                                 : ""));
    } else if (n.op == "expert_swiglu_down") {
      if (n.inputs.size() == 2) {
        const auto &r = std::get<Routes>(input(1));
        const auto &packed = std::get<ExpertRows>(input(0));
        result = grouped_linear(packed.packed, r, bank(weight()), true, &packed);
      } else {
        const auto &r = std::get<Routes>(input(2));
        const auto &gate = std::get<ExpertRows>(input(0)),
                   &up = std::get<ExpertRows>(input(1));
        std::vector<V> parts;
        for (size_t j = 0; j < r.ids.size(); ++j) {
          parts.push_back(gate.rows.at(j));
          parts.push_back(up.rows.at(j));
        }
        result =
            grouped_linear(resident::concat(parts), r, bank(weight()), true, &gate);
      }
    } else if (n.op == "expert_reduce") {
      const auto &r = std::get<Routes>(input(program.physical?2:1));
      const auto &rows = std::get<ExpertRows>(input(program.physical?1:0));
      result = weighted_rows(rows, r.scores, n.width,program.physical?&tensor(0):nullptr);
    } else if (n.op == "add") {
      result = rounded(binary(tensor(0), tensor(1), '+'));
    } else
      throw std::runtime_error("unregistered token kernel " + n.op);
    if (result.index() != n.kind)
      throw std::runtime_error("kernel result type " + n.name);
    if (auto v = std::get_if<V>(&result)) {
      if (program.physical && v->dtype() != (n.dtype == "bf16" ? ACL_BF16 : ACL_FLOAT))
        throw std::runtime_error("physical output storage mismatch: " + n.name);
      if (v->size() != n.width)
        throw std::runtime_error("kernel result shape " + n.name);
      trace(n.name, *v);
    }
    values[i] = std::make_unique<Value>(std::move(result));
    for (auto j : n.inputs)
      if (--remaining[j] == 0)
        values[j].reset();
  }
  execution.instruction(-1);
  return std::get<V>(std::move(*values[program.output]));
}

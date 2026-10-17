// Included inside the backend kernel collection. The program, not this switch,
// defines the architecture, residual wiring, layer count and weight bindings.
struct SelectedIndices { V value; };
struct IndexCacheRef { std::string name; };
static aclDataType storage_dtype(const std::string &dtype) {
  if(dtype=="bf16")return ACL_BF16;
  if(dtype=="f16")return ACL_FLOAT16;
  if(dtype=="f32")return ACL_FLOAT;
  throw std::runtime_error("unsupported physical tensor dtype");
}
using Value = std::variant<V, HC, Routes, ExpertRows, std::string, Quantized, RoutedQuantized, std::monostate, SelectedIndices, IndexCacheRef>;
static std::string expert_name(std::string pattern, int expert) {
  auto at = pattern.find("{expert}");
  if (at == std::string::npos)
    throw std::runtime_error("expert binding missing placeholder");
  pattern.replace(at, 8, std::to_string(expert));
  return pattern;
}
V execute_token() {
  metadata_enqueued.clear();
  std::vector<std::unique_ptr<Value>> values(program.nodes.size());
  auto remaining = program.uses;
  for (size_t i = 0; i < program.nodes.size(); ++i) {
    const auto &n = program.nodes[i];
    execution.instruction(int64_t(i),
        n.op=="copy" || n.op=="repeat" || n.op=="concat" || n.op=="expert_pack" ||
        (active_tokens>1 && (n.op=="gather" || n.op=="slice" || n.op=="query_project" || n.op=="query_rotary_extract" || n.op=="rope_bf16" || n.op=="rope_f32" || n.op=="rope" || n.op=="mla_query" || n.op=="mla_value" || n.op=="mome" || (n.op=="sparse_index" && program.nodes[n.inputs[0]].kind!=5))),
        n.op=="cache_c8_norm_rope_append" || n.op=="cache_quantize_append" || n.op=="sink_normalize" || n.op=="cache_append" || n.op=="cache_norm_rope_append" || n.op=="expert_gmm" || (n.op=="linear" && n.kernel=="paired_bf16"));
    inferfabric::profiling::Scope timing(n.op.c_str(), true);
    auto input = [&](size_t j) -> Value & {
      return *values.at(n.inputs.at(j));
    };
    auto tensor = [&](size_t j) -> const V & {
      const auto &v = std::get<V>(input(j));
      if (program.physical && v.dtype() != storage_dtype(program.nodes[n.inputs[j]].dtype))
        throw std::runtime_error("physical input storage mismatch: " + n.name);
      return v;
    };
    auto weight = [&]() -> const std::string & { return n.text("weight"); };
    auto row_tensor = [&](size_t j,uint32_t row) {
      const auto &v=tensor(j);
      if(v.size()%active_tokens)throw std::runtime_error("packed input row extent: "+n.name);
      size_t width=v.size()/active_tokens;
      return v.view(row*width,width);
    };
    Value result;
    if (n.op == "cast") {
      result = cast(tensor(0), storage_dtype(n.dtype),true);
    } else if(n.op=="index_quantize") {
      result=quantize(tensor(0),n.integer("block_width"));
    } else if(n.op=="quant_scales_f16") {
      const auto &q=std::get<Quantized>(input(0));
      result=Quantized{q.values,cast(q.scales,ACL_FLOAT16,true)};
    } else if (n.op == "quantize") {
      result = quantize(tensor(0),n.width);
    } else if (n.op == "copy") {
      V copy(tensor(0).size(), tensor(0).dtype());
      resident::copy(copy.data(), tensor(0).data(), copy.bytes());
      result = std::move(copy);
    } else if (n.op == "add_f32") {
      result = binary(tensor(0), tensor(1), '+');
    } else if (n.op == "embedding") {
      auto &embedding = w(weight() + ".weight");
      auto width = n.integer("width");
      DeviceTensor output({int64_t(active_tokens),int64_t(width)}, ACL_BF16, 2);
      View source(embedding.data,
                  {int64_t(embedding.bytes / (2 * width)), int64_t(width)},
                  ACL_BF16);
      View destination(output.data, {int64_t(active_tokens), int64_t(width)}, ACL_BF16);
      uint64_t bytes = 0;
      aclOpExecutor *executor = nullptr;
      checked(prepared::query("aclnnIndexSelect", false,
                              aclnnIndexSelectGetWorkspaceSize, source.t, 0,
                              prefix(*token_index,{int64_t(active_tokens)}), destination.t, &bytes,
                              &executor),
              "embedding device index prepare");
      run(aclnnIndexSelect, bytes, executor);
      result = this->result(output);
    } else if(n.op=="attention_metadata_zero") {
      auto &metadata=*attention_metadata.at(n.integer("window"));
      execution.zero(metadata.data,metadata.bytes);result=std::monostate{};
    } else if(n.op=="attention_metadata") {
      prepare_attention_metadata(n.integer("window"),true);result=std::monostate{};
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
      result = row_values(n.width,[&](uint32_t row){return resident::repeat(row_tensor(0,row), n.integer("count"));});
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
    else if (n.op == "mome" || n.op == "mome_inplace") {
      if(n.op=="mome_inplace") {
        each_row([&](uint32_t row){mome(row_tensor(0,row),weight(),current_position,n.name,true);});
        result=tensor(0);
      } else result=row_values(n.width,[&](uint32_t row){return mome(row_tensor(0,row),weight(),current_position,n.name,false);});
    }
    else if (n.op == "slice") {
      auto start = n.integer("start"), width = n.integer("width");
      if (size_t(start) + width > tensor(0).size()/active_tokens)
        throw std::runtime_error("slice bounds");
      result = row_values(n.width,[&](uint32_t row){return resident::slice(row_tensor(0,row), start, width);});
    } else if (n.op == "query_project") {
      result=row_values(n.width,[&](uint32_t row){return query_project(row_tensor(0,row),weight(),n.text("input_layout")=="rank_local");});
    } else if (n.op == "query_rotary_extract") {
      result=row_values(n.width,[&](uint32_t row){return query_rotary_extract(row_tensor(0,row),n.text("input_layout")=="rank_local");});
    } else if (n.op == "column_slice") {
      result=column_slice(tensor(0),n.integer("rows")*active_tokens,n.integer("start"),n.integer("width"));
    } else if (n.op == "rope_bf16") {
      result=row_values(n.width,[&](uint32_t row){return rope_bf16(row_tensor(0,row));});
    } else if (n.op == "concat_columns") {
      result=concat_columns(tensor(0),tensor(1),n.integer("rows")*active_tokens);
    } else if (n.op == "rope_f32") {
      result=row_values(n.width,[&](uint32_t row){return rope(row_tensor(0,row),current_position,true);});
    } else if (n.op == "rope") {
      result=row_values(n.width,[&](uint32_t row){return rope(row_tensor(0,row),current_position);});
    } else if (n.op == "concat") {
      result = active_tokens==1 ? resident::concat(tensor(0),tensor(1)) : concat_columns(tensor(0),tensor(1),active_tokens);
    } else if (n.op == "mla_query")
      result = row_values(n.width,[&](uint32_t row){return absorb_query(row_tensor(0,row), weight(), current_position,
                            n.attrs.count("input_layout") &&
                                n.text("input_layout") == "rank_local");});
    else if (n.op == "mla_value")
      result = row_values(n.width,[&](uint32_t row){return expand_value(row_tensor(0,row), weight());});
    else if (n.op == "sink_normalize")
      result = normalize_sinks(n.name, weight(), n.text("norm"));
    else if (n.op == "cache_append") {
      each_row([&](uint32_t row){append_cache(row_tensor(0,row), tensor(1), n.name, current_position);});
      result = n.name;
    } else if(n.op=="cache_c8_norm_rope_append") {
      each_row([&](uint32_t row){append_normalized_cache(row_tensor(0,row),tensor(1),weight(),n.name,current_position,true);});
      result=n.name;
    } else if(n.op=="index_cache_append") {
      auto &state=index_states[n.name];
      bool quant=n.attrs.count("cache_dtype") && n.text("cache_dtype")=="int8";
      if(!state)state=std::make_unique<resident::IndexState>(((program.context+63)/64)*64,program.resident_sequences,program.shared_pages,quant,program.prefill_tokens);
      each_row([&](uint32_t row){
        if(quant){const auto &q=std::get<Quantized>(input(0));state->append_quantized(q.values.view(row*128,128),q.scales.view(row,1),cache_append_descriptor(n.name,ACL_INT32));}
        else state->append(row_tensor(0,row),cache_append_descriptor(n.name,ACL_INT32));
      });
      result=IndexCacheRef{n.name};
    } else if(n.op=="sparse_index") {
      auto &state=*index_states.at(std::get<IndexCacheRef>(input(2)).name);
      if(state.quantized) {
        const auto &q=std::get<Quantized>(input(0));
        // TND causal alignment uses the final KV length and total query count.
        // All chunk keys have already been appended on this stream.
        result=SelectedIndices{state.select_quantized(q.values,q.scales,tensor(1),
            prefix(*packed_length,{1},(active_tokens-1)*sizeof(int32_t)),active_tokens)};
      } else result=SelectedIndices{row_values(n.width,[&](uint32_t row){
        return state.select_indices(row_tensor(0,row),row_tensor(1,row),control_row(*packed_length));
      })};
    } else if(n.op=="indexed_packed_attention") {
      const auto &indices=std::get<SelectedIndices>(input(2)).value;
      result=attend_packed(tensor(0),std::get<std::string>(input(1)),&indices);
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
      each_row([&](uint32_t row){
        if(program.physical) append_normalized_cache(row_tensor(0,row),tensor(1),weight(),n.name,current_position);
        else append_normalized_cache(resident::concat(row_tensor(0,row),row_tensor(1,row)),tensor(2),weight(),n.name,current_position);
      });
      result = n.name;
    } else if (n.op == "latent_attention")
      result = attend(tensor(0),std::get<std::string>(input(1)),n.integer("window"),n.inputs.size()==4?&tensor(3):nullptr);
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
      if (program.physical && v->dtype() != storage_dtype(n.dtype))
        throw std::runtime_error("physical output storage mismatch: " + n.name);
      if (v->size() != n.width*(n.op=="sink_normalize" ? 1 : active_tokens))
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

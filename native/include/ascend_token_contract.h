#pragma once
#include "initialization_plan.h"
#include "token_execution_plan.h"
#include <algorithm>
namespace inferfabric::execution {
inline bool exclusive_region(const Plan &p, uint32_t input, size_t owner) {
  struct Region { uint32_t root, start, width; bool fresh; };
  auto region = [&](uint32_t id) {
    uint32_t width = p.nodes[id].width, start = 0;
    while (p.nodes[id].op == "slice") {
      start += p.nodes[id].integer("start"); id = p.nodes[id].inputs[0];
    }
    auto &op = p.nodes[id].op;
    return Region{id, start, width, op == "linear" || op == "norm" || op == "mla_value" || op == "copy"};
  };
  auto a = region(input);
  if (!a.fresh) return false;
  auto overlap = [&](uint32_t id) { auto b=region(id); return b.root==a.root && b.start<a.start+a.width && a.start<b.start+b.width; };
  if (overlap(p.output)) return false;
  for (size_t j=0;j<p.nodes.size();++j) if (j != owner && p.nodes[j].op != "slice")
    for (auto id : p.nodes[j].inputs) if (overlap(id)) return false;
  return true;
}
inline uint32_t route_origin(const Plan &p,uint32_t id) {
  for(size_t depth=0;depth<p.nodes.size();++depth) {
    const auto &n=p.nodes.at(id);
    if(n.op=="route_quantize" && n.inputs.size()==2)return n.inputs[1];
    if((n.op=="expert_gmm" || n.op=="expert_act_quantize" || n.op=="expert_pack") && !n.inputs.empty()) {id=n.inputs[0];continue;}
    throw std::runtime_error("expert route provenance");
  }
  throw std::runtime_error("expert provenance cycle");
}
// Specialization capabilities of the registered Ascend kernels. This validates
// serialized IR independently of Rust before allocating model/device memory.
inline void validate_kernels(const Plan &p) {
  using T = std::pair<uint32_t, uint32_t>;
  if(p.physical) {
    auto count=[&](const char *op){return std::count_if(p.nodes.begin(),p.nodes.end(),[&](const auto &n){return n.op==op;});};
    if((count("packed_latent_attention") || count("sparse_index") || count("indexed_packed_attention")) && count("packed_attention_metadata")!=1)
      throw std::runtime_error("packed attention metadata must have one declared writer");
    std::set<uint32_t> windows,zeros;
    for(const auto &node:p.nodes) {
      if(node.op=="attention_metadata" && !windows.insert(node.integer("window")).second)
        throw std::runtime_error("attention metadata has duplicate window writer");
      if(node.op=="attention_metadata_zero" && !zeros.insert(node.integer("window")).second)
        throw std::runtime_error("attention metadata has duplicate window initialization");
    }
    if(windows!=zeros || (count("latent_attention") && windows.empty()))
      throw std::runtime_error("attention metadata window writers missing");
  }

  for (size_t i = 0; i < p.nodes.size(); ++i) {
    const auto &n = p.nodes[i];
    std::vector<T> in;
    for (auto j : n.inputs)
      in.emplace_back(p.nodes[j].kind, p.nodes[j].width);
    auto error = [&]() {
      throw std::runtime_error("invalid kernel contract: " + n.name + " (" +
                               n.op + ")");
    };
    auto check = [&](std::vector<T> expected, std::vector<std::string> attrs,
                     T out) {
      if (in != expected || T{n.kind, n.width} != out ||
          n.attrs.size() != attrs.size())
        error();
      for (auto &k : attrs)
        if (!n.attrs.count(k))
          error();
    };
    auto eq = [&](const char *k, const char *v) {
      if (n.text(k) != v)
        error();
    };
    auto tensor = [&](size_t j) {
      if (j >= in.size() || in[j].first != 0)
        error();
      return in.at(j).second;
    };
    auto w = [&]() {
      auto width = n.integer("width");
      if (!width || width > 200000)
        error();
      return width;
    };
    if (n.op == "cast") {
      if (!p.physical) error();
      check({{0, tensor(0)}}, {"dtype"}, {0, tensor(0)});
      if (n.text("dtype") != n.dtype || n.kernel != "aclnnCast" ||
          p.nodes[n.inputs[0]].dtype == n.dtype) error();
    } else if(p.physical && n.op=="index_quantize") {
      check({{0,tensor(0)}},{"block_width"},{5,tensor(0)});eq("block_width","128");
      if(n.width!=128 && n.width!=3072)error();
    } else if(p.physical && n.op=="quant_scales_f16") {
      check({{5,n.width}},{},{5,n.width});
      if(p.nodes[n.inputs[0]].op!="index_quantize")error();
    } else if (p.physical && n.op == "quantize") {
      check({{0,tensor(0)}},{"dtype","scheme","scale_dtype"},{5,tensor(0)});
      eq("dtype","int8"); eq("scheme","per_token_symmetric"); eq("scale_dtype","f32");
    } else if(p.physical && n.op=="mhc_sinkhorn") {
      check({{1,0}},{},{1,0});
      if(p.nodes[n.inputs[0]].op!="mhc_pre_project")error();
    } else if(p.physical && n.op=="router_select") {
      check({{0,256}},{},{2,0});
      if(p.nodes[n.inputs[0]].op!="router_projection")error();
    } else if(p.physical && n.op=="route_quantize") {
      check({{0,2560},{2,0}},{},{6,2560});
    } else if(p.physical && n.op=="expert_act_quantize") {
      check({{3,2048}},{},{6,1024});
      route_origin(p,n.inputs[0]);
    } else if(p.physical && n.op=="expert_pack") {
      check({{3,1024},{3,1024}},{},{3,2048});
      if(route_origin(p,n.inputs[0])!=route_origin(p,n.inputs[1]))error();
    } else if(p.physical && n.op=="expert_zero") {
      if(in.size()!=1 || in[0].first!=6 || (w()!=1024 && w()!=2048 && w()!=2560))error();
      check({in[0]},{"width"},{3,w()});
    } else if(p.physical && n.op=="expert_gmm") {
      if(in.size()!=2 || in[0].first!=6)error();
      auto output=w()*(n.attrs.count("paired_weight")?2:1);
      std::vector<std::string> attrs{"weight","width"};if(n.attrs.count("paired_weight"))attrs.push_back("paired_weight");
      check({{6,in[0].second},{3,output}},attrs,{3,output});
      auto &zero=p.nodes[n.inputs[1]];
      if(zero.op!="expert_zero" || zero.inputs!=std::vector<uint32_t>{n.inputs[0]} || p.output==n.inputs[1])error();
      size_t uses=0;for(auto &consumer:p.nodes)for(auto input:consumer.inputs)uses+=input==n.inputs[1];
      if(uses!=1)error();
      route_origin(p,n.inputs[0]);
    } else if(p.physical && n.op=="expert_values") {
      if(in.size()!=1 || in[0].first!=3)error();
      check({in[0]},{},{0,in[0].second*8});
    } else if(p.physical && n.op=="attention_metadata_zero") {
      check({},{"window"},{7,0});
      if(!n.integer("window"))error();
    } else if(p.physical && n.op=="attention_metadata") {
      check({{7,0}},{"window"},{7,0});
      if(p.nodes[n.inputs[0]].op!="attention_metadata_zero" || p.nodes[n.inputs[0]].integer("window")!=n.integer("window"))error();
    } else if (n.op == "embedding") {
      check({}, {"weight", "width", "vocab"}, {0, 2560});
      eq("width", "2560");
      eq("vocab", "151552");
    } else if (n.op == "repeat") {
      auto a = tensor(0), c = n.integer("count");
      if (!c || c > 16)
        error();
      check({{0, a}}, {"count"}, {0, a * c});
    } else if (n.op == "mhc_pre" || n.op == "mhc_merge" || n.op == "mhc_pre_project") {
      check({{0, 10240}}, {"weight", "streams", "sinkhorn", "eps"}, {1, 0});
      eq("streams", "4");
      eq("sinkhorn", "20");
      eq("eps", "1e-6");
    } else if (n.op == "mhc_input")
      check({{1, 0}}, {}, {0, 2560});
    else if (n.op == "mhc_post")
      check({{1, 0}, {0, 2560}}, {}, {0, 10240});
    else if (n.op == "norm" || n.op == "optional_norm") {
      auto a = tensor(0);
      check({{0, a}}, {"weight", "eps"}, {0, a});
      eq("eps", "1e-5");
    } else if (n.op == "linear") {
      if (n.attrs.count("paired_weight")) {
        auto first = w(), second = n.integer("paired_width");
        if (!second || second > 200000 - first || n.text("placement") != "replicated") error();
        check({{0, tensor(0)}}, {"weight", "width", "placement", "paired_weight", "paired_width"}, {0, first + second});
      } else {
        if (p.physical && n.kernel=="int8") {
          if(in.size()!=1 || in[0].first!=5) error();
          check({{5,in[0].second}}, {"weight", "width", "placement"}, {0,w()});
        } else check({{0, tensor(0)}}, {"weight", "width", "placement"}, {0, w()});
      }
      if (n.text("placement") != "tp_rows" &&
          n.text("placement") != "tp_columns" &&
          n.text("placement") != "replicated")
        error();
    } else if (n.op == "gather")
      check({{0, tensor(0)}}, {}, {0, tensor(0) * 8});
    else if (n.op == "allreduce") {
      std::vector<std::string> attrs;
      if (n.attrs.count("dtype")) {
        attrs.push_back("dtype");
        if (n.text("dtype") != "f32" && n.text("dtype") != "bf16") error();
      }
      check({{0, tensor(0)}}, attrs, {0, tensor(0)});
    }
    else if (p.physical && n.op == "copy") {
      check({{0, tensor(0)}}, {}, {0, tensor(0)});
    } else if (n.op == "mome" || (p.physical && n.op == "mome_inplace")) {
      std::vector<std::string> attrs{"weight", "taps"};
      if (n.attrs.count("placement")) {
        attrs.push_back("placement");
        if (n.text("placement") != "tp_channels" &&
            n.text("placement") != "replicated")
          error();
      }
      check({{0, tensor(0)}}, attrs, {0, tensor(0)});
      eq("taps", "3");
      if (n.op == "mome_inplace") {
        if (!exclusive_region(p, n.inputs[0], i)) error();
      }
    } else if (n.op == "slice") {
      auto a = tensor(0);
      if (uint64_t(n.integer("start")) + w() > a)
        error();
      check({{0, a}}, {"start", "width"}, {0, w()});
    } else if (p.physical && (n.op == "query_project" || n.op == "query_rotary_extract")) {
      auto layout=n.text("input_layout");
      if(layout!="rank_local" && layout!="replicated") error();
      std::vector<std::string> attrs{"input_layout"};
      if(n.op=="query_project") attrs.push_back("weight");
      check({{0,layout=="rank_local" ? 1152u : 9216u}},attrs,{0,n.op=="query_project" ? 3072u : 384u});
    } else if (p.physical && n.op == "rope_f32") {
      if(tensor(0)!=64 && tensor(0)!=384) error();
      check({{0,tensor(0)}},{"theta"},{0,tensor(0)});eq("theta","6400000");
    } else if (p.physical && n.op == "column_slice") {
      auto rows=n.integer("rows"), width=n.integer("width"), start=n.integer("start");
      if(!rows || rows>64 || tensor(0)%rows || !width || start>tensor(0)/rows || width>tensor(0)/rows-start)error();
      check({{0,tensor(0)}},{"rows","start","width"},{0,rows*width});
    } else if (p.physical && n.op == "rope_bf16") {
      if(!tensor(0) || tensor(0)%64 || tensor(0)>64*64)error();
      check({{0,tensor(0)}},{"theta"},{0,tensor(0)});eq("theta","6400000");
    } else if (p.physical && n.op == "concat_columns") {
      auto rows=n.integer("rows");
      if(!rows || rows>64 || tensor(0)%rows || tensor(1)%rows)error();
      check({{0,tensor(0)},{0,tensor(1)}},{"rows"},{0,tensor(0)+tensor(1)});
    } else if (n.op == "rope") {
      check({{0, 64}}, {"theta"}, {0, 64});
      eq("theta", "6400000");
    } else if (n.op == "concat")
      check({{0, tensor(0)}, {0, tensor(1)}}, {}, {0, tensor(0) + tensor(1)});
    else if (n.op == "mla_query") {
      std::vector<std::string> attrs{"weight", "theta", "heads",
                                     "nope",   "rope",  "latent"};
      uint32_t width = 9216;
      if (n.attrs.count("input_layout")) {
        attrs.push_back("input_layout");
        auto layout = n.text("input_layout");
        if (layout == "rank_local")
          width = 1152;
        else if (layout != "replicated")
          error();
      }
      check({{0, width}}, attrs, {0, 3456});
      eq("theta", "6400000");
      eq("heads", "48");
      eq("nope", "128");
      eq("rope", "64");
      eq("latent", "512");
    } else if (n.op == "mla_value") {
      check({{0, 3072}}, {"weight", "heads", "latent", "value"}, {0, 768});
      eq("heads", "48");
      eq("latent", "512");
      eq("value", "128");
    } else if (n.op == "sink_normalize") {
      check({}, {"weight", "norm", "count", "latent", "rope", "eps"},
            {0, 73728});
      eq("count", "128");
      eq("latent", "512");
      eq("rope", "64");
      eq("eps", "1e-5");
    } else if (n.op == "cache_append") {
      std::vector<std::string> attrs;
      if (n.attrs.count("cache_dtype")) { attrs.push_back("cache_dtype");eq("cache_dtype","bf16"); }
      check({{0, 576}, {0, 73728}}, attrs, {4, 0});
    }
    else if(p.physical && n.op=="cache_c8_norm_rope_append") {
      check({{0,576},{0,73728}},{"weight","eps","theta","cache_dtype","cache_scale_dtype","cache_quantization","cache_rope_dtype"},{4,0});
      eq("eps","1e-5");eq("theta","6400000");eq("cache_dtype","int8_ds_mla");
      eq("cache_scale_dtype","f32");eq("cache_quantization","pertile128");eq("cache_rope_dtype","bf16");
    } else if(p.physical && n.op=="index_cache_append") {
      bool quant=n.attrs.count("cache_dtype") && n.text("cache_dtype")=="int8";
      std::vector<std::string> attrs{"width","page_tokens"};
      if(n.attrs.count("cache_dtype")){attrs.push_back("cache_dtype");if(!quant)eq("cache_dtype","bf16");}
      check({{quant?5u:0u,128}},attrs,{9,128});eq("width","128");eq("page_tokens","64");
      if(quant && p.nodes[n.inputs[0]].op!="quant_scales_f16")error();
    } else if(p.physical && n.op=="sparse_index") {
      if(n.inputs.size()!=4)error();
      const auto &cache=p.nodes[n.inputs[2]];
      bool quant=cache.attrs.count("cache_dtype") && cache.text("cache_dtype")=="int8";
      check({{quant?5u:0u,3072},{0,24},{9,128},{7,0}},{"heads","head_dim","topk","scoring"},{8,2048});
      if(quant && p.nodes[n.inputs[0]].op!="quant_scales_f16")error();
      eq("heads","24");eq("head_dim","128");eq("topk","2048");eq("scoring","weighted_relu");
      if(p.nodes[n.inputs[3]].op!="packed_attention_metadata")error();
    } else if(p.physical && n.op=="indexed_packed_attention") {
      check({{0,3456},{4,0},{8,2048},{7,0}},{"scale_dim","window","selection"},{0,3072});
      eq("scale_dim","192");eq("window","2048");eq("selection","learned_indices");
      if(p.nodes[n.inputs[1]].op!="cache_c8_norm_rope_append" || p.nodes[n.inputs[3]].op!="packed_attention_metadata")error();
    } else if(p.physical && n.op=="packed_attention_metadata") {
      check({}, {}, {7,0});
    } else if(p.physical && n.op=="packed_latent_attention") {
      check({{0,3456},{4,0},{7,0}},{"scale_dim","window","selection"},{0,3072});
      eq("scale_dim","192");eq("selection","all_causal_within_window");
      if(p.context>n.integer("window") || p.nodes[n.inputs[1]].op!="cache_c8_norm_rope_append" || p.nodes[n.inputs[2]].op!="packed_attention_metadata")error();
    }
    else if (p.physical && n.op == "cache_normalize") {
      check({{0,576}}, {"weight","eps","theta"}, {0,576});
      eq("eps","1e-5");eq("theta","6400000");
    } else if(p.physical && (n.op=="cache_quantize_append" || n.op=="cache_dequantize")) {
      std::vector<std::string> attrs{"cache_dtype","cache_scale_dtype","cache_quantization"};
      if(n.op=="cache_quantize_append")check({{0,576},{0,73728}},attrs,{4,0});
      else {
        check({{4,0}},attrs,{0,((p.context+15)/16)*16*576});
        const auto &source=p.nodes[n.inputs[0]];
        if(source.op!="cache_quantize_append" || source.attrs!=n.attrs)error();
      }
      if(n.text("cache_dtype")!="int8" && n.text("cache_dtype")!="int4")error();
      eq("cache_scale_dtype","f32");eq("cache_quantization","per_token_head_symmetric");
    } else if (n.op == "cache_norm_rope_append") {
      std::vector<std::string> attrs{"weight", "eps", "theta"};
      if (n.attrs.count("cache_dtype")) { attrs.push_back("cache_dtype");eq("cache_dtype","bf16"); }
      if(p.physical) check({{0,576},{0,73728}},attrs,{4,0});
      else check({{0, 512}, {0, 64}, {0, 73728}}, attrs, {4, 0});
      eq("eps", "1e-5");
      eq("theta", "6400000");
    } else if (n.op == "latent_attention") {
      if(p.physical) {
        if(p.nodes[n.inputs.at(1)].op=="cache_quantize_append") {
          check({{0,3456},{4,0},{7,0},{0,((p.context+15)/16)*16*576}},{"scale_dim","window","selection"},{0,3072});
          const auto &decoded=p.nodes[n.inputs[3]];
          if(decoded.op!="cache_dequantize" || decoded.inputs!=std::vector<uint32_t>{n.inputs[1]})error();
        } else check({{0,3456},{4,0},{7,0}},{"scale_dim","window","selection"},{0,3072});
        if(p.nodes[n.inputs[2]].op!="attention_metadata" || p.nodes[n.inputs[2]].integer("window")!=n.integer("window"))error();
      } else check({{0, 3456}, {4, 0}}, {"scale_dim", "window", "selection"},{0,3072});
      eq("scale_dim", "192");
      eq("selection", "all_causal_within_window");
      auto window = n.integer("window");
      if ((window != 512 && window != 2048) || p.context > window)
        error();
    } else if (n.op == "swiglu") {
      if (n.inputs.size() == 1) {
        if (!tensor(0) || tensor(0) % 2) error();
        check({{0, tensor(0)}}, {}, {0, tensor(0) / 2});
      } else
        check({{0, tensor(0)}, {0, tensor(0)}}, {}, {0, tensor(0)});
    }
    else if (n.op == "router" || n.op == "router_projection") {
      check({{0, 2560}}, {"weight", "experts", "topk", "scoring", "scale"},
            n.op=="router_projection" ? T{0,256} : T{2,0});
      eq("experts", "256");
      eq("topk", "8");
      eq("scoring", "sigmoid_bias_group_topk");
      eq("scale", "2.5");
    } else if (n.op == "expert_linear") {
      if (n.attrs.count("paired_weight"))
        check({{0, 2560}, {2, 0}}, {"weight", "paired_weight", "width"},
              {3, 2048});
      else
        check({{0, 2560}, {2, 0}}, {"weight", "width"}, {3, 1024});
      eq("width", "1024");
    } else if (n.op == "expert_swiglu_down") {
      if (n.inputs.size() == 2)
        check({{3, 2048}, {2, 0}}, {"weight", "width"}, {3, 2560});
      else
        check({{3, 1024}, {3, 1024}, {2, 0}}, {"weight", "width"}, {3, 2560});
      eq("width", "2560");
    } else if (n.op == "expert_reduce") {
      if(p.physical) check({{0,20480},{3,2560},{2,0}},{},{0,2560});
      else check({{3,2560},{2,0}},{},{0,2560});
    }
    else if (n.op == "add" || (p.physical && n.op == "add_f32"))
      check({{0, tensor(0)}, {0, tensor(0)}}, {}, {0, tensor(0)});
    else
      error();
    if (p.physical && n.op != "cast") {
      std::string output_dtype = n.kind == 0 ? "bf16" : "opaque";
      if (n.op == "add_f32" || n.op == "expert_reduce" || n.op == "rope_f32" || n.op == "query_rotary_extract" || n.op == "router_projection") output_dtype = "f32";
      if (n.op == "allreduce") output_dtype = n.attrs.count("dtype") ? n.text("dtype") : "f32";
      if (n.op == "slice" || n.op == "repeat" || n.op == "concat" || (n.op == "optional_norm" && n.kernel == "alias"))
        output_dtype = p.nodes[n.inputs[0]].dtype;
      if (n.dtype != output_dtype) error();
      if (n.op == "concat" && p.nodes[n.inputs[0]].dtype != p.nodes[n.inputs[1]].dtype) error();
      auto expected_kernel = n.op == "add_f32" ? "add" : n.op;
      if (n.op != "linear" && n.op != "optional_norm" && n.kernel != expected_kernel) error();
      if (n.op == "linear" && n.kernel != "bf16" && n.kernel != "int8" && n.kernel != "paired_bf16") error();
      if (n.op == "optional_norm" && n.kernel != "norm" && n.kernel != "alias") error();
      for (auto j : n.inputs) {
        const auto &v = p.nodes[j];
        if (v.kind != 0) continue;
        std::string required = "bf16";
        if(n.op=="sparse_index" && j==n.inputs[1] && p.nodes[n.inputs[0]].kind==5)required="f16";
        if (n.op == "slice" || n.op == "repeat" || n.op == "concat" || (n.op == "optional_norm" && n.kernel == "alias")) continue;
        if (n.op == "router" || n.op == "rope" || n.op == "rope_f32" || n.op == "router_projection" || n.op == "router_select" || n.op == "expert_reduce" || n.op == "add_f32") required = "f32";
        if (n.op == "allreduce") required = n.attrs.count("dtype") ? n.text("dtype") : "f32";
        if (v.dtype != required) error();
      }
    }
    if (n.op == "expert_swiglu_down") {
      for (size_t j = 0; j + 1 < n.inputs.size(); ++j) {
        auto &g = p.nodes[n.inputs[j]];
        if (g.op != "expert_linear" || g.inputs[1] != n.inputs.back())
          error();
      }
    }
    if (n.op == "expert_reduce") {
      if(p.physical) {if(route_origin(p,n.inputs[1])!=n.inputs[2])error();}
      else {auto &d=p.nodes[n.inputs[0]];if(d.op!="expert_swiglu_down" || d.inputs.back()!=n.inputs[1])error();}
    }
    if (n.kind == 1 || n.kind == 4) {
      int inputs = 0, posts = 0, att = 0, dec = 0, all = 0;
      for (auto &c : p.nodes)
        if (std::find(c.inputs.begin(), c.inputs.end(), i) != c.inputs.end()) {
          ++all;
          inputs += c.op == "mhc_input";
          posts += c.op == "mhc_post";
          att += c.op == "latent_attention" || c.op == "packed_latent_attention" || c.op=="indexed_packed_attention";
          dec += c.op == "cache_dequantize";
        }
      if (n.kind == 4 && (att != 1 || (n.op=="cache_quantize_append" ? (all!=2 || dec!=1) : all!=1)))
        error();
      if(n.kind==1) {
        if(n.op=="mhc_pre_project") {
          if(all!=1)error();
          for(auto &c:p.nodes) if(std::find(c.inputs.begin(),c.inputs.end(),i)!=c.inputs.end() && c.op!="mhc_sinkhorn")error();
        } else if(inputs!=1 || posts!=(n.op=="mhc_pre" || n.op=="mhc_sinkhorn") || all!=inputs+posts) error();
      }
    }
  }
}
inline std::string bind_expert(std::string name, int e) {
  auto at = name.find("{expert}");
  if (at == std::string::npos)
    throw std::runtime_error("missing expert binding placeholder");
  name.replace(at, 8, std::to_string(e));
  return name;
}
inline void validate_bindings(const Plan &program,
                              const initialization::Plan &init) {
  using initialization::DType;
  std::map<std::string, const initialization::Parameter *> weights;
  for (auto &p : init.parameters)
    weights.emplace(p.name, &p);
  auto require = [&](const std::string &name, const std::vector<int64_t> &shape,
                     DType dtype) {
    auto it = weights.find(name);
    if (it == weights.end() || it->second->shape != shape ||
        it->second->dtype != dtype)
      throw std::runtime_error("parameter contract mismatch: " + name);
  };
  auto linear = [&](const std::string &name, int64_t input, int64_t output,
                    bool quant_only = false) {
    auto it = weights.find(name + ".weight");
    if (it == weights.end())
      throw std::runtime_error("missing projection " + name);
    auto dtype = it->second->dtype;
    if ((dtype != DType::Bf16 && dtype != DType::I8) ||
        (quant_only && dtype != DType::I8))
      throw std::runtime_error("projection dtype " + name);
    require(name + ".weight", {output, input}, dtype);
    if (dtype == DType::I8)
      require(name + ".weight_scale", {output, 1}, quant_only ? DType::Bf16 : DType::F32);
  };
  for (auto &n : program.nodes) {
    if (!n.attrs.count("weight"))
      continue;
    auto &name = n.text("weight");
    if (program.physical && n.op == "linear") {
      auto dtype = weights.at(name + ".weight")->dtype;
      std::string expected = n.attrs.count("paired_weight") ? "paired_bf16" : dtype == DType::Bf16 ? "bf16" : "int8";
      if (n.kernel != expected) throw std::runtime_error("planned projection kernel/checkpoint mismatch");
    }
    if (program.physical && n.op == "optional_norm" &&
        (n.kernel == "norm") != bool(weights.count(name + ".weight")))
      throw std::runtime_error("planned optional normalization/checkpoint mismatch");
    auto width = n.inputs.empty() ? 0 : program.nodes[n.inputs[0]].width;
    if (n.op == "embedding")
      require(name + ".weight", {151552, 2560}, DType::Bf16);
    else if (n.op == "linear" && n.attrs.count("paired_weight")) {
      require(name + ".weight", {n.integer("width"), width}, DType::Bf16);
      require(n.text("paired_weight") + ".weight", {n.integer("paired_width"), width}, DType::Bf16);
    }
    else if (n.op == "linear")
      linear(name, width, n.width);
    else if (n.op == "norm" || n.op == "optional_norm" || n.op == "cache_norm_rope_append" || n.op == "cache_normalize" || n.op == "cache_c8_norm_rope_append") {
      if (n.op != "optional_norm" || weights.count(name + ".weight"))
        require(name + ".weight", {(n.op=="cache_norm_rope_append" || n.op=="cache_normalize" || n.op=="cache_c8_norm_rope_append")?512:width}, DType::Bf16);
    } else if (n.op == "mome" || n.op == "mome_inplace")
      require(name + ".weight", {width, 1, 3}, DType::Bf16);
    else if (n.op == "mhc_pre" || n.op == "mhc_merge" || n.op == "mhc_pre_project") {
      bool merge = n.op == "mhc_merge";
      require(name + ".norm_gamma", {10240}, DType::F32);
      require(name + ".phi.weight", {merge ? 4 : 24, 10240}, DType::F32);
      require(name + (merge ? ".branch_alpha_pre" : ".branch_alpha"),
              {merge ? 1 : 3}, DType::F32);
      require(name + (merge ? ".branch_beta_pre" : ".branch_beta"),
              {merge ? 4 : 24}, DType::F32);
    } else if (n.op == "mla_query" || n.op == "query_project" || n.op == "mla_value")
      require(name + ".weight", {1536, 512}, DType::Bf16);
    else if (n.op == "sink_normalize") {
      require(name + ".param_sink_compressed_kv", {128, 512}, DType::Bf16);
      require(name + ".param_sink_k_pe", {128, 64}, DType::Bf16);
      require(n.text("norm") + ".weight", {512}, DType::Bf16);
    } else if (n.op == "router" || n.op == "router_projection") {
      require(name + ".gate.weight", {256, 2560}, DType::Bf16);
      require(name + ".e_score_correction_bias", {256}, DType::F32);
    } else if (n.op == "expert_linear" || n.op == "expert_swiglu_down" || n.op == "expert_gmm") {
      for (int e = int(init.rank) * 32; e < int(init.rank + 1) * 32; ++e) {
        linear(bind_expert(name, e),
               n.op == "expert_swiglu_down" && n.inputs.size() == 2 ? width / 2
                                                                    : width,
               n.integer("width"), true);
        if (n.attrs.count("paired_weight"))
          linear(bind_expert(n.text("paired_weight"), e), width,
                 n.integer("width"), true);
      }
    }
  }
}
} // namespace inferfabric::execution

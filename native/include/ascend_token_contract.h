#pragma once
#include "initialization_plan.h"
#include "token_execution_plan.h"
#include <algorithm>
namespace inferfabric::execution {
// Specialization capabilities of the registered Ascend kernels. This validates
// serialized IR independently of Rust before allocating model/device memory.
inline void validate_kernels(const Plan &p) {
  using T = std::pair<uint32_t, uint32_t>;
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
    if (n.op == "embedding") {
      check({}, {"weight", "width", "vocab"}, {0, 2560});
      eq("width", "2560");
      eq("vocab", "151552");
    } else if (n.op == "repeat") {
      auto a = tensor(0), c = n.integer("count");
      if (!c || c > 16)
        error();
      check({{0, a}}, {"count"}, {0, a * c});
    } else if (n.op == "mhc_pre" || n.op == "mhc_merge") {
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
      check({{0, tensor(0)}}, {"weight", "width", "placement"}, {0, w()});
      if (n.text("placement") != "tp_rows" &&
          n.text("placement") != "tp_columns" &&
          n.text("placement") != "replicated")
        error();
    } else if (n.op == "gather")
      check({{0, tensor(0)}}, {}, {0, tensor(0) * 8});
    else if (n.op == "allreduce")
      check({{0, tensor(0)}}, {}, {0, tensor(0)});
    else if (n.op == "mome") {
      std::vector<std::string> attrs{"weight", "taps"};
      if (n.attrs.count("placement")) {
        attrs.push_back("placement");
        if (n.text("placement") != "tp_channels" &&
            n.text("placement") != "replicated")
          error();
      }
      check({{0, tensor(0)}}, attrs, {0, tensor(0)});
      eq("taps", "3");
    } else if (n.op == "slice") {
      auto a = tensor(0);
      if (uint64_t(n.integer("start")) + w() > a)
        error();
      check({{0, a}}, {"start", "width"}, {0, w()});
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
    } else if (n.op == "cache_append")
      check({{0, 576}, {0, 73728}}, {}, {4, 0});
    else if (n.op == "latent_attention") {
      check({{0, 3456}, {4, 0}}, {"scale_dim", "window", "selection"},
            {0, 3072});
      eq("scale_dim", "192");
      eq("selection", "all_causal_within_window");
      auto window = n.integer("window");
      if ((window != 512 && window != 2048) || p.context > window)
        error();
    } else if (n.op == "swiglu")
      check({{0, tensor(0)}, {0, tensor(0)}}, {}, {0, tensor(0)});
    else if (n.op == "router") {
      check({{0, 2560}}, {"weight", "experts", "topk", "scoring", "scale"},
            {2, 0});
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
    } else if (n.op == "expert_reduce")
      check({{3, 2560}, {2, 0}}, {}, {0, 2560});
    else if (n.op == "add")
      check({{0, tensor(0)}, {0, tensor(0)}}, {}, {0, tensor(0)});
    else
      error();
    if (n.op == "expert_swiglu_down") {
      for (size_t j = 0; j + 1 < n.inputs.size(); ++j) {
        auto &g = p.nodes[n.inputs[j]];
        if (g.op != "expert_linear" || g.inputs[1] != n.inputs.back())
          error();
      }
    }
    if (n.op == "expert_reduce") {
      auto &d = p.nodes[n.inputs[0]];
      if (d.op != "expert_swiglu_down" || d.inputs.back() != n.inputs[1])
        error();
    }
    if (n.kind == 1 || n.kind == 4) {
      int inputs = 0, posts = 0, att = 0, all = 0;
      for (auto &c : p.nodes)
        if (std::find(c.inputs.begin(), c.inputs.end(), i) != c.inputs.end()) {
          ++all;
          inputs += c.op == "mhc_input";
          posts += c.op == "mhc_post";
          att += c.op == "latent_attention";
        }
      if (n.kind == 4 && (all != 1 || att != 1))
        error();
      if (n.kind == 1 && (inputs != 1 || posts != (n.op == "mhc_pre") ||
                          all != inputs + posts))
        error();
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
      require(name + ".weight_scale", {output, 1}, DType::F32);
  };
  for (auto &n : program.nodes) {
    if (!n.attrs.count("weight"))
      continue;
    auto &name = n.text("weight");
    auto width = n.inputs.empty() ? 0 : program.nodes[n.inputs[0]].width;
    if (n.op == "embedding")
      require(name + ".weight", {151552, 2560}, DType::Bf16);
    else if (n.op == "linear")
      linear(name, width, n.width);
    else if (n.op == "norm" || n.op == "optional_norm") {
      if (n.op == "norm" || weights.count(name + ".weight"))
        require(name + ".weight", {width}, DType::Bf16);
    } else if (n.op == "mome")
      require(name + ".weight", {width, 1, 3}, DType::Bf16);
    else if (n.op == "mhc_pre" || n.op == "mhc_merge") {
      bool merge = n.op == "mhc_merge";
      require(name + ".norm_gamma", {10240}, DType::F32);
      require(name + ".phi.weight", {merge ? 4 : 24, 10240}, DType::F32);
      require(name + (merge ? ".branch_alpha_pre" : ".branch_alpha"),
              {merge ? 1 : 3}, DType::F32);
      require(name + (merge ? ".branch_beta_pre" : ".branch_beta"),
              {merge ? 4 : 24}, DType::F32);
    } else if (n.op == "mla_query" || n.op == "mla_value")
      require(name + ".weight", {1536, 512}, DType::Bf16);
    else if (n.op == "sink_normalize") {
      require(name + ".param_sink_compressed_kv", {128, 512}, DType::Bf16);
      require(name + ".param_sink_k_pe", {128, 64}, DType::Bf16);
      require(n.text("norm") + ".weight", {512}, DType::Bf16);
    } else if (n.op == "router") {
      require(name + ".gate.weight", {256, 2560}, DType::Bf16);
      require(name + ".e_score_correction_bias", {256}, DType::F32);
    } else if (n.op == "expert_linear" || n.op == "expert_swiglu_down") {
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

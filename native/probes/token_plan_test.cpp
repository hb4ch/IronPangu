#include "ascend_token_contract.h"
#include <iostream>
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("usage: token_plan_test PLAN_DIR");
    auto p = inferfabric::execution::Plan::read(std::string(argv[1]) +
                                                "/execution.bin");
    inferfabric::execution::validate_kernels(p);
    for (int r = 0; r < 8; ++r) {
      auto init = inferfabric::initialization::Plan::read(
          std::string(argv[1]) + "/rank-" + std::to_string(r) + ".bin");
      inferfabric::execution::validate_bindings(p, init);
    }
    int rejected = 0;
    auto reject = [&](auto change) {
      auto bad = p;
      change(bad);
      try {
        inferfabric::execution::validate_kernels(bad);
      } catch (const std::exception &) {
        ++rejected;
        return;
      }
      throw std::runtime_error("invalid kernel contract accepted");
    };
    reject([](auto &x) { x.nodes[0].op = "unknown"; });
    reject([](auto &x) { x.nodes[0].width = 1; });
    reject([](auto &x) { x.nodes[0].attrs["ignored"] = "true"; });
    if (p.physical) {
      if(std::any_of(p.nodes.begin(),p.nodes.end(),[](const auto &n){return n.op=="attention_metadata";})) {
        reject([](auto &x){for(auto &n:x.nodes)if(n.op=="attention_metadata") {n.attrs["window"]="2047";break;}});
        reject([](auto &x){for(auto &n:x.nodes)if(n.op=="attention_metadata_zero") {n.attrs.erase("window");break;}});
        reject([](auto &x){for(auto &n:x.nodes)if(n.op=="latent_attention") {n.attrs["window"]=n.attrs["window"]=="512"?"2048":"512";break;}});
      }
      reject([](auto &x) { x.nodes[0].dtype = "f32"; });
      reject([](auto &x) { x.nodes[0].kernel = "unbound"; });
      reject([](auto &x) { for (auto &n : x.nodes) if (n.op == "cast") { n.kernel="copy"; break; } });
      reject([](auto &x) { for (auto &n : x.nodes) if (n.op == "allreduce") {
        // A BF16 embedding can be a valid BF16 collective operand. Corrupt
        // the value kind instead, independent of the chosen precision.
        for(size_t j=0;j<x.nodes.size();++j)if(x.nodes[j].kind!=0){n.inputs[0]=uint32_t(j);break;}
        break;
      } });
      if (std::any_of(p.nodes.begin(),p.nodes.end(),[&](const auto &n){return n.op=="mome_inplace" && p.nodes[n.inputs[0]].op=="slice";})) {
        reject([](auto &x) { for (auto &n : x.nodes) if (n.op=="mome_inplace" && x.nodes[n.inputs[0]].op=="slice") {
          auto &view=x.nodes[n.inputs[0]]; view.attrs["start"]=std::to_string(view.integer("start")+1); break;
        } });
      }
    }
    reject([](auto &x) {
      for (auto &n : x.nodes)
        if (n.op == "rope" || n.op == "cache_norm_rope_append" || n.op == "cache_normalize") {
          n.attrs["theta"] = "1";
          break;
        }
    });
    reject([](auto &x) {
      for (auto &n : x.nodes)
        if (n.op == "mhc_post") {
          n.inputs[0] = 0;
          break;
        }
    });
    reject([](auto &x) {
      for (auto &n : x.nodes)
        if (n.op == "expert_swiglu_down" || n.op == "route_quantize") {
          n.inputs.back() = 0;
          break;
        }
    });
    reject([](auto &x) {
      for (auto &n : x.nodes)
        if(n.op=="head_matmul") {n.attrs["input_rank_stride"]="4294967295";break;}
        else if ((n.op == "mla_query" || n.op == "query_project")) {
          n.attrs["input_layout"] =
              n.attrs.count("input_layout") &&
                      n.attrs["input_layout"] == "rank_local"
                  ? "replicated"
                  : "rank_local";
          break;
        }
    });
    reject([](auto &x) {
      for (auto &n : x.nodes)
        if(n.op=="head_matmul") {n.attrs["weight_offset"]="4294967295";break;}
        else if ((n.op == "mla_query" || n.op == "query_project")) {
          n.attrs["input_layout"] = "unknown";
          break;
        }
    });
    reject([](auto &x) {
      for (auto &n : x.nodes)
        if (n.op == "allreduce") {
          n.attrs["dtype"] = "f16";
          break;
        }
    });
    if (std::any_of(p.nodes.begin(), p.nodes.end(), [](const auto &n) {
          return n.op == "linear" && n.attrs.count("paired_weight");
        })) {
      reject([](auto &x) {
        for (auto &n : x.nodes) if (n.op == "linear" && n.attrs.count("paired_weight")) {
          n.attrs["paired_width"] = "0";
          break;
        }
      });
      reject([](auto &x) {
        for (auto &n : x.nodes) if (n.op == "linear" && n.attrs.count("paired_weight")) {
          n.attrs["placement"] = "tp_rows";
          break;
        }
      });
    }
    if (std::any_of(p.nodes.begin(), p.nodes.end(), [](const auto &n) {
          return n.op == "swiglu" && n.inputs.size() == 1;
        })) {
      reject([](auto &x) {
        for (auto &n : x.nodes) if (n.op == "swiglu" && n.inputs.size() == 1) {
          n.inputs.push_back(n.inputs[0]);
          break;
        }
      });
    }
    if(std::any_of(p.nodes.begin(),p.nodes.end(),[](const auto &n){return n.op=="expert_gmm";})) {
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="expert_gmm"){x.output=n.inputs[1];break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="expert_reduce"){n.inputs[0]=x.nodes[n.inputs[0]].inputs[0];break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="latent_attention"){n.inputs.pop_back();break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="expert_act_quantize"){n.inputs[0]=0;break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="mhc_sinkhorn"){n.inputs[0]=0;break;}});
    }
    if(std::any_of(p.nodes.begin(),p.nodes.end(),[](const auto &n){return n.op=="cache_c8_norm_rope_append";})) {
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="cache_c8_norm_rope_append"){n.attrs["cache_quantization"]="per_token_head_symmetric";break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="cache_c8_norm_rope_append"){n.attrs["cache_rope_dtype"]="int8";break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="cache_c8_norm_rope_append"){n.attrs["cache_scale_dtype"]="bf16";break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="packed_latent_attention" || n.op=="indexed_packed_attention"){n.inputs.back()=0;break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="packed_latent_attention" || n.op=="indexed_packed_attention"){n.inputs[1]=0;break;}});
    }
    if(std::any_of(p.nodes.begin(),p.nodes.end(),[](const auto &n){return n.op=="index_quantize";})) {
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="index_quantize"){n.attrs["block_width"]="3072";break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="sparse_index"){auto &q=x.nodes[n.inputs[0]];n.inputs[0]=q.inputs[0];break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="sparse_index"){auto &w=x.nodes[n.inputs[1]];n.inputs[1]=w.inputs[0];break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="index_cache_append"){n.attrs["cache_dtype"]="int4";break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="index_cache_append"){auto &q=x.nodes[n.inputs[0]];n.inputs[0]=q.inputs[0];break;}});
    }
    if(std::any_of(p.nodes.begin(),p.nodes.end(),[](const auto &n){return n.op=="cache_quantize_append";})) {
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="cache_quantize_append"){n.attrs["cache_dtype"]="int2";break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="cache_quantize_append"){n.attrs["cache_scale_dtype"]="bf16";break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="cache_dequantize"){n.attrs["cache_dtype"]="bf16";break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="cache_dequantize"){n.width--;break;}});
      reject([](auto &x){for(auto &n:x.nodes)if(n.op=="latent_attention"){n.inputs[3]=0;break;}});
    }
    auto init = inferfabric::initialization::Plan::read(std::string(argv[1]) +
                                                        "/rank-0.bin");
    init.parameters[0].shape[0] = 1;
    try {
      inferfabric::execution::validate_bindings(p, init);
      throw std::logic_error("bad binding accepted");
    } catch (const std::logic_error &) {
      throw;
    } catch (const std::runtime_error &) {
      ++rejected;
    }
    std::cout << "token graph nodes=" << p.nodes.size()
              << "; 8 rank bindings passed; " << rejected
              << " negative contracts rejected\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

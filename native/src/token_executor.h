#pragma once
#include "cache_quantization.h"
// Device-resident execution of a validated generic token program.
// No activation or routing value has a host representation.
#include "ascend_token_contract.h"
#include "compiled_kernel.h"
#include "execution_capture.h"
#include "initialization_acl.h"
#include "openpangu_decoder_state.h"
#include "openpangu_mhc.h"
#include "prepared_weight_views.h"
#include "resident_tensor.h"
#include "resident_rotary.h"
#include "resident_transfer.h"
#include <aclnn_ai_infra_fused_causal_conv1d.h>
#include <aclnn_ai_infra_kv_rms_norm_rope_cache.h>
#include "aclnn_ai_infra_kv_quant_sparse_flash_attention.h"
#include <aclnn_ai_infra_sparse_flash_attention_pioneer.h>
#include <aclnn_ai_infra_fused_infer_attention_sink.h>
#include <aclnn_ai_infra_fused_infer_attention_sink_metadata.h>
#include <aclnnop/aclnn_batch_matmul.h>
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_dequant_swiglu_quant.h>
#include <aclnnop/aclnn_dynamic_quant.h>
#include <aclnnop/aclnn_grouped_matmul_v5.h>
#include <aclnnop/aclnn_index_copy.h>
#include <aclnnop/aclnn_moe_finalize_routing_v2.h>
#include <aclnnop/aclnn_moe_init_routing_v3.h>
#include <aclnnop/aclnn_quant_matmul_v5.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_rotary_position_embedding.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <hccl/hccl.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <variant>
namespace inferfabric::token {
using pangu::MomeState;
using pangu::AttentionState;
using pangu::MhcState;
using pangu::training_api;
namespace prepared = inferfabric::prepared;
using Host = std::vector<float>;
using V = inferfabric::resident::Tensor;
namespace resident = inferfabric::resident;
using openpangu::checked;
using openpangu::DeviceTensor;
float frombf(uint16_t b) {
  uint32_t n = uint32_t(b) << 16;
  float x;
  std::memcpy(&x, &n, 4);
  return x;
}
uint16_t tobf(float x) {
  uint32_t n;
  std::memcpy(&n, &x, 4);
  return uint16_t((n + 0x7fff + ((n >> 16) & 1)) >> 16);
}
float roundbf(float x) { return frombf(tobf(x)); }
Host unpacked(const std::vector<uint16_t> &b) {
  Host x;
  for (auto v : b)
    x.push_back(frombf(v));
  return x;
}
struct View {
  aclTensor *t = nullptr;
  bool cached = false;
  View(void *p, std::vector<int64_t> shape, aclDataType type,
       bool transpose = false) {
    if (prepared::active) {
      t = prepared::active->descriptor(p, shape, type, transpose);
      cached = true;
      return;
    }
    auto storage = shape;
    std::vector<int64_t> s(shape.size(), 1);
    for (int i = int(s.size()) - 2; i >= 0; --i)
      s[i] = s[i + 1] * shape[i + 1];
    if (transpose) {
      std::swap(shape[0], shape[1]);
      std::swap(s[0], s[1]);
    }
    t = aclCreateTensor(shape.data(), shape.size(), type, s.data(), 0,
                        ACL_FORMAT_ND, storage.data(), storage.size(), p);
    if (!t)
      throw std::runtime_error("weight view");
  }
  ~View() {
    if (t && !cached)
      aclDestroyTensor(t);
  }
};
struct Decoder {
  // Declared first: outlives all values, descriptors and state.
  prepared::Context execution;
  resident::PreparedWeightViews weight_views;
  std::vector<std::unique_ptr<inferfabric::PreparedKernel>> pending_kernels;
  aclmdlRI decode_graph = nullptr;
  aclmdlRI prefill_graph = nullptr;
  V prefill_output;
  std::vector<std::unique_ptr<inferfabric::PreparedKernel>> prefill_kernels;
  V graph_output;
  std::vector<std::unique_ptr<inferfabric::PreparedKernel>> graph_kernels;
  std::unique_ptr<DeviceTensor> token_index, position_index, cache_position, zero_cache_position,
      rotary_cos, rotary_sin, rotary_cos_bf16, rotary_sin_bf16, sequence_length64, query_length64, attention_metadata, attention_mask;
  int64_t cube_cores = 0, vector_cores = 0;
  bool metadata_enqueued = false;
  std::unique_ptr<DeviceTensor> packed_indices, packed_length;
  int rank;
  int current_position = -1;
  std::map<std::string, std::unique_ptr<openpangu::Router>> routers;
  const inferfabric::execution::Plan &program;
  std::map<std::string, std::unique_ptr<MhcState>> mhc_states;
  std::map<std::pair<std::string, std::string>, std::unique_ptr<DeviceTensor>> paired_projections;
  std::map<std::string, std::unique_ptr<MomeState>> mome_states;
  std::map<std::string, std::unique_ptr<AttentionState>> attention_states;
  std::map<std::string, size_t> cache_control_offsets;
  std::unique_ptr<DeviceTensor> cache_controls;
  std::unique_ptr<DeviceTensor> resident_index;
  uint32_t active_resident = 0;
  uint32_t active_tokens = 1, active_row = 0;
  aclTensor *prefix(DeviceTensor &x, const std::vector<int64_t> &shape, size_t offset=0) {
    size_t bytes=resident::element_bytes(x.type);
    for(auto dimension:shape) { if(dimension<=0 || size_t(dimension)>x.bytes/bytes)throw std::runtime_error("prepared prefix bounds"); bytes*=dimension; }
    if(offset>x.bytes || bytes>x.bytes-offset)throw std::runtime_error("prepared prefix extent");
    return execution.descriptor(static_cast<uint8_t *>(x.data)+offset,shape,x.type);
  }
  aclTensor *control_row(DeviceTensor &x, const std::vector<int64_t> &shape={1}) {
    size_t bytes=resident::element_bytes(x.type);
    for(auto dimension:shape)bytes*=dimension;
    return prefix(x,shape,active_row*bytes);
  }
  std::vector<uint8_t> cache_control_host;
  std::map<std::string, V> prepared_sinks;
  aclrtStream stream = nullptr;
  HcclComm comm = nullptr;
  inferfabric::CompiledKernel swiglu;
  inferfabric::initialization::ParameterStore weights;
  std::map<std::string, inferfabric::initialization::ParameterId> bindings;
  std::map<std::string, Host> host;
  std::map<inferfabric::initialization::ParameterId,
           inferfabric::initialization::DType>
      source_types;
  Decoder(const inferfabric::initialization::Plan &p, const std::string &root,
          HcclRootInfo &info, const inferfabric::execution::Plan &execution, const std::string &arena_path, uint64_t fingerprint, const std::string &workspace_path, uint64_t backend_fingerprint = 0)
      : execution(arena_path,fingerprint,workspace_path,backend_fingerprint), rank(p.rank), program(execution),
        swiglu(std::getenv("INFERFABRIC_SWIGLU_LIBRARY")
                   ? std::getenv("INFERFABRIC_SWIGLU_LIBRARY")
                   : IF_DEFAULT_SWIGLU_LIBRARY,
               "inferfabric.swiglu.bf16.v1", "ascend"),
        weights(inferfabric::initialization::Executor::execute(
            p, root, inferfabric::initialization::acl_device_memory())) {
    for (const auto &parameter : p.parameters)
      bindings.emplace(parameter.name, parameter.id);
    for (const auto &load : p.instructions)
      source_types.emplace(load.destination, load.source_dtype);
    stream = this->execution.stream;
    resident_index=std::make_unique<DeviceTensor>(std::vector<int64_t>{1},ACL_INT32,4);
    resident::control_upload(*resident_index,std::vector<int32_t>{0});
    for (const auto &node : program.nodes)
      if (node.op == "cache_append" || node.op == "cache_norm_rope_append" || node.op == "cache_c8_norm_rope_append") {
        auto offset=cache_control_offsets.size()*16*program.prefill_tokens;
        if (!cache_control_offsets.emplace(node.name,offset).second)
          throw std::runtime_error("duplicate cache control binding");
      }
    if (!cache_control_offsets.empty()) {
      cache_control_host.resize(cache_control_offsets.size()*16*program.prefill_tokens,0);
      cache_controls=std::make_unique<DeviceTensor>(std::vector<int64_t>{int64_t(cache_control_host.size())},ACL_UINT8,1);
    }
    token_index =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{program.prefill_tokens}, ACL_INT32, 4);
    position_index =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{program.prefill_tokens}, ACL_INT32, 4);
    cache_position =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{program.prefill_tokens}, ACL_INT64, 8);
    zero_cache_position=std::make_unique<DeviceTensor>(std::vector<int64_t>{1},ACL_INT64,8);
    resident::control_upload(*zero_cache_position,std::vector<int64_t>{0});
    sequence_length64 =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{program.prefill_tokens}, ACL_INT64, 8);
    rotary_cos =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{int64_t(program.prefill_tokens)*64}, ACL_FLOAT, 4);
    rotary_sin =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{int64_t(program.prefill_tokens)*64}, ACL_FLOAT, 4);
    rotary_cos_bf16 =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{program.prefill_tokens, 1, 1, 64}, ACL_BF16, 2);
    rotary_sin_bf16 =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{program.prefill_tokens, 1, 1, 64}, ACL_BF16, 2);
    query_length64 = std::make_unique<DeviceTensor>(std::vector<int64_t>{1}, ACL_INT64, 8);
    attention_metadata = std::make_unique<DeviceTensor>(std::vector<int64_t>{int64_t(program.prefill_tokens)*1024}, ACL_INT32, 4);
    resident::control_upload(*query_length64, std::vector<int64_t>{1});
    attention_mask = std::make_unique<DeviceTensor>(std::vector<int64_t>{2048, 2048}, ACL_BOOL, 1);
    std::vector<uint8_t> mask(2048 * 2048);
    for (int i = 0; i < 2048; ++i) for (int j = i + 1; j < 2048; ++j) mask[i * 2048 + j] = 1;
    resident::control_upload(*attention_mask, mask);
    int32_t device = -1;
    checked(aclrtGetDevice(&device), "attention device");
    checked(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM, &cube_cores), "attention cube cores");
    checked(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vector_cores), "attention vector cores");
    if (cube_cores <= 0 || vector_cores <= 0) throw std::runtime_error("attention core contract");
    checked(HcclCommInitRootInfo(8, &info, rank, &comm), "HCCL");
  }
  ~Decoder() {
    if (stream)
      aclrtSynchronizeStream(stream);
    execution.release_graphs();
    if (comm)
      HcclCommDestroy(comm);
  }
  void run(openpangu::Launcher f, uint64_t bytes, aclOpExecutor *e) {
    execution.require_stream(stream);
    void *workspace = execution.workspace(bytes);
    {
      inferfabric::profiling::Scope timing("kernel_launch");
      checked(f(workspace, bytes, e, stream), "operator");
    }
  }
  inferfabric::initialization::ResidentParameter &w(const std::string &n) {
    auto it = bindings.find(n);
    if (it == bindings.end())
      throw std::runtime_error("unbound model parameter " + n);
    return weights.at(it->second);
  }
  const Host &cpu(const std::string &n) {
    resident::TransferScope boundary(
        resident::TransferRole::ParameterPreparation);
    auto it = host.find(n);
    if (it != host.end())
      return it->second;
    auto &a = w(n);
    Host v;
    if (a.bytes > 8 * 1024 * 1024)
      throw std::runtime_error("scalar weight staging bound " + n);
    if (a.dtype == inferfabric::initialization::DType::F32) {
      v.resize(a.bytes / 4);
      checked(aclrtMemcpy(v.data(), a.bytes, a.data, a.bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "float weight");
    } else if (a.dtype == inferfabric::initialization::DType::Bf16) {
      std::vector<uint16_t> b(a.bytes / 2);
      checked(aclrtMemcpy(b.data(), a.bytes, a.data, a.bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "BF16 weight");
      v = unpacked(b);
    } else
      throw std::runtime_error("scalar dtype");
    return host.emplace(n, std::move(v)).first->second;
  }
  V cast(const V &x, aclDataType dtype, bool declared = false) {
    if (x.dtype() == dtype)
      return x;
    if(program.physical && !declared) throw std::runtime_error("implicit physical storage conversion forbidden");
    V out(x.size(), dtype);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnCast", true, aclnnCastGetWorkspaceSize,
                            x.tensor(), dtype, out.tensor(), &bytes, &e),
            "resident cast prepare");
    run(aclnnCast, bytes, e);
    return out;
  }
  V rounded(const V &x) { return cast(x, ACL_BF16); }
  V result(const DeviceTensor &x, aclDataType dtype = ACL_BF16) {
    return V::alias(x, dtype);
  }
  void to_device(DeviceTensor &dst, const V &src,
                 aclDataType dtype = ACL_BF16) {
    auto value = cast(src, dtype);
    if (value.bytes() != dst.bytes)
      throw std::runtime_error("resident copy shape");
    resident::copy(dst.data, value.data(), dst.bytes);
  }
  V binary(const V &lhs, const V &rhs, char kind) {
    auto a = cast(lhs, ACL_FLOAT), b = cast(rhs, ACL_FLOAT);
    if (a.dtype() != ACL_FLOAT || b.dtype() != ACL_FLOAT ||
        (a.size() != b.size() && b.size() != 1))
      throw std::runtime_error("resident binary contract");
    V out(a.size());
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    aclScalar *one = execution.one();
    aclnnStatus status =
        kind == '*'
            ? prepared::query("aclnnMul", true, aclnnMulGetWorkspaceSize,
                              a.tensor(), b.tensor(), out.tensor(), &bytes, &e)
        : kind == '+'
            ? prepared::query("aclnnAdd", true, aclnnAddGetWorkspaceSize,
                              a.tensor(), b.tensor(), one, out.tensor(), &bytes,
                              &e)
            : prepared::query("aclnnSub", true, aclnnSubGetWorkspaceSize,
                              a.tensor(), b.tensor(), one, out.tensor(), &bytes,
                              &e);
    checked(status, "resident binary prepare");
    run(kind == '*' ? aclnnMul : kind == '+' ? aclnnAdd : aclnnSub, bytes, e);
    return out;
  }
  V gather(const V &x) {
    auto a = cast(x, ACL_BF16);
    DeviceTensor b({int64_t(x.size() * 8)}, ACL_BF16, 2);
    {
      inferfabric::profiling::Scope timing("HcclAllGather");
      checked(HcclAllGather(a.data(), b.data, x.size(), HCCL_DATA_TYPE_BFP16,
                            comm, stream),
              "all gather");
    }
    if(active_tokens==1)return result(b);
    // HCCL produces rank-major rows; the next operator consumes token-major rows.
    V output(x.size()*8,ACL_BF16);
    auto width=x.size()/active_tokens;
    for(uint32_t token=0;token<active_tokens;++token)for(size_t peer=0;peer<8;++peer)
      resident::copy(static_cast<uint16_t *>(output.data())+(token*8+peer)*width,
                     static_cast<uint16_t *>(b.data)+(peer*active_tokens+token)*width,width*2);
    return output;
  }
  V sum(const V &x, bool bf16 = false) {
    auto dtype = bf16 ? ACL_BF16 : ACL_FLOAT;
    auto a = cast(x, dtype);
    DeviceTensor b({int64_t(x.size())}, dtype, bf16 ? 2 : 4);
    {
      inferfabric::profiling::Scope timing("HcclAllReduce");
      checked(HcclAllReduce(a.data(), b.data, x.size(),
                            bf16 ? HCCL_DATA_TYPE_BFP16 : HCCL_DATA_TYPE_FP32,
                            HCCL_REDUCE_SUM, comm, stream),
              "all reduce");
    }
    return result(b, dtype);
  }
  V matrix(const V &x, void *weight, int rows, int cols,
           bool transposed = true) {
    int output = transposed ? rows : cols;
    int input_width = transposed ? cols : rows;
    if(input_width<=0 || x.size()%input_width)throw std::runtime_error("matrix token rows");
    int64_t tokens=x.size()/input_width;
    auto value = cast(x, ACL_BF16);
    View input(value.data(), {tokens, input_width}, ACL_BF16);
    DeviceTensor out({tokens, output}, ACL_BF16, 2);
    View view(weight, {rows, cols}, ACL_BF16, transposed);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnMatmul", true, aclnnMatmulGetWorkspaceSize,
                            input.t, view.t, out.tensor, 0, &bytes, &e),
            "BF16 matmul prepare");
    run(aclnnMatmul, bytes, e);
    return result(out);
  }
  struct Quantized { V values, scales; };
  Quantized quantize(const V &x, size_t width=0) {
    if (x.dtype()!=ACL_BF16) throw std::runtime_error("quantize requires planned BF16 input");
    if(!width)width=x.size();
    if(x.size()%width)throw std::runtime_error("quantize token rows");
    int64_t tokens=x.size()/width;
    View input(x.data(), {tokens,int64_t(width)}, ACL_BF16);
    DeviceTensor quant({tokens,int64_t(width)}, ACL_INT8, 1), scale({tokens},ACL_FLOAT,4);
    uint64_t bytes=0; aclOpExecutor *e=nullptr;
    checked(prepared::query("aclnnDynamicQuant", false, aclnnDynamicQuantGetWorkspaceSize,
                            input.t,nullptr,quant.tensor,scale.tensor,&bytes,&e),"quant prepare");
    run(aclnnDynamicQuant,bytes,e);
    return {result(quant,ACL_INT8),result(scale,ACL_FLOAT)};
  }
  V quantized_linear(const std::string &name, const Quantized &x) {
    auto &a=w(name+".weight"), &s=w(name+".weight_scale");
    int n=a.descriptor.shape[0],k=a.descriptor.shape[1];
    int64_t tokens=x.scales.size();
    if (x.values.dtype()!=ACL_INT8 || x.values.size()!=size_t(k)*tokens || x.scales.dtype()!=ACL_FLOAT || !tokens)
      throw std::runtime_error("quantized linear input contract");
    View quant(x.values.data(),{tokens,k},ACL_INT8), scale(x.scales.data(),{tokens},ACL_FLOAT);
    View weight(a.data,{n,k},ACL_INT8), scales(s.data,{n},ACL_FLOAT);
    DeviceTensor out({tokens,n},ACL_BF16,2); uint64_t bytes=0; aclOpExecutor *e=nullptr;
    checked(prepared::query("aclnnQuantMatmulV5",false,aclnnQuantMatmulV5GetWorkspaceSize,
                            quant.t,weight.t,scale.t,scales.t,nullptr,nullptr,nullptr,nullptr,
                            nullptr,false,true,0,out.tensor,&bytes,&e),"int8 matmul prepare");
    run(aclnnQuantMatmulV5,bytes,e); return result(out);
  }
  V linear(const std::string &name, const V &x, const std::string &implementation = "") {
    auto &a=w(name+".weight");
    int n=a.descriptor.shape[0],k=a.descriptor.shape[1];
    if (!k || x.size()%size_t(k)) throw std::runtime_error("linear shape "+name);
    if (implementation=="bf16" || (implementation.empty() && a.dtype==inferfabric::initialization::DType::Bf16))
      return matrix(x,a.data,n,k);
    if (program.physical) throw std::runtime_error("physical INT8 projection requires explicit quantize");
    return quantized_linear(name,quantize(cast(x,ACL_BF16),k));
  }
  V linear_pair(const std::string &first, const std::string &second, const V &x) {
    auto &a = w(first + ".weight"), &b = w(second + ".weight");
    if (a.dtype != inferfabric::initialization::DType::Bf16 || b.dtype != a.dtype ||
        a.descriptor.shape[1] != b.descriptor.shape[1] || x.size()%a.descriptor.shape[1])
      throw std::runtime_error("paired projection dtype/shape");
    int64_t rows = a.descriptor.shape[0] + b.descriptor.shape[0];
    auto &packed = paired_projections[{first, second}];
    if (!packed) {
      prepared::Context::InitializationCopies declared(execution);
      packed = std::make_unique<DeviceTensor>(std::vector<int64_t>{rows, a.descriptor.shape[1]}, ACL_BF16, 2);
      execution.copy(packed->data, a.data, a.bytes);
      execution.copy(static_cast<char *>(packed->data) + a.bytes, b.data, b.bytes);
    }
    return matrix(x, packed->data, rows, a.descriptor.shape[1]);
  }
  void trace(const std::string &, const V &) {}
  V norm(const V &x, const std::string &name) {
    auto &gamma = w(name + ".weight");
    size_t width=gamma.bytes/2;
    if (!width || x.size()%width)
      throw std::runtime_error("norm shape");
    auto value = cast(x, ACL_BF16);
    int64_t tokens=x.size()/width;
    View input(value.data(), {tokens, int64_t(width)}, ACL_BF16);
    DeviceTensor out({tokens, int64_t(width)}, ACL_BF16, 2),
        rstd({tokens, 1}, ACL_FLOAT, 4);
    View weight(gamma.data, {int64_t(width)}, ACL_BF16);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnRmsNorm", false, aclnnRmsNormGetWorkspaceSize,
                            input.t, weight.t, 1e-5, out.tensor, rstd.tensor,
                            &bytes, &e),
            "RMSNorm prepare");
    run(aclnnRmsNorm, bytes, e);
    auto y = result(out);
    trace(name, y);
    return y;
  }
  V mome(V x, const std::string &name, int position,
         const std::string &state_name, bool inplace = false) {
    auto &ptr = mome_states[state_name];
    if (!ptr) {
      resident::TransferScope constants(resident::TransferRole::ControlInput);
      ptr = std::make_unique<MomeState>(x.size(), program.resident_sequences);
      auto &source = cpu(name + ".weight");
      std::vector<uint16_t> weight(source.size());
      for (size_t j = 0; j < x.size(); ++j)
        for (int tap = 0; tap < 3; ++tap)
          weight[tap * x.size() + j] = tobf(source[j * 3 + tap]);
      resident::parameter_upload(ptr->weight, weight);
    }
    auto &s = *ptr;
    if (!inplace) to_device(s.input, x);
    View input_view(inplace ? x.data() : s.input.data, {1, int64_t(x.size())}, ACL_BF16);
    if (position != current_position)
      throw std::runtime_error("position input mismatch");
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnAiInfraFusedCausalConv1d", false,
                            aclnnAiInfraFusedCausalConv1dGetWorkspaceSize,
                            input_view.t, s.weight.tensor, s.state.tensor,
                            s.starts.tensor, resident_index->tensor, nullptr, nullptr,
                            nullptr, control_row(*position_index), nullptr, nullptr,
                            nullptr, 0, -1, 0, 1, 1, 128, 1, true, s.out.tensor,
                            &bytes, &e),
            "MoME prepare");
    run(aclnnAiInfraFusedCausalConv1d, bytes, e);
    return inplace ? x : result(s.input);
  }
  struct HC {
    MhcState *device = nullptr;
    V input, residual;
  };
  HC pre(const V &x, const std::string &name, bool merge,
         const std::string &state_name, bool defer_sink = false) {
    auto &ptr = mhc_states[state_name];
    if (!ptr) {
      ptr = std::make_unique<MhcState>(program.prefill_tokens);
      auto &phi = cpu(name + ".phi.weight");
      auto &gamma = cpu(name + ".norm_gamma");
      Host folded(24 * 10240, 0), alpha(3, 0), beta(24, 0);
      for (size_t j = 0; j < phi.size(); ++j)
        folded[j] = phi[j] * gamma[j % 10240];
      auto &a = cpu(name + (merge ? ".branch_alpha_pre" : ".branch_alpha"));
      auto &b = cpu(name + (merge ? ".branch_beta_pre" : ".branch_beta"));
      std::copy(a.begin(), a.end(), alpha.begin());
      std::copy(b.begin(), b.end(), beta.begin());
      resident::parameter_upload(ptr->phi, folded);
      resident::parameter_upload(ptr->alpha, alpha);
      resident::parameter_upload(ptr->beta, beta);
    }
    auto &s = *ptr;
    auto residual = cast(x, ACL_BF16);
    int64_t tokens=active_tokens;
    View residual_view(residual.data(), {tokens, 4, 2560}, ACL_BF16);
    auto &api = training_api();
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("mHC.pre", false, api.pre_size, residual_view.t,
                            s.phi.tensor, s.alpha.tensor, s.beta.tensor,
                            nullptr, 0, 1e-6, 1e-6, prefix(s.input,{tokens,2560}),
                            prefix(s.post,{tokens,4}), prefix(s.res,{tokens,4,4}), prefix(s.rms,{tokens}),
                            prefix(s.mix,{tokens,24}), prefix(s.pre_weights,{tokens,4}), &bytes, &e),
            "mHC pre prepare");
    run(api.pre, bytes, e);
    if (!merge && !defer_sink) {
      checked(prepared::query("mHC.sink", false, api.sink_size, prefix(s.res,{tokens,4,4}), 0,
                              1e-6, 20, prefix(s.sink,{tokens,4,4}), nullptr, nullptr, &bytes,
                              &e),
              "mHC Sinkhorn prepare");
      run(api.sink, bytes, e);
    }
    HC h;
    h.device = &s;
    h.residual = residual;
    h.input = result(s.input).view(0,tokens*2560);
    return h;
  }
  HC sinkhorn(HC h) {
    if(!h.device) throw std::runtime_error("missing mHC projection");
    auto &s=*h.device;auto &api=training_api();uint64_t bytes=0;aclOpExecutor *e=nullptr;
    checked(prepared::query("mHC.sink",false,api.sink_size,prefix(s.res,{active_tokens,4,4}),0,1e-6,20,prefix(s.sink,{active_tokens,4,4}),nullptr,nullptr,&bytes,&e),"mHC Sinkhorn prepare");
    run(api.sink,bytes,e);return h;
  }
  V post(const V &, const V &branch, const HC &h) {
    if (!h.device)
      throw std::runtime_error("missing mHC pre state");
    auto &s = *h.device;
    auto value = cast(branch, ACL_BF16);
    View residual(h.residual.data(), {active_tokens, 4, 2560}, ACL_BF16),
        branch_view(value.data(), {active_tokens, 2560}, ACL_BF16);
    auto &api = training_api();
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("mHC.post", false, api.post_size, residual.t,
                            prefix(s.sink,{active_tokens,4,4}), branch_view.t, prefix(s.post,{active_tokens,4}),
                            prefix(s.out,{active_tokens,4,2560}), &bytes, &e),
            "mHC post prepare");
    run(api.post, bytes, e);
    return result(s.out).view(0,size_t(active_tokens)*10240);
  }
  V column_slice(const V &input, int64_t rows, int64_t start, int64_t width) {
    return resident::column_slice(input,rows,start,width);
  }
  V rope_bf16(const V &input) {
    return resident::rope_bf16(input,control_row(*rotary_cos_bf16,{1,1,1,64}),
        control_row(*rotary_sin_bf16,{1,1,1,64}));
  }
  V rope(const V &input, int pos, bool fp32_output = false) {
    if (!input.size() || input.size() % 64 || pos != current_position)
      throw std::runtime_error("rotary shape/position contract");
    // Preserve FP32 products/sum and the existing final BF16 rounding while
    // fusing the half rotation, products and sum into one vendor operation.
    auto value = cast(input, ACL_FLOAT);
    int64_t heads = input.size() / 64;
    DeviceTensor out({1, heads, 1, 64}, ACL_FLOAT, 4);
    View x(value.data(), {1, heads, 1, 64}, ACL_FLOAT),
        cos(static_cast<float *>(rotary_cos->data)+active_row*64, {1, 1, 1, 64}, ACL_FLOAT),
        sin(static_cast<float *>(rotary_sin->data)+active_row*64, {1, 1, 1, 64}, ACL_FLOAT);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnRotaryPositionEmbedding", false,
                            aclnnRotaryPositionEmbeddingGetWorkspaceSize, x.t,
                            cos.t, sin.t, 0, out.tensor, &bytes, &e),
            "rotary prepare");
    run(aclnnRotaryPositionEmbedding, bytes, e);
    return fp32_output ? result(out, ACL_FLOAT) : rounded(result(out, ACL_FLOAT));
  }
  V query_project(const V &q, const std::string &weight_name, bool local_heads) {
    auto &weight = w(weight_name + ".weight");
    auto local = resident::slice(q, local_heads ? 0 : rank * 1152, 1152);
    if(local.dtype()!=ACL_BF16) throw std::runtime_error("query project storage");
    auto *input = execution.strided_descriptor(local.data(), 1152, {6, 1, 128}, {192, 128, 1}, ACL_BF16);
    auto *bank = weight_views.get(weight.data, 6 * 256 * 512, {6, 128, 512}, {256 * 512, 512, 1}, ACL_BF16);
    return batched_matrix(input, bank, 6, 512);
  }
  V query_rotary_extract(const V &q, bool local_heads) {
    auto local = resident::slice(q, local_heads ? 0 : rank * 1152, 1152);
    if(local.dtype()!=ACL_BF16) throw std::runtime_error("query extraction storage");
    auto *rotary_view = execution.strided_descriptor(static_cast<uint16_t *>(local.data()) + 128, 1024, {6, 64}, {192, 1}, ACL_BF16);
    DeviceTensor rotary({6,64},ACL_FLOAT,4);
    uint64_t bytes=0; aclOpExecutor *e=nullptr;
    checked(prepared::query("query_rotary_extract",false,aclnnCastGetWorkspaceSize,rotary_view,ACL_FLOAT,rotary.tensor,&bytes,&e),"query rotary extract prepare");
    run(aclnnCast,bytes,e);return result(rotary,ACL_FLOAT);
  }
  V absorb_query(const V &q, const std::string &weight_name, int pos, bool local_heads) {
    auto projected=query_project(q,weight_name,local_heads);
    auto rotary=query_rotary_extract(q,local_heads);
    return concat_columns(projected,rope(rotary,pos),6);
  }
  V concat_columns(const V &a, const V &b, int64_t rows) {
    if (rows <= 0 || a.dtype() != b.dtype() || a.size() % rows || b.size() % rows)
      throw std::runtime_error("column concatenation contract");
    int64_t aw = a.size() / rows, bw = b.size() / rows;
    DeviceTensor out({rows, aw + bw}, a.dtype(), resident::element_bytes(a.dtype()));
    auto descriptor = [&](const V &value, int64_t width) {
      int64_t shape[] = {rows, width}, strides[] = {width, 1};
      return aclCreateTensor(shape, 2, value.dtype(), strides, 0, ACL_FORMAT_ND,
                             shape, 2, value.data());
    };
    aclTensor *parts[] = {descriptor(a, aw), descriptor(b, bw)};
    if (!parts[0] || !parts[1]) {
      for (auto *part : parts) if (part) aclDestroyTensor(part);
      throw std::runtime_error("column concatenation descriptors");
    }
    struct List {
      aclTensorList *value;
      ~List() { if (value) aclDestroyTensorList(value); }
    } list{aclCreateTensorList(parts, 2)};
    if (!list.value) {
      for (auto *part : parts) aclDestroyTensor(part);
      throw std::runtime_error("column concatenation tensor list");
    }
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnCat", false, aclnnCatGetWorkspaceSize,
                            list.value, 1, out.tensor, &bytes, &e),
            "column concatenation prepare");
    run(aclnnCat, bytes, e);
    return result(out, a.dtype());
  }

  V batched_matrix(aclTensor *input, aclTensor *weight, int heads, int width) {
    DeviceTensor out({heads, 1, width}, ACL_BF16, 2);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnBatchMatMul", true,
                            aclnnBatchMatMulGetWorkspaceSize, input, weight,
                            out.tensor, 0, &bytes, &e),
            "batched matrix prepare");
    run(aclnnBatchMatMul, bytes, e);
    return result(out);
  }
  V expand_value(const V &values, const std::string &weight_name) {
    auto &weight = w(weight_name + ".weight");
    auto value = cast(values, ACL_BF16);
    View input(value.data(), {6, 1, 512}, ACL_BF16);
    auto *bank = weight_views.get(
        static_cast<uint16_t *>(weight.data) + 128 * 512, (6 * 256 - 128) * 512,
        {6, 512, 128}, {256 * 512, 1, 512}, ACL_BF16);
    return batched_matrix(input.t, bank, 6, 128);
  }
  V normalize_sinks(const std::string &state_name,
                    const std::string &weight_name,
                    const std::string &norm_name) {
    auto found = prepared_sinks.find(state_name);
    if (found != prepared_sinks.end())
      return found->second;
    prepared::Context::InitializationCopies declared(execution);
    auto &sk = w(weight_name + ".param_sink_compressed_kv");
    auto &sr = w(weight_name + ".param_sink_k_pe");
    V values;
    for (int i = 0; i < 128; ++i) {
      auto row = resident::from_device(
          static_cast<uint16_t *>(sk.data) + i * 512, 512, ACL_BF16);
      row = norm(row, norm_name);
      auto rot = resident::from_device(
          static_cast<uint16_t *>(sr.data) + i * 64, 64, ACL_BF16);
      row = resident::concat(row, rot);
      values = i ? resident::concat(values, row) : row;
    }
    prepared_sinks.emplace(state_name, values);
    return values;
  }
  void append_cache(const V &latent, const V &sink_values,
                    const std::string &state_name, int pos) {
    auto &ptr = attention_states[state_name];
    if (!ptr) {
      resident::TransferScope constants(resident::TransferRole::ControlInput);
      ptr = std::make_unique<AttentionState>(16,512,false,program.resident_sequences,program.shared_pages);
      prepared::Context::InitializationCopies declared(execution);
      to_device(ptr->sink, sink_values);
    }
    auto &state = *ptr;
    auto key = cast(latent, ACL_BF16);
    if (pos != current_position)
      throw std::runtime_error("cache position mismatch");
    View cache(state.key.data, {int64_t(state.physical_capacity), 576}, ACL_BF16);
    View source(key.data(), {1, 576}, ACL_BF16);
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    checked(prepared::query("aclnnInplaceIndexCopy", false,
                            aclnnInplaceIndexCopyGetWorkspaceSize, cache.t, 0,
                            cache_append_descriptor(state_name,ACL_INT32), source.t, &bytes,
                            &executor),
            "cache device index prepare");
    run(aclnnInplaceIndexCopy, bytes, executor);
  }

  void append_normalized_cache(const V &input,
                               const V &sinks, const std::string &norm_name,
                               const std::string &state_name, int pos, bool packed=false) {
    if (input.size() != 576 || input.dtype()!=ACL_BF16 || pos != current_position)
      throw std::runtime_error("fused cache shape/position contract");
    auto &ptr = attention_states[state_name];
    if (!ptr) {
      resident::TransferScope constants(resident::TransferRole::ControlInput);
      ptr = std::make_unique<AttentionState>(packed?8:16,((program.context+31)/32)*32,packed,program.resident_sequences,program.shared_pages);
      prepared::Context::InitializationCopies declared(execution);
      to_device(ptr->sink, sinks);
    }
    View kv(input.data(), {1, 1, 1, 576}, ACL_BF16);
    auto &gamma = w(norm_name + ".weight");
    View norm_weight(gamma.data, {512}, ACL_BF16);
    // This vendor API requires output descriptors even for a cache-only consumer.
    DeviceTensor rope_out({1, 1, 1, 64}, ACL_BF16, 2),
        latent_out({1, 1, 1, 512}, ACL_BF16, 2);
    // The vendor's absent separate rope cache selects combined [latent, rope]
    // storage, matching the existing paged cache contract. Position stays on device.
    char cache_mode[] = "PA", rotary_mode[] = "half";
    char quant_none[]="none", quant_c8[]="pertile128";
    char *quant_mode=packed?quant_c8:quant_none;
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnAiInfraKvRmsNormRopeCache", false,
                            aclnnAiInfraKvRmsNormRopeCacheGetWorkspaceSize,
                            kv.t, norm_weight.t, control_row(*rotary_cos_bf16,{1,1,1,64}),
                            control_row(*rotary_sin_bf16,{1,1,1,64}), cache_append_descriptor(state_name,ACL_INT64),
                            nullptr, ptr->key.tensor, nullptr, nullptr,
                            nullptr, nullptr, nullptr, 1e-5, cache_mode,
                            rotary_mode, quant_mode, true, rope_out.tensor, latent_out.tensor,
                            &bytes, &e),
            "fused cache normalization/rotary prepare");
    run(aclnnAiInfraKvRmsNormRopeCache, bytes, e);
  }
  V normalize_cache_row(const V &input, const std::string &norm_name) {
    if(input.size()!=576 || input.dtype()!=ACL_BF16)throw std::runtime_error("cache normalize contract");
    View kv(input.data(), {1,1,1,576},ACL_BF16);
    auto &gamma=w(norm_name+".weight"); View norm_weight(gamma.data,{512},ACL_BF16);
    DeviceTensor row({1,16,1,576},ACL_BF16,2);
    DeviceTensor rope_out({1,1,1,64},ACL_BF16,2),latent_out({1,1,1,512},ACL_BF16,2);
    char cache_mode[]="PA",rotary_mode[]="half",quant_mode[]="none";
    uint64_t bytes=0;aclOpExecutor *e=nullptr;
    checked(prepared::query("cache_normalize",false,aclnnAiInfraKvRmsNormRopeCacheGetWorkspaceSize,
        kv.t,norm_weight.t,rotary_cos_bf16->tensor,rotary_sin_bf16->tensor,zero_cache_position->tensor,
        nullptr,row.tensor,nullptr,nullptr,nullptr,nullptr,nullptr,1e-5,cache_mode,rotary_mode,quant_mode,
        true,rope_out.tensor,latent_out.tensor,&bytes,&e),"normalize cache row prepare");
    run(aclnnAiInfraKvRmsNormRopeCache,bytes,e);
    return result(row).view(0,576);
  }
  void quantize_cache(const V &input,const V &sinks,const std::string &name,unsigned bits) {
#ifdef IF_CACHE_KERNELS
    auto &ptr=attention_states[name];
    if(!ptr){
      resident::TransferScope constants(resident::TransferRole::ControlInput);
      ptr=std::make_unique<AttentionState>(bits,((program.context+15)/16)*16);
      prepared::Context::InitializationCopies declared(execution);
      to_device(ptr->sink,sinks);
    }
    if(ptr->cache_bits!=bits || input.size()!=576 || input.dtype()!=ACL_BF16)throw std::runtime_error("integer cache storage contract");
    execution.require_stream(stream);
    if_cache_quantize(input.data(),ptr->key.data,ptr->scales->data,cache_position->data,576,program.context,bits,stream);
#else
    (void)input;(void)sinks;(void)name;(void)bits;
    throw std::runtime_error("integer cache kernels not built");
#endif
  }
  V dequantize_cache(const std::string &name) {
#ifdef IF_CACHE_KERNELS
    auto &state=*attention_states.at(name);
    if(state.cache_bits==16 || !state.scales)throw std::runtime_error("cache dequantization contract");
    V decoded(state.cache_capacity*576,ACL_BF16);
    execution.require_stream(stream);
    if_cache_dequantize(state.key.data,state.scales->data,decoded.data(),cache_position->data,576,state.cache_capacity,state.cache_bits,stream);
    return decoded;
#else
    (void)name;throw std::runtime_error("integer cache kernels not built");
#endif
  }
  void prepare_packed_attention_metadata() {
#ifdef IF_CACHE_KERNELS
    if(!packed_indices) {
      packed_indices=std::make_unique<DeviceTensor>(std::vector<int64_t>{program.prefill_tokens,1,512},ACL_INT32,4);
      packed_length=std::make_unique<DeviceTensor>(std::vector<int64_t>{program.prefill_tokens},ACL_INT32,4);
    }
    execution.require_stream(stream);
    for(uint32_t row=0;row<active_tokens;++row)
      if_packed_attention_metadata(static_cast<int64_t *>(cache_position->data)+row,
          static_cast<int32_t *>(packed_indices->data)+row*512,
          static_cast<int32_t *>(packed_length->data)+row,512,stream);
#else
    throw std::runtime_error("packed attention metadata kernel not built");
#endif
  }
  V attend_packed(const V &query,const std::string &name) {
    auto &state=*attention_states.at(name);
    if(!state.packed_c8 || !packed_indices || query.dtype()!=ACL_BF16 || query.size()!=3456)
      throw std::runtime_error("packed attention contract");
    View q(query.data(),{1,6,576},ACL_BF16);
    auto *sink_values=weight_views.get(state.sink.data,128*576,{128,1,512},{576,576,1},ACL_BF16);
    char layout[]="TND",cache_layout[]="PA_BSND";
    uint64_t bytes=0;aclOpExecutor *e=nullptr;
    checked(prepared::query("aclnnAiInfraKvQuantSparseFlashAttention",false,
        aclnnAiInfraKvQuantSparseFlashAttentionGetWorkspaceSize,
        q.t,state.key.tensor,state.key.tensor,control_row(*packed_indices,{1,1,512}),nullptr,nullptr,
        state.table.tensor,state.qlen.tensor,control_row(*packed_length),state.sink.tensor,sink_values,
        1/std::sqrt(192.0),2,2,1,layout,cache_layout,3,INT64_MAX,INT64_MAX,2,1,128,64,
        state.out.tensor,&bytes,&e),"packed C8 attention prepare");
    run(aclnnAiInfraKvQuantSparseFlashAttention,bytes,e);
    return result(state.out);
  }
  void prepare_attention_metadata(bool cleared = false) {
    if (metadata_enqueued) return;
    if(!cleared) execution.zero(attention_metadata->data, attention_metadata->bytes);
    auto previous=active_row;
    for(active_row=0;active_row<active_tokens;++active_row)prepare_attention_metadata_row();
    active_row=previous;
    metadata_enqueued = true;
  }
  void prepare_attention_metadata_row() {
    char layout[] = "TND", cache_layout[] = "BnBsH";
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("attention_metadata", false,
        aclnnAiInfraFusedInferAttentionSinkMetadataGetWorkspaceSize,
        query_length64->tensor, control_row(*sequence_length64),
        6, 1, 512, 512, 1, 4, int64_t(program.context) - 1, 0,
        layout, cache_layout, 0, 128, 64, 16, cube_cores, vector_cores,
        control_row(*attention_metadata,{1024}), &bytes, &e), "attention metadata prepare");
    run(aclnnAiInfraFusedInferAttentionSinkMetadata, bytes, e);
  }
  V attend(const V &queries, const std::string &state_name, const V *decoded=nullptr) {
    if(!program.physical) prepare_attention_metadata();
    if(!metadata_enqueued) throw std::runtime_error("missing planned attention metadata");
    auto &state = *attention_states.at(state_name);
    if((state.cache_bits!=16) != bool(decoded))throw std::runtime_error("unplanned cache conversion");
    auto query = cast(queries, ACL_BF16);
    auto *q = execution.strided_descriptor(query.data(), 3456,
        {1, 6, 512}, {3456, 576, 1}, ACL_BF16);
    auto *qr = execution.strided_descriptor(
        static_cast<uint16_t *>(query.data()) + 512, 2944,
        {1, 6, 64}, {3456, 576, 1}, ACL_BF16);
    // Preserve the actual backing extent and view offset for paged cache views.
    auto cache_view = [&](int64_t width, int64_t offset) {
      int64_t shape[] = {state.physical_capacity/16, 16, width};
      int64_t storage[] = {state.physical_capacity/16, 16, 576};
      int64_t strides[] = {16 * 576, 576, 1};
      return aclCreateTensor(shape, 3, ACL_BF16, strides, offset, ACL_FORMAT_ND,
                             storage, 3, decoded?decoded->data():state.key.data);
    };
    struct List {
      aclTensorList *value = nullptr;
      explicit List(aclTensor *tensor) {
        if (!tensor) throw std::runtime_error("attention cache view");
        value = aclCreateTensorList(&tensor, 1);
        if (!value) {
          aclDestroyTensor(tensor);
          throw std::runtime_error("attention cache list");
        }
      }
      ~List() { aclDestroyTensorList(value); }
    } keys(cache_view(512, 0)), values(cache_view(512, 0));
    auto rope_cache = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>(
        cache_view(64, 512), aclDestroyTensor);
    if (!rope_cache) throw std::runtime_error("attention rotary cache view");
    auto *sink = weight_views.get(state.sink.data, 128 * 576,
                                  {128, 1, 512}, {576, 576, 1}, ACL_BF16);
    auto *sink_rope = weight_views.get(
        static_cast<uint16_t *>(state.sink.data) + 512, 128 * 576 - 512,
        {128, 1, 64}, {576, 576, 1}, ACL_BF16);
    View output(state.out.data, {1, 6, 512}, ACL_BF16);
    char layout[] = "TND";
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnAiInfraFusedInferAttentionSink", false,
        aclnnAiInfraFusedInferAttentionSinkGetWorkspaceSize,
        q, keys.value, values.value,
        nullptr, attention_mask->tensor, query_length64->tensor, control_row(*sequence_length64),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        state.table.tensor, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        qr, rope_cache.get(), nullptr, nullptr, control_row(*attention_metadata,{1024}),
        sink, sink_rope, sink,
        6, 1 / std::sqrt(192.0), int64_t(program.context) - 1, 0, layout,
        1, 4, 0, 16, 0, false, 0, 0, 0, 0,
        output.t, nullptr, &bytes, &e), "dense sink attention prepare");
    run(aclnnAiInfraFusedInferAttentionSink, bytes, e);
    return result(state.out);
  }
  V activate(const V &gate, const V &up) {
    return activate(active_tokens==1 ? resident::concat(gate, up) : concat_columns(gate,up,active_tokens));
  }
  V activate(const V &joined) {
    if (!joined.size() || joined.size() % 2)
      throw std::runtime_error("packed activation shape");
    if(joined.size()%(2*active_tokens))throw std::runtime_error("activation token rows");
    int width = joined.size() / (2*active_tokens);
    auto input = cast(joined, ACL_BF16);
    DeviceTensor output({active_tokens, width}, ACL_BF16, 2);
    auto descriptor = [&](void *data, int64_t columns,
                         uint32_t access) {
      IFKernelTensorV1 result{};
      result.struct_bytes = sizeof(result);
      result.dtype = 2;
      result.rank = 2;
      result.access = access;
      result.data = data;
      result.allocation_bytes = uint64_t(columns) * active_tokens * 2;
      result.shape[0] = active_tokens;
      result.shape[1] = columns;
      result.strides[0] = columns;
      result.strides[1] = 1;
      return result;
    };
    auto invocation =
        std::make_unique<inferfabric::PreparedKernel>(swiglu.prepare(
            {descriptor(input.data(), 2 * width, 1)}, {descriptor(output.data, width, 2)}));
    auto *workspace = execution.workspace(invocation->workspace_bytes());
    // Retain foreign invocation metadata until the explicit output boundary.
    pending_kernels.push_back(std::move(invocation));
    pending_kernels.back()->enqueue(workspace, execution.workspace_capacity(),
                                    stream);
    return result(output);
  }
  struct Routes {
    V ids, scores;
  };
  struct ExpertRows {
    V packed, groups, indices;
    std::vector<V> rows;
  };
  V project_route(const V &x, const std::string &p,
               const std::string &state_name) {
    auto &ptr = routers[state_name];
    if (!ptr) {
      ptr = std::make_unique<openpangu::Router>(program.prefill_tokens, 2560, 256, 8, stream);
      auto &gate = cpu(p + ".gate.weight");
      Host transposed(gate.size());
      for (int e = 0; e < 256; ++e)
        for (int j = 0; j < 2560; ++j)
          transposed[j * 256 + e] = gate[e * 2560 + j];
      resident::parameter_upload(ptr->weight, transposed);
      resident::parameter_upload(ptr->bias,
                                 cpu(p + ".e_score_correction_bias"));
    }
    auto value = cast(x, ACL_FLOAT);
    View input(value.data(), {active_tokens, 2560}, ACL_FLOAT);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("router_projection", false,
                            aclnnMatmulGetWorkspaceSize, input.t,
                            ptr->weight.tensor, prefix(ptr->logits,{active_tokens,256}), 0, &bytes, &e),
            "router projection prepare");
    run(aclnnMatmul, bytes, e);
    return result(ptr->logits, ACL_FLOAT).view(0,size_t(active_tokens)*256);
  }
  Routes select_route(const V &logits,const std::string &state_name) {
    auto &ptr=routers.at(state_name);
    if(logits.dtype()!=ACL_FLOAT || logits.data()!=ptr->logits.data) throw std::runtime_error("routing projection provenance");
    uint64_t bytes=0;aclOpExecutor *e=nullptr;
    checked(prepared::query("router_select",false,aclnnMoeGatingTopKGetWorkspaceSize,
        prefix(ptr->logits,{active_tokens,256}),ptr->bias.tensor,8,1,1,1,0,1,true,2.5,1e-20,
        prefix(ptr->scores,{active_tokens,8}),prefix(ptr->ids,{active_tokens,8}),prefix(ptr->normalized,{active_tokens,256}),&bytes,&e),"router topk prepare");
    run(aclnnMoeGatingTopK,bytes,e);
    return {result(ptr->ids, ACL_INT32).view(0,size_t(active_tokens)*8), result(ptr->scores, ACL_FLOAT).view(0,size_t(active_tokens)*8)};
  }
  Routes route(const V &x,const std::string &p,const std::string &state_name) {
    return select_route(project_route(x,p,state_name),state_name);
  }
  struct ExpertBank {
    V weights, grouped_scales;
    int n, k;
    ExpertBank(int count, int rows, int columns)
        : weights(size_t(count) * rows * columns, ACL_INT8),
          grouped_scales(size_t(count) * rows,ACL_BF16), n(rows), k(columns) {}
  };
  std::map<std::pair<std::string, std::string>, std::unique_ptr<ExpertBank>>
      expert_banks;
  V select(const V &source, const V &indices, size_t rows, size_t width) {
    if (source.size() != rows * width)
      throw std::runtime_error("indexed device selection shape");
    V out(indices.size() * width, source.dtype());
    View sv(source.data(), {int64_t(rows), int64_t(width)}, source.dtype());
    View ov(out.data(), {int64_t(indices.size()), int64_t(width)},
            source.dtype());
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnIndexSelect", false,
                            aclnnIndexSelectGetWorkspaceSize, sv.t, 0,
                            indices.tensor(), ov.t, &bytes, &e),
            "device index prepare");
    run(aclnnIndexSelect, bytes, e);
    return out;
  }
  ExpertBank &bank(const std::string &pattern, const std::string &paired = "") {
    auto &ptr = expert_banks[{pattern, paired}];
    if (ptr)
      return *ptr;
    prepared::Context::InitializationCopies declared(execution);
    auto &first = w(expert_name(pattern, rank * 32) + ".weight");
    int n = first.descriptor.shape[0], k = first.descriptor.shape[1];
    if (first.dtype != inferfabric::initialization::DType::I8)
      throw std::runtime_error("resident expert requires int8 contract");
    int projections = paired.empty() ? 1 : 2;
    ptr = std::make_unique<ExpertBank>(32, n * projections, k);
    for (int j = 0; j < 32; ++j) {
      for (int projection = 0; projection < projections; ++projection) {
        auto name = expert_name(projection ? paired : pattern, rank * 32 + j);
        auto &weight = w(name + ".weight"), &scale = w(name + ".weight_scale");
        if (source_types.at(scale.descriptor.id) !=
            inferfabric::initialization::DType::Bf16)
          throw std::runtime_error(
              "grouped expert requires lossless BF16 checkpoint scales");
        if (weight.dtype != inferfabric::initialization::DType::I8 ||
            weight.descriptor.shape != first.descriptor.shape ||
            weight.bytes != size_t(n) * k || scale.bytes != size_t(n) * 2 || scale.dtype != inferfabric::initialization::DType::Bf16)
          throw std::runtime_error("expert bank shape");
        resident::copy(static_cast<char *>(ptr->weights.data()) +
                           size_t(j * projections + projection) * n * k,
                       weight.data, weight.bytes);
        resident::copy(static_cast<uint16_t *>(ptr->grouped_scales.data()) +
                           size_t(j * projections + projection) * n,
                       scale.data, scale.bytes);
      }
    }
    return *ptr;
  }
  struct RoutedQuantized { V values,scales,groups,indices; };
  RoutedQuantized quantize_experts(const V &batch, const Routes &routes,
                            int k, bool fused = false,
                            const ExpertRows *source = nullptr) {
    const int rows = 8*active_tokens;
    if (batch.dtype() != ACL_BF16 ||
        batch.size() != size_t(k * (fused ? rows * 2 : active_tokens)))
      throw std::runtime_error("grouped expert input contract");
    V groups, indices;
    if (fused) {
      if (!source || source->groups.size() != 32 || source->indices.size() != size_t(rows))
        throw std::runtime_error("fused expert routing provenance");
      groups = source->groups;
      indices = source->indices;
    } else {
      groups = V(32, ACL_INT64);
      indices = V(rows, ACL_INT32);
    }
    View input(batch.data(), {fused ? rows : int(active_tokens), k * (fused ? 2 : 1)}, ACL_BF16);
    View group_view(groups.data(), {32}, ACL_INT64);
    DeviceTensor quant({rows, k}, ACL_INT8, 1),
        scale({rows}, ACL_FLOAT, 4);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    if (fused) {
      char mode[] = "dynamic";
      checked(prepared::query("aclnnDequantSwigluQuant", false,
                              aclnnDequantSwigluQuantGetWorkspaceSize, input.t,
                              nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, true, mode, quant.tensor, scale.tensor,
                              &bytes, &e),
              "grouped expert activation prepare");
      run(aclnnDequantSwigluQuant, bytes, e);
    } else {
      // One device operation expands/quantizes the token and filters/sorts
      // routes into this rank's expert interval. Invalid inverse indices are -1.
      View ids(routes.ids.data(), {active_tokens, 8}, ACL_INT32);
      int64_t bounds[] = {rank * 32, (rank + 1) * 32};
      auto *range = aclCreateIntArray(bounds, 2);
      if (!range)
        throw std::runtime_error("expert interval descriptor");
      auto status = prepared::query("aclnnMoeInitRoutingV3", false,
                                     aclnnMoeInitRoutingV3GetWorkspaceSize,
                                     input.t, ids.t, nullptr, nullptr,
                                     rows, 0, 256, 0, 1, true, 1, range, 0,
                                     quant.tensor, indices.tensor(), group_view.t,
                                     scale.tensor, &bytes, &e);
      aclDestroyIntArray(range);
      checked(status, "fused expert routing/quantization prepare");
      run(aclnnMoeInitRoutingV3, bytes, e);
    }
    return {result(quant,ACL_INT8),result(scale,ACL_FLOAT),groups,indices};
  }
  ExpertRows zero_expert_output(const RoutedQuantized &q,int width) {
    ExpertRows output;output.packed=V(size_t(8)*active_tokens*width,ACL_BF16);
    output.groups=q.groups;output.indices=q.indices;
    execution.zero(output.packed.data(),output.packed.bytes());
    for(uint32_t i=0;i<8*active_tokens;++i)output.rows.push_back(output.packed.view(size_t(i)*width,width));
    return output;
  }
  ExpertRows expert_gmm(const RoutedQuantized &q,ExpertRows output,ExpertBank &bank) {
    const int rows=8*active_tokens;
    if(q.values.dtype()!=ACL_INT8 || q.values.size()!=size_t(rows*bank.k) || q.scales.dtype()!=ACL_FLOAT || q.scales.size()!=size_t(rows) || q.groups.size()!=32 || output.packed.dtype()!=ACL_BF16 || output.packed.size()!=size_t(rows*bank.n) || output.groups.data()!=q.groups.data() || output.indices.data()!=q.indices.data()) throw std::runtime_error("grouped matmul physical contract");
    View group_view(q.groups.data(),{32},ACL_INT64);uint64_t bytes=0;aclOpExecutor *e=nullptr;
    // ACL tensor lists own their element descriptors. Give them independent
    // descriptors, never pointers owned by the prepared descriptor cache.
    struct List {
      aclTensorList *value;
      List(void *data, std::vector<int64_t> shape, aclDataType dtype,
           bool transpose = false) {
        auto dims = shape;
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t j = shape.size() - 1; j > 0; --j)
          strides[j - 1] = strides[j] * shape[j];
        if (transpose) {
          std::swap(dims[dims.size() - 2], dims.back());
          std::swap(strides[strides.size() - 2], strides.back());
        }
        auto *tensor =
            aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0,
                            ACL_FORMAT_ND, shape.data(), shape.size(), data);
        if (!tensor)
          throw std::runtime_error("grouped tensor descriptor");
        value = aclCreateTensorList(&tensor, 1);
        if (!value) {
          aclDestroyTensor(tensor);
          throw std::runtime_error("grouped tensor list");
        }
      }
      ~List() { aclDestroyTensorList(value); }
    } xs(q.values.data(), {rows, bank.k}, ACL_INT8),
        ws(bank.weights.data(), {32, bank.n, bank.k}, ACL_INT8, true),
        ss(bank.grouped_scales.data(), {32, bank.n}, ACL_BF16),
        ts(q.scales.data(), {rows}, ACL_FLOAT),
        ys(output.packed.data(), {rows, bank.n}, ACL_BF16);
    checked(prepared::query("aclnnGroupedMatmulV5", false,
                            aclnnGroupedMatmulV5GetWorkspaceSize, xs.value,
                            ws.value, nullptr, ss.value, nullptr, nullptr,
                            nullptr, ts.value, group_view.t, nullptr, nullptr,
                            nullptr, 3, 0, 1, 0, nullptr, ys.value, nullptr,
                            nullptr, &bytes, &e),
            "grouped expert matmul prepare");
    run(aclnnGroupedMatmulV5, bytes, e);
    return output;
  }
  ExpertRows grouped_linear(const V &batch,const Routes &routes,ExpertBank &bank,bool fused=false,const ExpertRows *source=nullptr) {
    auto q=quantize_experts(batch,routes,bank.k,fused,source);
    return expert_gmm(q,zero_expert_output(q,bank.n),bank);
  }
  ExpertRows pack_expert_rows(const ExpertRows &a,const ExpertRows &b) {
    if(a.groups.data()!=b.groups.data() || a.indices.data()!=b.indices.data()) throw std::runtime_error("expert packing provenance");
    ExpertRows result;result.groups=a.groups;result.indices=a.indices;
    std::vector<V> parts;for(size_t j=0;j<a.rows.size();++j){parts.push_back(a.rows[j]);parts.push_back(b.rows.at(j));}
    result.packed=resident::concat(parts);
    for(size_t j=0;j<a.rows.size();++j)result.rows.push_back(result.packed.view(j*(a.rows[j].size()+b.rows[j].size()),a.rows[j].size()+b.rows[j].size()));
    return result;
  }
  V weighted_rows(const ExpertRows &rows, const V &scores, size_t width, const V *prepared_input=nullptr) {
    if (scores.dtype() != ACL_FLOAT ||
        rows.packed.size() != scores.size() * width ||
        rows.indices.size() != scores.size())
      throw std::runtime_error("weighted expert reduction contract");
    auto input = prepared_input ? *prepared_input : cast(rows.packed, ACL_FLOAT);
    if(input.dtype()!=ACL_FLOAT || input.size()!=rows.packed.size()) throw std::runtime_error("planned expert reduction storage");
    V out(width*active_tokens);
    View x(input.data(), {int64_t(scores.size()), 1, int64_t(width)}, ACL_FLOAT),
        s(scores.data(), {active_tokens, 8}, ACL_FLOAT),
        y(out.data(), {active_tokens, int64_t(width)}, ACL_FLOAT);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnMoeFinalizeRoutingV2", false,
                            aclnnMoeFinalizeRoutingV2GetWorkspaceSize, x.t,
                            rows.indices.tensor(), nullptr, nullptr,
                            nullptr, s.t, nullptr, 3, y.t, &bytes, &e),
            "fused expert reduction prepare");
    run(aclnnMoeFinalizeRoutingV2, bytes, e);
    return out;
  }
  int emit(const V &resident_logits, int step, bool diagnostic = true) {
    auto logits = cast(resident_logits, ACL_FLOAT,true);
    V top(5), ids(5, ACL_INT64), probs(logits.size());
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnTopk", false, aclnnTopkGetWorkspaceSize,
                            logits.tensor(), 5, 0, true, true, top.tensor(),
                            ids.tensor(), &bytes, &e),
            "device output topk prepare");
    run(aclnnTopk, bytes, e);
    checked(prepared::query("aclnnLogSoftmax", false,
                            aclnnLogSoftmaxGetWorkspaceSize, logits.tensor(), 0,
                            probs.tensor(), &bytes, &e),
            "device logsoftmax prepare");
    run(aclnnLogSoftmax, bytes, e);
    auto selected_probs = select(probs, ids, logits.size(), 1);
    V selected(1, ACL_INT64), peers(8, ACL_INT64);
    checked(prepared::query("aclnnArgMax", false, aclnnArgMaxGetWorkspaceSize,
                            logits.tensor(), 0, true, selected.tensor(), &bytes,
                            &e),
            "device greedy argmax prepare");
    run(aclnnArgMax, bytes, e);
    checked(HcclAllGather(selected.data(), peers.data(), 1,
                          HCCL_DATA_TYPE_INT64, comm, stream),
            "device token consensus");
    resident::TransferScope boundary(resident::TransferRole::RequestOutput);
    // Queue all explicitly permitted output copies on the owning stream before
    // its single completion boundary. No synchronous copy/global fence.
    auto *host_output = static_cast<char *>(execution.output_host(144));
    checked(aclrtMemcpyAsync(host_output, 64, peers.data(), 64,
                             ACL_MEMCPY_DEVICE_TO_HOST, stream),
            "output consensus IDs");
    if (rank == 0) {
      checked(aclrtMemcpyAsync(host_output + 64, 40, ids.data(), 40,
                               ACL_MEMCPY_DEVICE_TO_HOST, stream),
              "output top IDs");
      checked(aclrtMemcpyAsync(host_output + 104, 20, selected_probs.data(), 20,
                               ACL_MEMCPY_DEVICE_TO_HOST, stream),
              "output top log probabilities");
      checked(aclrtMemcpyAsync(host_output + 124, 20, top.data(), 20,
                               ACL_MEMCPY_DEVICE_TO_HOST, stream),
              "output top scores");
    }
    execution.complete();
    pending_kernels.clear();
    std::array<int64_t, 8> peer_ids{};
    std::array<int64_t, 5> order{};
    std::array<float, 5> lp{}, scores{};
    std::memcpy(peer_ids.data(), host_output, 64);
    int next = int(peer_ids[0]);
    for (auto id : peer_ids)
      if (id != next)
        throw std::runtime_error("ranks disagree on token");
    if (rank == 0 && diagnostic) {
      std::memcpy(order.data(), host_output + 64, 40);
      std::memcpy(lp.data(), host_output + 104, 20);
      std::memcpy(scores.data(), host_output + 124, 20);
      std::cout << "{\"step\":" << step << ",\"token\":" << next
                << ",\"logit\":" << scores[0] << ",\"top5\":[";
      for (int j = 0; j < 5; ++j) {
        if (j)
          std::cout << ',';
        std::cout << "{\"token\":" << order[j] << ",\"logprob\":" << lp[j]
                  << '}';
      }
      std::cout << "]}" << std::endl;
    }
    return next;
  }
  uint32_t vocabulary() const {
    for (const auto &node : program.nodes) if (node.op == "embedding") return node.integer("vocab");
    throw std::runtime_error("missing embedding vocabulary contract");
  }
  aclTensor *cache_append_descriptor(const std::string &name, aclDataType dtype) {
    if(dtype!=ACL_INT64 && dtype!=ACL_INT32)throw std::runtime_error("cache control dtype");
    auto offset=cache_control_offsets.at(name)+active_row*16+(dtype==ACL_INT32?8:0);
    return execution.descriptor(static_cast<uint8_t *>(cache_controls->data)+offset,{1},dtype);
  }
  void set_chunk_inputs(const std::vector<uint32_t> &tokens, int pos) {
    auto count=tokens.size();
    if((count!=1 && count!=program.prefill_tokens) || pos<0 || size_t(pos)+count>program.context)
      throw std::runtime_error("compiled token chunk bounds");
    for(auto id:tokens)if(id>=vocabulary())throw std::runtime_error("chunk token vocabulary");
    active_tokens=count;active_row=0;current_position=pos;
    std::vector<int32_t> ids(program.prefill_tokens,0), positions(program.prefill_tokens,0);
    std::vector<int64_t> positions64(program.prefill_tokens,0), lengths(program.prefill_tokens,0);
    Host c(program.prefill_tokens*64,0), sn(program.prefill_tokens*64,0);
    std::vector<uint16_t> cb(program.prefill_tokens*64,0), sb(program.prefill_tokens*64,0);
    for(size_t row=0;row<count;++row) {
      int position=pos+row;ids[row]=tokens[row];positions[row]=position;
      positions64[row]=position;lengths[row]=position+1;
      for(const auto &entry:cache_control_offsets) {
        auto state=attention_states.find(entry.first);
        uint64_t physical=state==attention_states.end()?uint64_t(position):state->second->page_binding.append(position);
        if(physical>INT32_MAX)throw std::runtime_error("physical append position overflow");
        int64_t wide=int64_t(physical);int32_t narrow=int32_t(physical);
        std::memcpy(cache_control_host.data()+entry.second+row*16,&wide,sizeof(wide));
        std::memcpy(cache_control_host.data()+entry.second+row*16+8,&narrow,sizeof(narrow));
      }
      for(int i=0;i<32;++i) {
        float angle=float(position)/std::pow(6400000.f,float(2*i)/64);
        c[row*64+i]=c[row*64+i+32]=roundbf(std::cos(angle));
        sn[row*64+i]=sn[row*64+i+32]=roundbf(std::sin(angle));
      }
      for(size_t i=0;i<64;++i){cb[row*64+i]=tobf(c[row*64+i]);sb[row*64+i]=tobf(sn[row*64+i]);}
    }
    if(cache_controls)resident::control_upload(*cache_controls,cache_control_host);
    resident::control_upload(*token_index,ids);resident::control_upload(*position_index,positions);
    resident::control_upload(*cache_position,positions64);resident::control_upload(*sequence_length64,lengths);
    resident::control_upload(*rotary_cos,c);resident::control_upload(*rotary_sin,sn);
    resident::control_upload(*rotary_cos_bf16,cb);resident::control_upload(*rotary_sin_bf16,sb);
  }
  void set_token_inputs(int id,int pos) {
    if(id<0)throw std::runtime_error("negative token");
    set_chunk_inputs({uint32_t(id)},pos);
  }

  V token(int id, int pos, bool decode = false) {
    set_token_inputs(id, pos);
    execution.begin_step(pos);
    V output;
    if (decode && std::getenv("INFERFABRIC_DECODE_GRAPH")) {
      if (!decode_graph) {
        decode_graph = execution.capture("token_decode", [&] {
          pending_kernels.clear();
          graph_output = execute_token();
        });
        graph_kernels = std::move(pending_kernels);
      }
      execution.replay(decode_graph);
      output = graph_output;
    } else
      output = execute_token();
    execution.end_step();
    return output;
  }
  // A request boundary is explicit control work. All preceding work has completed.
  V prefill(const std::vector<uint32_t> &tokens,int pos,bool replay=true) {
    if(tokens.size()!=program.prefill_tokens || tokens.size()<=1)throw std::runtime_error("prefill variant size");
    set_chunk_inputs(tokens,pos);execution.begin_step(pos);
    V output;
    if(replay) {
      if(!prefill_graph) {
        prefill_graph=execution.capture("packed_prefill",[&]{ pending_kernels.clear();prefill_output=execute_token(); });
        prefill_kernels=std::move(pending_kernels);
      }
      execution.replay(prefill_graph);output=prefill_output;
    } else output=execute_token();
    execution.end_step();
    return output.view((tokens.size()-1)*vocabulary(),vocabulary());
  }
  template<class F> void each_row(F f) {
    struct Restore { Decoder &d;int pos;uint32_t row;~Restore(){d.current_position=pos;d.active_row=row;} } restore{*this,current_position,active_row};
    for(uint32_t row=0;row<active_tokens;++row){active_row=row;current_position=restore.pos+row;f(row);}
  }
  template<class F> V row_values(size_t width,F f) {
    if(active_tokens==1)return f(0);
    V result;
    each_row([&](uint32_t row){
      V value=f(row);
      if(value.size()!=width)throw std::runtime_error("packed row result width");
      if(!row)result=V(width*active_tokens,value.dtype());
      if(value.dtype()!=result.dtype())throw std::runtime_error("packed row result dtype");
      resident::copy(static_cast<uint8_t *>(result.data())+row*value.bytes(),value.data(),value.bytes());
    });
    return result;
  }
  // No model-specific layer traversal: only instantiated state bindings are reset.
  void reset_request() {
    for (auto &entry : mome_states)
      execution.zero(entry.second->state.data, entry.second->state.bytes);
    for (auto &entry : attention_states) {
      auto &s = *entry.second;
      execution.zero(s.key.data, s.key.bytes);
      if (s.scales) execution.zero(s.scales->data, s.scales->bytes);
    }
    execution.complete();
    select_resident(0, -1);
    current_position = -1;
    execution.clear_step_history();
  }
  void select_resident(uint32_t slot, int last_position) {
    if(slot>=program.resident_sequences || last_position < -1 || last_position>=int(program.context))
      throw std::runtime_error("resident selection bounds");
    if(slot!=active_resident) {
      resident::control_upload(*resident_index,std::vector<int32_t>{int32_t(slot)});
      for(auto &entry:attention_states) {
        auto &state=*entry.second;
        state.page_binding.select(slot);
        upload_page_table(state);
      }
      active_resident=slot;
    }
    current_position=last_position;
  }
  void reset_resident(uint32_t slot) {
    if(slot>=program.resident_sequences)throw std::runtime_error("resident reset bounds");
    for(auto &entry:mome_states) {
      auto &state=entry.second->state;
      auto bytes=state.bytes/program.resident_sequences;
      execution.zero(static_cast<uint8_t *>(state.data)+slot*bytes,bytes);
    }
    for(auto &entry:attention_states) {
      auto &state=*entry.second;
      if(program.shared_pages) {
        auto page_tokens=state.packed_c8?32:16;
        auto page_bytes=state.key.bytes/(state.physical_capacity/page_tokens);
        for(auto page:state.page_binding.pages(slot))
          execution.zero(static_cast<uint8_t *>(state.key.data)+size_t(page)*page_bytes,page_bytes);
        state.page_binding.release(slot);
        if(slot==active_resident)upload_page_table(state);
      } else {
        auto bytes=state.key.bytes/program.resident_sequences;
        execution.zero(static_cast<uint8_t *>(state.key.data)+slot*bytes,bytes);
      }
      if(state.scales)execution.zero(state.scales->data,state.scales->bytes);
    }
    execution.complete();
    if(slot==active_resident)current_position=-1;
  }
  void upload_page_table(AttentionState &state) {
    const auto &pages=state.page_binding.pages();
    if(pages.size()>32)throw std::runtime_error("device block table capacity");
    std::vector<int32_t> table(32,0);
    std::copy(pages.begin(),pages.end(),table.begin());
    resident::control_upload(state.table,table);
  }
  // Preparation uses complete partitions to exercise every shape. Serving starts
  // with no cache ownership; the external allocator supplies request reservations.
  void release_preparation_pages() {
    if(!program.shared_pages)return;
    for(auto &entry:attention_states) {
      auto &state=*entry.second;
      for(uint32_t slot=0;slot<program.resident_sequences;++slot)state.page_binding.release(slot);
      upload_page_table(state);
    }
    execution.complete();
  }
  void bind_cache_pages(const std::string &name, const std::vector<int32_t> &pages) {
    if (current_position != -1) throw std::runtime_error("page binding requires a reset request");
    auto found = attention_states.find(name);
    if (found == attention_states.end()) throw std::runtime_error("unknown cache state binding");
    auto &state = *found->second;
    if (state.cache_bits != 16 && !state.packed_c8)
      throw std::runtime_error("row-quantized cache paging is not implemented");
    if (pages.size()>32) throw std::runtime_error("device block table capacity");
    state.page_binding.bind(pages);
    upload_page_table(state);
    // No graph pointer changes: only declared control metadata is updated.
  }
  // Runtime dispatch is defined separately from kernel implementations.
#include "token_dispatch.inc"
};
} // namespace inferfabric::token

// Device-resident execution of a validated generic token program.
// No activation or routing value has a host representation.
#include "ascend_token_contract.h"
#include "compiled_kernel.h"
#include "execution_capture.h"
#include "initialization_acl.h"
#include "openpangu_decoder_state.h"
#include "openpangu_mhc.h"
#include "resident_tensor.h"
#include "resident_transfer.h"
#include <aclnn_ai_infra_fused_causal_conv1d.h>
#include <aclnn_ai_infra_sparse_flash_attention_pioneer.h>
#include <aclnnop/aclnn_dequant_swiglu_quant.h>
#include <aclnnop/aclnn_dynamic_quant.h>
#include <aclnnop/aclnn_grouped_matmul_v5.h>
#include <aclnnop/aclnn_index_copy.h>
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
namespace pangu {
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
  std::vector<std::unique_ptr<inferfabric::PreparedKernel>> pending_kernels;
  aclmdlRI decode_graph = nullptr;
  V graph_output;
  std::vector<std::unique_ptr<inferfabric::PreparedKernel>> graph_kernels;
  std::unique_ptr<DeviceTensor> token_index, position_index, sequence_length,
      causal_indices, rotary_cos, rotary_sin;
  int rank;
  int current_position = -1;
  std::map<std::string, std::unique_ptr<openpangu::Router>> routers;
  const inferfabric::execution::Plan &program;
  std::map<std::string, std::unique_ptr<MhcState>> mhc_states;
  std::map<std::string, std::unique_ptr<MomeState>> mome_states;
  std::map<std::string, std::unique_ptr<AttentionState>> attention_states;
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
          HcclRootInfo &info, const inferfabric::execution::Plan &execution)
      : rank(p.rank), program(execution),
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
    token_index =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{1}, ACL_INT32, 4);
    position_index =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{1}, ACL_INT32, 4);
    sequence_length =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{1}, ACL_INT32, 4);
    causal_indices = std::make_unique<DeviceTensor>(
        std::vector<int64_t>{1, 1, int64_t(program.context)}, ACL_INT32, 4);
    rotary_cos =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{64}, ACL_FLOAT, 4);
    rotary_sin =
        std::make_unique<DeviceTensor>(std::vector<int64_t>{64}, ACL_FLOAT, 4);
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
  V cast(const V &x, aclDataType dtype) {
    if (x.dtype() == dtype)
      return x;
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
    return result(b);
  }
  V sum(const V &x) {
    auto a = cast(x, ACL_FLOAT);
    DeviceTensor b({int64_t(x.size())}, ACL_FLOAT, 4);
    {
      inferfabric::profiling::Scope timing("HcclAllReduce");
      checked(HcclAllReduce(a.data(), b.data, x.size(), HCCL_DATA_TYPE_FP32,
                            HCCL_REDUCE_SUM, comm, stream),
              "all reduce");
    }
    return result(b, ACL_FLOAT);
  }
  V matrix(const V &x, void *weight, int rows, int cols,
           bool transposed = true) {
    int output = transposed ? rows : cols;
    auto value = cast(x, ACL_BF16);
    View input(value.data(), {1, int64_t(x.size())}, ACL_BF16);
    DeviceTensor out({1, output}, ACL_BF16, 2);
    View view(weight, {rows, cols}, ACL_BF16, transposed);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnMatmul", true, aclnnMatmulGetWorkspaceSize,
                            input.t, view.t, out.tensor, 0, &bytes, &e),
            "BF16 matmul prepare");
    run(aclnnMatmul, bytes, e);
    return result(out);
  }
  V linear(const std::string &name, const V &x, bool distributed = false) {
    auto &a = w(name + ".weight");
    int n = a.descriptor.shape[0], k = a.descriptor.shape[1];
    if (int(x.size()) != k)
      throw std::runtime_error("linear shape " + name);
    V result;
    if (a.dtype == inferfabric::initialization::DType::Bf16)
      result = matrix(x, a.data, n, k);
    else {
      auto &s = w(name + ".weight_scale");
      auto value = cast(x, ACL_BF16);
      View input(value.data(), {1, k}, ACL_BF16);
      DeviceTensor quant({1, k}, ACL_INT8, 1), scale({1}, ACL_FLOAT, 4),
          out({1, n}, ACL_BF16, 2);
      View weight(a.data, {n, k}, ACL_INT8), scales(s.data, {n}, ACL_FLOAT);
      uint64_t bytes = 0;
      aclOpExecutor *e = nullptr;
      checked(prepared::query("aclnnDynamicQuant", false,
                              aclnnDynamicQuantGetWorkspaceSize, input.t,
                              nullptr, quant.tensor, scale.tensor, &bytes, &e),
              "quant prepare");
      run(aclnnDynamicQuant, bytes, e);
      checked(prepared::query("aclnnQuantMatmulV5", false,
                              aclnnQuantMatmulV5GetWorkspaceSize, quant.tensor,
                              weight.t, scale.tensor, scales.t, nullptr,
                              nullptr, nullptr, nullptr, nullptr, false, true,
                              0, out.tensor, &bytes, &e),
              "int8 matmul prepare");
      run(aclnnQuantMatmulV5, bytes, e);
      result = this->result(out);
    }
    auto output = distributed ? gather(result) : result;
    trace(name, output);
    return output;
  }
  void trace(const std::string &, const V &) {}
  V norm(const V &x, const std::string &name) {
    auto &gamma = w(name + ".weight");
    if (gamma.bytes != x.size() * 2)
      throw std::runtime_error("norm shape");
    auto value = cast(x, ACL_BF16);
    View input(value.data(), {1, int64_t(x.size())}, ACL_BF16);
    DeviceTensor out({1, int64_t(x.size())}, ACL_BF16, 2),
        rstd({1, 1}, ACL_FLOAT, 4);
    View weight(gamma.data, {int64_t(x.size())}, ACL_BF16);
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
         const std::string &state_name) {
    auto &ptr = mome_states[state_name];
    if (!ptr) {
      resident::TransferScope constants(resident::TransferRole::ControlInput);
      ptr = std::make_unique<MomeState>(x.size());
      auto &source = cpu(name + ".weight");
      std::vector<uint16_t> weight(source.size());
      for (size_t j = 0; j < x.size(); ++j)
        for (int tap = 0; tap < 3; ++tap)
          weight[tap * x.size() + j] = tobf(source[j * 3 + tap]);
      resident::parameter_upload(ptr->weight, weight);
    }
    auto &s = *ptr;
    to_device(s.input, x);
    if (position != current_position)
      throw std::runtime_error("position input mismatch");
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnAiInfraFusedCausalConv1d", false,
                            aclnnAiInfraFusedCausalConv1dGetWorkspaceSize,
                            s.input.tensor, s.weight.tensor, s.state.tensor,
                            s.starts.tensor, s.slots.tensor, nullptr, nullptr,
                            nullptr, position_index->tensor, nullptr, nullptr,
                            nullptr, 0, -1, 0, 1, 1, 128, 1, true, s.out.tensor,
                            &bytes, &e),
            "MoME prepare");
    run(aclnnAiInfraFusedCausalConv1d, bytes, e);
    return result(s.input);
  }
  struct HC {
    MhcState *device = nullptr;
    V input, post, res;
  };
  HC pre(const V &x, const std::string &name, bool merge,
         const std::string &state_name) {
    auto &ptr = mhc_states[state_name];
    if (!ptr) {
      ptr = std::make_unique<MhcState>();
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
    to_device(s.x, x);
    auto &api = training_api();
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("mHC.pre", false, api.pre_size, s.x.tensor,
                            s.phi.tensor, s.alpha.tensor, s.beta.tensor,
                            nullptr, 0, 1e-6, 1e-6, s.input.tensor,
                            s.post.tensor, s.res.tensor, s.rms.tensor,
                            s.mix.tensor, s.pre_weights.tensor, &bytes, &e),
            "mHC pre prepare");
    run(api.pre, bytes, e);
    if (!merge) {
      checked(prepared::query("mHC.sink", false, api.sink_size, s.res.tensor, 0,
                              1e-6, 20, s.sink.tensor, nullptr, nullptr, &bytes,
                              &e),
              "mHC Sinkhorn prepare");
      run(api.sink, bytes, e);
    }
    HC h;
    h.device = &s;
    h.input = result(s.input);
    return h;
  }
  V post(const V &, const V &branch, const HC &h) {
    if (!h.device)
      throw std::runtime_error("missing mHC pre state");
    auto &s = *h.device;
    to_device(s.branch, branch);
    auto &api = training_api();
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("mHC.post", false, api.post_size, s.x.tensor,
                            s.sink.tensor, s.branch.tensor, s.post.tensor,
                            s.out.tensor, &bytes, &e),
            "mHC post prepare");
    run(api.post, bytes, e);
    return result(s.out);
  }
  V rope(const V &input, int pos) {
    if (input.size() != 64 || pos != current_position)
      throw std::runtime_error("rotary shape/position contract");
    // Preserve FP32 products/sum and the existing final BF16 rounding while
    // fusing the half rotation, products and sum into one vendor operation.
    auto value = cast(input, ACL_FLOAT);
    DeviceTensor out({1, 1, 1, 64}, ACL_FLOAT, 4);
    View x(value.data(), {1, 1, 1, 64}, ACL_FLOAT),
        cos(rotary_cos->data, {1, 1, 1, 64}, ACL_FLOAT),
        sin(rotary_sin->data, {1, 1, 1, 64}, ACL_FLOAT);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnRotaryPositionEmbedding", false,
                            aclnnRotaryPositionEmbeddingGetWorkspaceSize, x.t,
                            cos.t, sin.t, 0, out.tensor, &bytes, &e),
            "rotary prepare");
    run(aclnnRotaryPositionEmbedding, bytes, e);
    return rounded(result(out, ACL_FLOAT));
  }
  V absorb_query(const V &q, const std::string &weight_name, int pos,
                 bool local_heads) {
    auto &weight = w(weight_name + ".weight");
    std::vector<V> queries;
    for (int head = 0; head < 6; ++head) {
      size_t offset = ((local_heads ? 0 : rank * 6) + head) * 192;
      auto no = resident::slice(q, offset, 128),
           rot = rope(resident::slice(q, offset + 128, 64), pos);
      auto query =
          matrix(no, static_cast<uint16_t *>(weight.data) + head * 256 * 512,
                 128, 512, false);
      query = resident::concat(query, rot);
      queries.push_back(query);
    }
    return resident::concat(queries);
  }
  V expand_value(const V &values, const std::string &weight_name) {
    auto &weight = w(weight_name + ".weight");
    std::vector<V> local;
    for (int head = 0; head < 6; ++head) {
      auto value = resident::slice(values, head * 512, 512);
      auto out = matrix(value,
                        static_cast<uint16_t *>(weight.data) +
                            (head * 256 + 128) * 512,
                        128, 512);
      local.push_back(out);
    }
    return resident::concat(local);
  }
  V normalize_sinks(const std::string &state_name,
                    const std::string &weight_name,
                    const std::string &norm_name) {
    auto found = prepared_sinks.find(state_name);
    if (found != prepared_sinks.end())
      return found->second;
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
      ptr = std::make_unique<AttentionState>();
      to_device(ptr->sink, sink_values);
    }
    auto &state = *ptr;
    auto key = cast(latent, ACL_BF16);
    if (pos != current_position)
      throw std::runtime_error("cache position mismatch");
    View cache(state.key.data, {int64_t(program.context), 576}, ACL_BF16);
    View source(key.data(), {1, 576}, ACL_BF16);
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
    checked(prepared::query("aclnnInplaceIndexCopy", false,
                            aclnnInplaceIndexCopyGetWorkspaceSize, cache.t, 0,
                            position_index->tensor, source.t, &bytes,
                            &executor),
            "cache device index prepare");
    run(aclnnInplaceIndexCopy, bytes, executor);
  }

  V attend(const V &queries, const std::string &state_name) {
    auto &state = *attention_states.at(state_name);
    to_device(state.query, queries);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    char layout[] = "TND", cache_layout[] = "PA_BSND";
    checked(prepared::query(
                "aclnnAiInfraSparseFlashAttentionPioneer", false,
                aclnnAiInfraSparseFlashAttentionPioneerGetWorkspaceSize,
                state.query.tensor, state.key.tensor, state.value.tensor,
                causal_indices->tensor, state.table.tensor, state.qlen.tensor,
                sequence_length->tensor, nullptr, nullptr, state.sink.tensor,
                state.sinkvalue.tensor, 1 / std::sqrt(192.0), 1, layout,
                cache_layout, 3, INT64_MAX, INT64_MAX, 2, false,
                state.out.tensor, nullptr, nullptr, &bytes, &e),
            "MLA attention prepare");
    run(aclnnAiInfraSparseFlashAttentionPioneer, bytes, e);
    return result(state.out);
  }
  V activate(const V &gate, const V &up) {
    int width = gate.size();
    V joined = resident::concat(gate, up);
    DeviceTensor input({1, 2 * width}, ACL_BF16, 2),
        output({1, width}, ACL_BF16, 2);
    to_device(input, joined);
    auto descriptor = [](DeviceTensor &tensor, int64_t columns,
                         uint32_t access) {
      IFKernelTensorV1 result{};
      result.struct_bytes = sizeof(result);
      result.dtype = 2;
      result.rank = 2;
      result.access = access;
      result.data = tensor.data;
      result.allocation_bytes = uint64_t(columns) * 2;
      result.shape[0] = 1;
      result.shape[1] = columns;
      result.strides[0] = columns;
      result.strides[1] = 1;
      return result;
    };
    auto invocation =
        std::make_unique<inferfabric::PreparedKernel>(swiglu.prepare(
            {descriptor(input, 2 * width, 1)}, {descriptor(output, width, 2)}));
    auto *workspace = execution.workspace(invocation->workspace_bytes());
    // Retain foreign invocation metadata until the explicit output boundary.
    pending_kernels.push_back(std::move(invocation));
    pending_kernels.back()->enqueue(workspace, execution.workspace_capacity(),
                                    stream);
    return result(output);
  }
  struct Routes {
    V ids, scores, order, groups;
  };
  struct ExpertRows {
    V packed;
    std::vector<V> rows;
  };
  Routes route(const V &x, const std::string &p,
               const std::string &state_name) {
    auto &ptr = routers[state_name];
    if (!ptr) {
      ptr = std::make_unique<openpangu::Router>(1, 2560, 256, 8, stream);
      auto &gate = cpu(p + ".gate.weight");
      Host transposed(gate.size());
      for (int e = 0; e < 256; ++e)
        for (int j = 0; j < 2560; ++j)
          transposed[j * 256 + e] = gate[e * 2560 + j];
      resident::parameter_upload(ptr->weight, transposed);
      resident::parameter_upload(ptr->bias,
                                 cpu(p + ".e_score_correction_bias"));
    }
    to_device(ptr->input, x, ACL_FLOAT);
    ptr->launch();
    Routes routes{
        result(ptr->ids, ACL_INT32), result(ptr->scores, ACL_FLOAT), {}, {}};
    prepare_routes(routes);
    return routes;
  }
  struct ExpertBank {
    V weights, scales, grouped_scales;
    int n, k;
    ExpertBank(int count, int rows, int columns)
        : weights(size_t(count) * rows * columns, ACL_INT8),
          scales(size_t(count) * rows), n(rows), k(columns) {}
  };
  std::map<std::string, std::unique_ptr<ExpertBank>> expert_banks;
  V route_owner_table;
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
  ExpertBank &bank(const std::string &pattern) {
    auto &ptr = expert_banks[pattern];
    if (ptr)
      return *ptr;
    auto &first = w(expert_name(pattern, rank * 32) + ".weight");
    int n = first.descriptor.shape[0], k = first.descriptor.shape[1];
    if (first.dtype != inferfabric::initialization::DType::I8)
      throw std::runtime_error("resident expert requires int8 contract");
    ptr = std::make_unique<ExpertBank>(32, n, k);
    for (int j = 0; j < 32; ++j) {
      auto name = expert_name(pattern, rank * 32 + j);
      auto &weight = w(name + ".weight"), &scale = w(name + ".weight_scale");
      if (source_types.at(scale.descriptor.id) !=
          inferfabric::initialization::DType::Bf16)
        throw std::runtime_error(
            "grouped expert requires lossless BF16 checkpoint scales");
      if (weight.bytes != size_t(n) * k || scale.bytes != size_t(n) * 4)
        throw std::runtime_error("expert bank shape");
      resident::copy(static_cast<char *>(ptr->weights.data()) +
                         size_t(j) * n * k,
                     weight.data, weight.bytes);
      resident::copy(static_cast<float *>(ptr->scales.data()) + size_t(j) * n,
                     scale.data, scale.bytes);
    }
    ptr->grouped_scales = cast(ptr->scales, ACL_BF16);
    return *ptr;
  }
  // Group-list entries and selected weight offsets remain on device. Zero
  // count terminates the sparse list; unused capacity has no matmul work.
  V sparse_group_table, sparse_group_padding, route_priority;
  void prepare_routes(Routes &routes) {
    if (!sparse_group_table.size()) {
      DeviceTensor table({256, 2}, ACL_INT64, 8),
          padding({24, 2}, ACL_INT64, 8), owners({256}, ACL_FLOAT, 4),
          priority({8}, ACL_FLOAT, 4);
      std::vector<int64_t> entries(512);
      Host owner(256), position(8);
      for (int i = 0; i < 256; ++i) {
        entries[2 * i] = i % 32;
        entries[2 * i + 1] = i / 32 == rank ? 1 : 0;
        owner[i] = float(entries[2 * i + 1]);
      }
      for (int i = 0; i < 8; ++i)
        position[i] = float(8 - i);
      resident::control_upload(table, entries);
      resident::control_upload(padding, std::vector<int64_t>(48, 0));
      resident::control_upload(owners, owner);
      resident::control_upload(priority, position);
      sparse_group_table = result(table, ACL_INT64);
      sparse_group_padding = result(padding, ACL_INT64);
      route_owner_table = result(owners, ACL_FLOAT);
      route_priority = result(priority, ACL_FLOAT);
    }
    // Local entries come first, in original route order. Foreign entries have
    // zero counts; sparse GMM stops there, including the zero-local-expert
    // case.
    auto owners = select(route_owner_table, routes.ids, 256, 1);
    auto priority = binary(owners, route_priority, '*');
    V sorted(8), order(8, ACL_INT64);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("route_owner_order", false,
                            aclnnTopkGetWorkspaceSize, priority.tensor(), 8, -1,
                            true, true, sorted.tensor(), order.tensor(), &bytes,
                            &e),
            "route order prepare");
    run(aclnnTopk, bytes, e);
    routes.order = order;
    auto ids = select(routes.ids, order, 8, 1);
    routes.groups = resident::concat(select(sparse_group_table, ids, 256, 2),
                                     sparse_group_padding);
  }
  ExpertRows grouped_linear(const V &batch, const Routes &routes,
                            ExpertBank &bank, bool fused = false) {
    const int rows = 8;
    if (batch.dtype() != ACL_BF16 ||
        batch.size() != size_t(rows * bank.k * (fused ? 2 : 1)))
      throw std::runtime_error("grouped expert input contract");
    const auto &groups = routes.groups;
    View input(batch.data(), {rows, bank.k * (fused ? 2 : 1)}, ACL_BF16);
    View group_view(groups.data(), {32, 2}, ACL_INT64);
    DeviceTensor quant({rows, bank.k}, ACL_INT8, 1),
        scale({rows}, ACL_FLOAT, 4), out({rows, bank.n}, ACL_BF16, 2);
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
      checked(prepared::query("aclnnDynamicQuant", false,
                              aclnnDynamicQuantGetWorkspaceSize, input.t,
                              nullptr, quant.tensor, scale.tensor, &bytes, &e),
              "grouped expert quant prepare");
      run(aclnnDynamicQuant, bytes, e);
    }
    // Unselected rows are defined zero, including ranks with no local route.
    execution.zero(out.data, out.bytes);
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
    } xs(quant.data, {rows, bank.k}, ACL_INT8),
        ws(bank.weights.data(), {32, bank.n, bank.k}, ACL_INT8, true),
        ss(bank.grouped_scales.data(), {32, bank.n}, ACL_BF16),
        ts(scale.data, {rows}, ACL_FLOAT),
        ys(out.data, {rows, bank.n}, ACL_BF16);
    checked(prepared::query("aclnnGroupedMatmulV5", false,
                            aclnnGroupedMatmulV5GetWorkspaceSize, xs.value,
                            ws.value, nullptr, ss.value, nullptr, nullptr,
                            nullptr, ts.value, group_view.t, nullptr, nullptr,
                            nullptr, 3, 0, 2, 0, nullptr, ys.value, nullptr,
                            nullptr, &bytes, &e),
            "grouped expert matmul prepare");
    run(aclnnGroupedMatmulV5, bytes, e);
    ExpertRows output;
    auto value = result(out);
    output.packed = value;
    for (int i = 0; i < rows; ++i)
      output.rows.push_back(value.view(size_t(i) * bank.n, bank.n));
    return output;
  }
  V weighted_rows(const ExpertRows &rows, const V &scores, size_t width) {
    if (scores.dtype() != ACL_FLOAT ||
        rows.packed.size() != scores.size() * width)
      throw std::runtime_error("weighted expert reduction contract");
    auto input = cast(rows.packed, ACL_FLOAT);
    V products(input.size()), out(width);
    View x(input.data(), {int64_t(scores.size()), int64_t(width)}, ACL_FLOAT),
        s(scores.data(), {int64_t(scores.size()), 1}, ACL_FLOAT),
        y(products.data(), {int64_t(scores.size()), int64_t(width)}, ACL_FLOAT);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(prepared::query("aclnnMul", true, aclnnMulGetWorkspaceSize, x.t,
                            s.t, y.t, &bytes, &e),
            "expert weighted rows prepare");
    run(aclnnMul, bytes, e);
    int64_t axis = 0;
    auto *axes = aclCreateIntArray(&axis, 1);
    if (!axes)
      throw std::runtime_error("expert reduction axes");
    auto status =
        prepared::query("aclnnReduceSum", false, aclnnReduceSumGetWorkspaceSize,
                        y.t, axes, false, ACL_FLOAT, out.tensor(), &bytes, &e);
    aclDestroyIntArray(axes);
    checked(status, "expert reduction prepare");
    run(aclnnReduceSum, bytes, e);
    return out;
  }
  int emit(const V &resident_logits, int step) {
    auto logits = cast(resident_logits, ACL_FLOAT);
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
    if (rank == 0) {
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
  void set_token_inputs(int id, int pos) {
    if (pos < 0 || pos >= int(program.context) || id < 0 || id >= 151552)
      throw std::runtime_error("token/context bound");
    current_position = pos;
    resident::control_upload(*token_index, std::vector<int32_t>{id});
    resident::control_upload(*position_index, std::vector<int32_t>{pos});
    resident::control_upload(*sequence_length, std::vector<int32_t>{pos + 1});
    std::vector<int32_t> indices(program.context, -1);
    for (int i = 0; i <= pos; ++i)
      indices[i] = i;
    resident::control_upload(*causal_indices, indices);
    Host c(64), sn(64);
    for (int i = 0; i < 32; ++i) {
      float angle = float(pos) / std::pow(6400000.f, float(2 * i) / 64);
      c[i] = c[i + 32] = roundbf(std::cos(angle));
      sn[i] = sn[i + 32] = roundbf(std::sin(angle));
    }
    resident::control_upload(*rotary_cos, c);
    resident::control_upload(*rotary_sin, sn);
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
  // Runtime dispatch is defined separately from kernel implementations.
#include "token_dispatch.inc"
};
} // namespace pangu
int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "usage: openpangu_decode LOAD_PLAN_DIR CHECKPOINT NEW_TOKENS "
                 "TOKEN_ID...\n";
    return 2;
  }
  try {
    if (std::getenv("IF_PANGU_TRACE_DIR"))
      throw std::runtime_error(
          "activation host tracing is forbidden by device residency");
    int steps = std::stoi(argv[3]);
    std::vector<int> prompt;
    for (int i = 4; i < argc; ++i)
      prompt.push_back(std::stoi(argv[i]));
    if (steps < 1 || steps > 128 || prompt.size() + steps > 512)
      throw std::runtime_error("token budget");
    auto execution = inferfabric::execution::Plan::read(std::string(argv[1]) +
                                                        "/execution.bin");
    inferfabric::execution::validate_kernels(execution);
    if (prompt.size() + steps > execution.context)
      throw std::runtime_error("execution plan context budget");
    std::vector<inferfabric::initialization::Plan> initialization;
    std::set<uint32_t> devices;
    for (int rank = 0; rank < 8; ++rank) {
      auto plan = inferfabric::initialization::Plan::read(
          std::string(argv[1]) + "/rank-" + std::to_string(rank) + ".bin");
      if (plan.rank != uint32_t(rank) || !devices.insert(plan.device).second)
        throw std::runtime_error("initialization rank/device mismatch");
      inferfabric::execution::validate_bindings(execution, plan);
      initialization.push_back(std::move(plan));
    }
    openpangu::checked(aclInit(nullptr), "init");
    openpangu::checked(aclrtSetDevice(initialization[0].device), "bootstrap");
    HcclRootInfo info;
    openpangu::checked(HcclGetRootInfo(&info), "root");
    inferfabric::profiling::Capture capture(
        std::vector<uint32_t>(devices.begin(), devices.end()));
    std::vector<std::thread> threads;
    for (int rank = 0; rank < 8; ++rank)
      threads.emplace_back([&, rank] {
        try {
          const auto &plan = initialization[rank];
          openpangu::checked(aclrtSetDevice(plan.device), "device");
          {
            pangu::Decoder decoder(plan, argv[2], info, execution);
            pangu::V logits;
            int pos = 0;
            for (int token : prompt) {
              const bool prepare_decode =
                  pos > 0 && pos + 1 == int(prompt.size());
              logits = decoder.token(token, pos++, prepare_decode);
            }
            for (int i = 0; i < steps; ++i) {
              int next = decoder.emit(logits, i);
              if (i > 0 && i <= 2 &&
                  (i == 2 || next == int(execution.eos) || i + 1 == steps) &&
                  !capture.directory.empty())
                capture.rendezvous(false);
              if (next == int(execution.eos) || i + 1 == steps)
                break;
              if (i == 0 && !capture.directory.empty())
                capture.rendezvous(true);
              inferfabric::profiling::state.active =
                  !capture.directory.empty() && i < 2;
              {
                inferfabric::profiling::Scope timing("decode_total");
                logits = decoder.token(next, pos++, true);
              }
              inferfabric::profiling::state.active = false;
            }
            if (!capture.directory.empty())
              inferfabric::profiling::write(capture.directory + "/rank-" +
                                            std::to_string(rank) + ".csv");
            if (const char *dir =
                    std::getenv("INFERFABRIC_PREPARED_AUDIT_DIR")) {
              std::filesystem::create_directories(dir);
              std::ofstream out(std::filesystem::path(dir) /
                                ("rank-" + std::to_string(rank) + ".json"));
              decoder.execution.write(out);
              if (!out)
                throw std::runtime_error("prepared audit write");
            }
            if (const char *dir =
                    std::getenv("INFERFABRIC_TRANSFER_AUDIT_DIR")) {
              std::filesystem::create_directories(dir);
              std::ofstream audit(std::filesystem::path(dir) /
                                  ("rank-" + std::to_string(rank) + ".json"));
              inferfabric::resident::transfers.write(audit);
              if (!audit)
                throw std::runtime_error("transfer audit write");
            }
          }
          openpangu::checked(aclrtResetDevice(plan.device), "reset");
        } catch (const std::exception &e) {
          std::cerr << "rank=" << rank << " " << e.what() << std::endl;
          std::_Exit(1);
        }
      });
    for (auto &t : threads)
      t.join();
    capture.close();
    openpangu::checked(aclFinalize(), "finalize");
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

// Device-resident execution of a validated generic token program.
// No activation or routing value has a host representation.
#include "ascend_token_contract.h"
#include "compiled_kernel.h"
#include "initialization_acl.h"
#include "openpangu_decoder_state.h"
#include "openpangu_mhc.h"
#include "resident_tensor.h"
#include "resident_transfer.h"
#include <aclnn_ai_infra_fused_causal_conv1d.h>
#include <aclnn_ai_infra_sparse_flash_attention_pioneer.h>
#include <aclnnop/aclnn_dequant_swiglu_quant.h>
#include <aclnnop/aclnn_dynamic_quant.h>
#include <aclnnop/aclnn_quant_matmul_v5.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <hccl/hccl.h>
#include <iostream>
#include <numeric>
#include <thread>
#include <variant>
namespace pangu {
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
  View(void *p, std::vector<int64_t> shape, aclDataType type,
       bool transpose = false) {
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
    if (t)
      aclDestroyTensor(t);
  }
};
struct Decoder {
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
  void *workspace = nullptr;
  uint64_t capacity = 0;
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
    checked(aclrtCreateStream(&stream), "stream");
    checked(HcclCommInitRootInfo(8, &info, rank, &comm), "HCCL");
  }
  ~Decoder() {
    if (stream)
      aclrtSynchronizeStream(stream);
    if (workspace)
      aclrtFree(workspace);
    if (comm)
      HcclCommDestroy(comm);
    if (stream)
      aclrtDestroyStream(stream);
  }
  void reserve_workspace(uint64_t bytes) {
    if (bytes > capacity) {
      if (workspace)
        checked(aclrtFree(workspace), "workspace free");
      capacity = bytes;
      checked(aclrtMalloc(&workspace, capacity, ACL_MEM_MALLOC_HUGE_FIRST),
              "workspace");
    }
  }
  void run(openpangu::Launcher f, uint64_t bytes, aclOpExecutor *e) {
    reserve_workspace(bytes);
    checked(f(workspace, bytes, e, stream), "operator");
    checked(aclrtSynchronizeStream(stream), "fence");
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
    checked(
        aclnnCastGetWorkspaceSize(x.tensor(), dtype, out.tensor(), &bytes, &e),
        "resident cast prepare");
    run(aclnnCast, bytes, e);
    return out;
  }
  V rounded(const V &x) { return cast(cast(x, ACL_BF16), ACL_FLOAT); }
  V result(const DeviceTensor &x, aclDataType dtype = ACL_BF16) {
    return cast(resident::from_device(
                    x.data, x.bytes / resident::element_bytes(dtype), dtype),
                ACL_FLOAT);
  }
  void to_device(DeviceTensor &dst, const V &src,
                 aclDataType dtype = ACL_BF16) {
    auto value = cast(src, dtype);
    if (value.bytes() != dst.bytes)
      throw std::runtime_error("resident copy shape");
    resident::copy(dst.data, value.data(), dst.bytes);
  }
  V binary(const V &a, const V &b, char kind) {
    if (a.dtype() != ACL_FLOAT || b.dtype() != ACL_FLOAT ||
        (a.size() != b.size() && b.size() != 1))
      throw std::runtime_error("resident binary contract");
    V out(a.size());
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    float f = 1;
    aclScalar *one = aclCreateScalar(&f, ACL_FLOAT);
    aclnnStatus status =
        kind == '*'   ? aclnnMulGetWorkspaceSize(a.tensor(), b.tensor(),
                                                 out.tensor(), &bytes, &e)
        : kind == '+' ? aclnnAddGetWorkspaceSize(a.tensor(), b.tensor(), one,
                                                 out.tensor(), &bytes, &e)
                      : aclnnSubGetWorkspaceSize(a.tensor(), b.tensor(), one,
                                                 out.tensor(), &bytes, &e);
    aclDestroyScalar(one);
    checked(status, "resident binary prepare");
    run(kind == '*' ? aclnnMul : kind == '+' ? aclnnAdd : aclnnSub, bytes, e);
    return out;
  }
  V gather(const V &x) {
    DeviceTensor a({int64_t(x.size())}, ACL_BF16, 2),
        b({int64_t(x.size() * 8)}, ACL_BF16, 2);
    to_device(a, x);
    checked(HcclAllGather(a.data, b.data, x.size(), HCCL_DATA_TYPE_BFP16, comm,
                          stream),
            "all gather");
    checked(aclrtSynchronizeStream(stream), "gather fence");
    return result(b);
  }
  V sum(const V &x) {
    DeviceTensor a({int64_t(x.size())}, ACL_FLOAT, 4),
        b({int64_t(x.size())}, ACL_FLOAT, 4);
    to_device(a, x, ACL_FLOAT);
    checked(HcclAllReduce(a.data, b.data, x.size(), HCCL_DATA_TYPE_FP32,
                          HCCL_REDUCE_SUM, comm, stream),
            "all reduce");
    checked(aclrtSynchronizeStream(stream), "reduce fence");
    return result(b, ACL_FLOAT);
  }
  V matrix(const V &x, void *weight, int rows, int cols,
           bool transposed = true) {
    int output = transposed ? rows : cols;
    DeviceTensor input({1, int64_t(x.size())}, ACL_BF16, 2),
        out({1, output}, ACL_BF16, 2);
    to_device(input, x);
    View view(weight, {rows, cols}, ACL_BF16, transposed);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(aclnnMatmulGetWorkspaceSize(input.tensor, view.t, out.tensor, 0,
                                        &bytes, &e),
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
      DeviceTensor input({1, k}, ACL_BF16, 2), quant({1, k}, ACL_INT8, 1),
          scale({1}, ACL_FLOAT, 4), out({1, n}, ACL_BF16, 2);
      to_device(input, x);
      View weight(a.data, {n, k}, ACL_INT8), scales(s.data, {n}, ACL_FLOAT);
      uint64_t bytes = 0;
      aclOpExecutor *e = nullptr;
      checked(aclnnDynamicQuantGetWorkspaceSize(input.tensor, nullptr,
                                                quant.tensor, scale.tensor,
                                                &bytes, &e),
              "quant prepare");
      run(aclnnDynamicQuant, bytes, e);
      checked(aclnnQuantMatmulV5GetWorkspaceSize(
                  quant.tensor, weight.t, scale.tensor, scales.t, nullptr,
                  nullptr, nullptr, nullptr, nullptr, false, true, 0,
                  out.tensor, &bytes, &e),
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
    DeviceTensor input({1, int64_t(x.size())}, ACL_BF16, 2),
        out({1, int64_t(x.size())}, ACL_BF16, 2), rstd({1, 1}, ACL_FLOAT, 4);
    to_device(input, x);
    View weight(gamma.data, {int64_t(x.size())}, ACL_BF16);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(aclnnRmsNormGetWorkspaceSize(input.tensor, weight.t, 1e-5,
                                         out.tensor, rstd.tensor, &bytes, &e),
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
    resident::control_upload(s.computed, std::vector<int32_t>{position});
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(aclnnAiInfraFusedCausalConv1dGetWorkspaceSize(
                s.input.tensor, s.weight.tensor, s.state.tensor,
                s.starts.tensor, s.slots.tensor, nullptr, nullptr, nullptr,
                s.computed.tensor, nullptr, nullptr, nullptr, 0, -1, 0, 1, 1,
                128, 1, true, s.out.tensor, &bytes, &e),
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
    checked(api.pre_size(s.x.tensor, s.phi.tensor, s.alpha.tensor,
                         s.beta.tensor, nullptr, 1, 1e-6, 1e-6, s.input.tensor,
                         s.post.tensor, s.res.tensor, s.rms.tensor,
                         s.mix.tensor, s.pre_weights.tensor, &bytes, &e),
            "mHC pre prepare");
    run(api.pre, bytes, e);
    if (!merge) {
      checked(api.sink_size(s.res.tensor, 0, 1e-6, 20, s.sink.tensor, nullptr,
                            nullptr, &bytes, &e),
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
    checked(api.post_size(s.x.tensor, s.sink.tensor, s.branch.tensor,
                          s.post.tensor, s.out.tensor, &bytes, &e),
            "mHC post prepare");
    run(api.post, bytes, e);
    return result(s.out);
  }
  V rope(const V &v, int pos) {
    if (v.size() != 64)
      throw std::runtime_error("rotary width");
    Host c(32), sn(32);
    for (int i = 0; i < 32; ++i) {
      float angle = float(pos) / std::pow(6400000.f, float(2 * i) / 64);
      c[i] = roundbf(std::cos(angle));
      sn[i] = roundbf(std::sin(angle));
    }
    // Position-dependent constants are control inputs, never activation
    // readback.
    DeviceTensor dc({32}, ACL_FLOAT, 4), ds({32}, ACL_FLOAT, 4);
    resident::control_upload(dc, c);
    resident::control_upload(ds, sn);
    auto cv = result(dc, ACL_FLOAT), sv = result(ds, ACL_FLOAT);
    auto a = resident::slice(v, 0, 32), b = resident::slice(v, 32, 32);
    return rounded(
        resident::concat(binary(binary(a, cv, '*'), binary(b, sv, '*'), '-'),
                         binary(binary(b, cv, '*'), binary(a, sv, '*'), '+')));
  }
  V absorb_query(const V &q, const std::string &weight_name, int pos) {
    auto &weight = w(weight_name + ".weight");
    V queries;
    for (int head = 0; head < 6; ++head) {
      size_t offset = (rank * 6 + head) * 192;
      auto no = resident::slice(q, offset, 128),
           rot = rope(resident::slice(q, offset + 128, 64), pos);
      auto query =
          matrix(no, static_cast<uint16_t *>(weight.data) + head * 256 * 512,
                 128, 512, false);
      query = resident::concat(query, rot);
      queries = head ? resident::concat(queries, query) : query;
    }
    return queries;
  }
  V expand_value(const V &values, const std::string &weight_name) {
    auto &weight = w(weight_name + ".weight");
    V local;
    for (int head = 0; head < 6; ++head) {
      auto value = resident::slice(values, head * 512, 512);
      auto out = matrix(value,
                        static_cast<uint16_t *>(weight.data) +
                            (head * 256 + 128) * 512,
                        128, 512);
      local = head ? resident::concat(local, out) : out;
    }
    return local;
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
      auto row =
          cast(resident::from_device(static_cast<uint16_t *>(sk.data) + i * 512,
                                     512, ACL_BF16),
               ACL_FLOAT);
      row = norm(row, norm_name);
      auto rot =
          cast(resident::from_device(static_cast<uint16_t *>(sr.data) + i * 64,
                                     64, ACL_BF16),
               ACL_FLOAT);
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
    resident::copy(static_cast<uint16_t *>(state.key.data) + size_t(pos) * 576,
                   key.data(), 1152);
    std::vector<int32_t> indices(512, -1);
    for (int i = 0; i <= pos; ++i)
      indices[i] = i;
    resident::control_upload(state.indices, indices);
    resident::control_upload(state.klen, std::vector<int32_t>{pos + 1});
  }
  V attend(const V &queries, const std::string &state_name) {
    auto &state = *attention_states.at(state_name);
    to_device(state.query, queries);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    char layout[] = "TND", cache_layout[] = "PA_BSND";
    checked(aclnnAiInfraSparseFlashAttentionPioneerGetWorkspaceSize(
                state.query.tensor, state.key.tensor, state.value.tensor,
                state.indices.tensor, state.table.tensor, state.qlen.tensor,
                state.klen.tensor, nullptr, nullptr, state.sink.tensor,
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
    auto prepared = swiglu.prepare({descriptor(input, 2 * width, 1)},
                                   {descriptor(output, width, 2)});
    reserve_workspace(prepared.workspace_bytes());
    try {
      prepared.enqueue(workspace, capacity, stream);
      checked(aclrtSynchronizeStream(stream), "compiled SwiGLU fence");
    } catch (...) {
      // Foreign metadata and device buffers cannot be freed before completion.
      aclrtSynchronizeStream(stream);
      throw;
    }
    return result(output);
  }
  struct Routes {
    V ids, scores;
  };
  struct ExpertRows {
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
    checked(aclrtSynchronizeStream(stream), "router fence");
    return {resident::from_device(ptr->ids.data, 8, ACL_INT32),
            result(ptr->scores, ACL_FLOAT)};
  }
  struct ExpertBank {
    V weights, scales;
    int n, k;
    ExpertBank(int count, int rows, int columns)
        : weights(size_t(count) * rows * columns, ACL_INT8),
          scales(size_t(count) * rows), n(rows), k(columns) {}
  };
  std::map<std::string, std::unique_ptr<ExpertBank>> expert_banks;
  V route_index_table, route_owner_table;
  V select(const V &source, const V &indices, size_t rows, size_t width) {
    if (source.size() != rows * width)
      throw std::runtime_error("indexed device selection shape");
    V out(indices.size() * width, source.dtype());
    View sv(source.data(), {int64_t(rows), int64_t(width)}, source.dtype());
    View ov(out.data(), {int64_t(indices.size()), int64_t(width)},
            source.dtype());
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(aclnnIndexSelectGetWorkspaceSize(sv.t, 0, indices.tensor(), ov.t,
                                             &bytes, &e),
            "device index prepare");
    run(aclnnIndexSelect, bytes, e);
    return out;
  }
  V local_route_ids(const Routes &routes) {
    if (!route_index_table.size()) {
      DeviceTensor ids({256}, ACL_INT32, 4), owners({256}, ACL_FLOAT, 4);
      std::vector<int32_t> iv(256);
      Host ov(256);
      for (int j = 0; j < 256; ++j) {
        iv[j] = j % 32;
        ov[j] = j / 32 == rank ? 1.f : 0.f;
      }
      resident::control_upload(ids, iv);
      resident::control_upload(owners, ov);
      route_index_table = resident::from_device(ids.data, 256, ACL_INT32);
      route_owner_table = result(owners, ACL_FLOAT);
    }
    return select(route_index_table, routes.ids, 256, 1);
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
      if (weight.bytes != size_t(n) * k || scale.bytes != size_t(n) * 4)
        throw std::runtime_error("expert bank shape");
      resident::copy(static_cast<char *>(ptr->weights.data()) +
                         size_t(j) * n * k,
                     weight.data, weight.bytes);
      resident::copy(static_cast<float *>(ptr->scales.data()) + size_t(j) * n,
                     scale.data, scale.bytes);
    }
    return *ptr;
  }
  V selected_linear(const V &x, const V &index, ExpertBank &bank,
                    bool fused = false) {
    auto weight = select(bank.weights, index, 32, size_t(bank.n) * bank.k);
    auto scales = select(bank.scales, index, 32, bank.n);
    DeviceTensor input({1, int64_t(x.size())}, ACL_BF16, 2),
        quant({1, bank.k}, ACL_INT8, 1), scale({1}, ACL_FLOAT, 4),
        out({1, bank.n}, ACL_BF16, 2);
    to_device(input, x);
    View wv(weight.data(), {bank.n, bank.k}, ACL_INT8),
        sv(scales.data(), {bank.n}, ACL_FLOAT);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    if (fused) {
      char mode[] = "dynamic";
      checked(aclnnDequantSwigluQuantGetWorkspaceSize(
                  input.tensor, nullptr, nullptr, nullptr, nullptr, nullptr,
                  nullptr, true, mode, quant.tensor, scale.tensor, &bytes, &e),
              "resident expert activation prepare");
      run(aclnnDequantSwigluQuant, bytes, e);
    } else {
      checked(aclnnDynamicQuantGetWorkspaceSize(input.tensor, nullptr,
                                                quant.tensor, scale.tensor,
                                                &bytes, &e),
              "resident expert quant prepare");
      run(aclnnDynamicQuant, bytes, e);
    }
    checked(aclnnQuantMatmulV5GetWorkspaceSize(quant.tensor, wv.t, scale.tensor,
                                               sv.t, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, false, true, 0,
                                               out.tensor, &bytes, &e),
            "resident expert matmul prepare");
    run(aclnnQuantMatmulV5, bytes, e);
    return result(out);
  }
  int emit(const V &logits, int step) {
    V top(5), ids(5, ACL_INT64), probs(logits.size());
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(aclnnTopkGetWorkspaceSize(logits.tensor(), 5, 0, true, true,
                                      top.tensor(), ids.tensor(), &bytes, &e),
            "device output topk prepare");
    run(aclnnTopk, bytes, e);
    checked(aclnnLogSoftmaxGetWorkspaceSize(logits.tensor(), 0, probs.tensor(),
                                            &bytes, &e),
            "device logsoftmax prepare");
    run(aclnnLogSoftmax, bytes, e);
    auto selected_probs = select(probs, ids, logits.size(), 1);
    V selected(1, ACL_INT64), peers(8, ACL_INT64);
    checked(aclnnArgMaxGetWorkspaceSize(logits.tensor(), 0, true,
                                        selected.tensor(), &bytes, &e),
            "device greedy argmax prepare");
    run(aclnnArgMax, bytes, e);
    checked(HcclAllGather(selected.data(), peers.data(), 1,
                          HCCL_DATA_TYPE_INT64, comm, stream),
            "device token consensus");
    checked(aclrtSynchronizeStream(stream), "output completion");
    resident::TransferScope boundary(resident::TransferRole::RequestOutput);
    // Explicit request-output boundary: only selected IDs/top-five diagnostics.
    std::array<int64_t, 8> peer_ids{};
    std::array<int64_t, 5> order{};
    std::array<float, 5> lp{}, scores{};
    checked(aclrtMemcpy(peer_ids.data(), 64, peers.data(), 64,
                        ACL_MEMCPY_DEVICE_TO_HOST),
            "output consensus IDs");
    int next = int(peer_ids[0]);
    for (auto id : peer_ids)
      if (id != next)
        throw std::runtime_error("ranks disagree on token");
    if (rank == 0) {
      checked(aclrtMemcpy(order.data(), 40, ids.data(), 40,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "output top IDs");
      checked(aclrtMemcpy(lp.data(), 20, selected_probs.data(), 20,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "output top log probabilities");
      checked(aclrtMemcpy(scores.data(), 20, top.data(), 20,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "output top scores");
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
            for (int token : prompt)
              logits = decoder.token(token, pos++);
            for (int i = 0; i < steps; ++i) {
              int next = decoder.emit(logits, i);
              if (next == int(execution.eos) || i + 1 == steps)
                break;
              logits = decoder.token(next, pos++);
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
    openpangu::checked(aclFinalize(), "finalize");
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

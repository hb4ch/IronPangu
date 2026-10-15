// Full checkpoint diagnostic decoder. All projection GEMMs execute on Ascend;
// normalization, mHC, MoME, routing and attention use NPU kernels. Host
// orchestration, RoPE and expert-output combination remain diagnostic paths.
#include "compiled_kernel.h"
#include "initialization_acl.h"
#include "openpangu_decoder_state.h"
#include "openpangu_mhc.h"
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
namespace pangu {
using V = std::vector<float>;
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
std::vector<uint16_t> packed(const V &x) {
  std::vector<uint16_t> b;
  for (float v : x)
    b.push_back(tobf(v));
  return b;
}
V unpacked(const std::vector<uint16_t> &b) {
  V x;
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
  std::map<int, std::unique_ptr<openpangu::Router>> routers;
  std::map<std::string, std::unique_ptr<MhcState>> mhc_states;
  std::map<std::string, std::unique_ptr<MomeState>> mome_states;
  std::array<std::unique_ptr<AttentionState>, 46> attention_states;
  aclrtStream stream = nullptr;
  HcclComm comm = nullptr;
  inferfabric::CompiledKernel swiglu;
  inferfabric::initialization::ParameterStore weights;
  std::map<std::string, inferfabric::initialization::ParameterId> bindings;
  std::map<std::string, V> host;
  std::map<std::string, std::array<V, 2>> history;
  std::array<std::vector<V>, 46> cache;
  std::array<std::vector<V>, 46> sinks;
  void *workspace = nullptr;
  uint64_t capacity = 0;
  Decoder(const inferfabric::initialization::Plan &p, const std::string &root,
          HcclRootInfo &info)
      : rank(p.rank), swiglu(std::getenv("INFERFABRIC_SWIGLU_LIBRARY")
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
  const V &cpu(const std::string &n) {
    auto it = host.find(n);
    if (it != host.end())
      return it->second;
    auto &a = w(n);
    V v;
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
  V gather(const V &x) {
    DeviceTensor a({int64_t(x.size())}, ACL_BF16, 2),
        b({int64_t(x.size() * 8)}, ACL_BF16, 2);
    a.upload(packed(x));
    checked(HcclAllGather(a.data, b.data, x.size(), HCCL_DATA_TYPE_BFP16, comm,
                          stream),
            "all gather");
    checked(aclrtSynchronizeStream(stream), "gather fence");
    return unpacked(b.read<uint16_t>());
  }
  V sum(const V &x) {
    DeviceTensor a({int64_t(x.size())}, ACL_FLOAT, 4),
        b({int64_t(x.size())}, ACL_FLOAT, 4);
    a.upload(x);
    checked(HcclAllReduce(a.data, b.data, x.size(), HCCL_DATA_TYPE_FP32,
                          HCCL_REDUCE_SUM, comm, stream),
            "all reduce");
    checked(aclrtSynchronizeStream(stream), "reduce fence");
    return b.read<float>();
  }
  V matrix(const V &x, void *weight, int rows, int cols,
           bool transposed = true) {
    int output = transposed ? rows : cols;
    DeviceTensor input({1, int64_t(x.size())}, ACL_BF16, 2),
        out({1, output}, ACL_BF16, 2);
    input.upload(packed(x));
    View view(weight, {rows, cols}, ACL_BF16, transposed);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(aclnnMatmulGetWorkspaceSize(input.tensor, view.t, out.tensor, 0,
                                        &bytes, &e),
            "BF16 matmul prepare");
    run(aclnnMatmul, bytes, e);
    return unpacked(out.read<uint16_t>());
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
      input.upload(packed(x));
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
      result = unpacked(out.read<uint16_t>());
    }
    auto output = distributed ? gather(result) : result;
    trace(name, output);
    return output;
  }
  void trace(const std::string &name, const V &x) {
    const char *dir = std::getenv("IF_PANGU_TRACE_DIR");
    if (!dir || rank != 0 || (current_position != 0 && current_position != 11))
      return;
    std::filesystem::create_directories(dir);
    std::ofstream file(
        std::filesystem::path(dir) /
            ("pos" + std::to_string(current_position) + "-" + name + ".f32"),
        std::ios::binary);
    file.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
    if (!file)
      throw std::runtime_error("trace write");
  }
  V norm(const V &x, const std::string &name) {
    auto &gamma = w(name + ".weight");
    if (gamma.bytes != x.size() * 2)
      throw std::runtime_error("norm shape");
    DeviceTensor input({1, int64_t(x.size())}, ACL_BF16, 2),
        out({1, int64_t(x.size())}, ACL_BF16, 2), rstd({1, 1}, ACL_FLOAT, 4);
    input.upload(packed(x));
    View weight(gamma.data, {int64_t(x.size())}, ACL_BF16);
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(aclnnRmsNormGetWorkspaceSize(input.tensor, weight.t, 1e-5,
                                         out.tensor, rstd.tensor, &bytes, &e),
            "RMSNorm prepare");
    run(aclnnRmsNorm, bytes, e);
    auto y = unpacked(out.read<uint16_t>());
    trace(name, y);
    return y;
  }
  V mome(V x, const std::string &name, int position) {
    auto &ptr = mome_states[name];
    if (!ptr) {
      ptr = std::make_unique<MomeState>(x.size());
      auto &source = cpu(name + ".weight");
      std::vector<uint16_t> weight(source.size());
      for (size_t j = 0; j < x.size(); ++j)
        for (int tap = 0; tap < 3; ++tap)
          weight[tap * x.size() + j] = tobf(source[j * 3 + tap]);
      ptr->weight.upload(weight);
    }
    auto &s = *ptr;
    s.input.upload(packed(x));
    s.computed.upload(std::vector<int32_t>{position});
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(aclnnAiInfraFusedCausalConv1dGetWorkspaceSize(
                s.input.tensor, s.weight.tensor, s.state.tensor,
                s.starts.tensor, s.slots.tensor, nullptr, nullptr, nullptr,
                s.computed.tensor, nullptr, nullptr, nullptr, 0, -1, 0, 1, 1,
                128, 1, true, s.out.tensor, &bytes, &e),
            "MoME prepare");
    run(aclnnAiInfraFusedCausalConv1d, bytes, e);
    return unpacked(s.input.read<uint16_t>());
  }
  struct HC {
    MhcState *device = nullptr;
    V input, post, res;
  };
  HC pre(const V &x, const std::string &name, bool merge = false) {
    auto &ptr = mhc_states[name];
    if (!ptr) {
      ptr = std::make_unique<MhcState>();
      auto &phi = cpu(name + ".phi.weight");
      auto &gamma = cpu(name + ".norm_gamma");
      V folded(24 * 10240, 0), alpha(3, 0), beta(24, 0);
      for (size_t j = 0; j < phi.size(); ++j)
        folded[j] = phi[j] * gamma[j % 10240];
      auto &a = cpu(name + (merge ? ".branch_alpha_pre" : ".branch_alpha"));
      auto &b = cpu(name + (merge ? ".branch_beta_pre" : ".branch_beta"));
      std::copy(a.begin(), a.end(), alpha.begin());
      std::copy(b.begin(), b.end(), beta.begin());
      ptr->phi.upload(folded);
      ptr->alpha.upload(alpha);
      ptr->beta.upload(beta);
    }
    auto &s = *ptr;
    s.x.upload(packed(x));
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
    h.input = unpacked(s.input.read<uint16_t>());
    return h;
  }
  V post(const V &, const V &branch, const HC &h) {
    if (!h.device)
      throw std::runtime_error("missing mHC pre state");
    auto &s = *h.device;
    s.branch.upload(packed(branch));
    auto &api = training_api();
    uint64_t bytes = 0;
    aclOpExecutor *e = nullptr;
    checked(api.post_size(s.x.tensor, s.sink.tensor, s.branch.tensor,
                          s.post.tensor, s.out.tensor, &bytes, &e),
            "mHC post prepare");
    run(api.post, bytes, e);
    return unpacked(s.out.read<uint16_t>());
  }
  void rope(V &v, int pos) {
    V copy = v;
    for (int i = 0; i < 32; ++i) {
      float angle = float(pos) / std::pow(6400000.f, float(2 * i) / 64);
      float c = roundbf(std::cos(angle)), s = roundbf(std::sin(angle));
      v[i] = roundbf(copy[i] * c - copy[i + 32] * s);
      v[i + 32] = roundbf(copy[i + 32] * c + copy[i] * s);
    }
  }
  V attention(const V &x, int layer, int pos) {
    std::string p = "model.layers." + std::to_string(layer) + ".self_attn";
    auto qa = norm(mome(linear(p + ".q_a_proj", x), p + ".qa_conv", pos),
                   p + ".q_a_layernorm");
    auto q = linear(p + ".q_b_proj", qa, true);
    auto kv = linear(p + ".kv_a_proj_with_mqa", x);
    V latent(kv.begin(), kv.begin() + 512), kr(kv.begin() + 512, kv.end());
    latent =
        norm(mome(latent, p + ".compresskv_conv", pos), p + ".kv_a_layernorm");
    rope(kr, pos);
    latent.insert(latent.end(), kr.begin(), kr.end());

    if (sinks[layer].empty()) {
      auto &sk = cpu(p + ".param_sink_compressed_kv");
      auto &sr = cpu(p + ".param_sink_k_pe");
      for (int i = 0; i < 128; ++i) {
        V v(sk.begin() + i * 512, sk.begin() + (i + 1) * 512);
        v = norm(v, p + ".kv_a_layernorm");
        v.insert(v.end(), sr.begin() + i * 64, sr.begin() + (i + 1) * 64);
        sinks[layer].push_back(std::move(v));
      }
    }
    auto &weight = w(p + ".kv_b_proj.weight");
    auto &ptr = attention_states[layer];
    if (!ptr) {
      ptr = std::make_unique<AttentionState>();
      V sink_values;
      for (auto &row : sinks[layer])
        sink_values.insert(sink_values.end(), row.begin(), row.end());
      ptr->sink.upload(packed(sink_values));
    }
    auto &state = *ptr;
    auto key = packed(latent);
    checked(
        aclrtMemcpy(static_cast<uint16_t *>(state.key.data) + size_t(pos) * 576,
                    1152, key.data(), 1152, ACL_MEMCPY_HOST_TO_DEVICE),
        "cache append");
    std::vector<int32_t> indices(512, -1);
    for (int i = 0; i <= pos; ++i)
      indices[i] = i;
    state.indices.upload(indices);
    state.klen.upload(std::vector<int32_t>{pos + 1});
    V queries;
    for (int head = 0; head < 6; ++head) {
      int offset = (rank * 6 + head) * 192;
      V no(q.begin() + offset, q.begin() + offset + 128),
          rot(q.begin() + offset + 128, q.begin() + offset + 192);
      rope(rot, pos);
      auto query =
          matrix(no, static_cast<uint16_t *>(weight.data) + head * 256 * 512,
                 128, 512, false);
      query.insert(query.end(), rot.begin(), rot.end());
      queries.insert(queries.end(), query.begin(), query.end());
    }
    state.query.upload(packed(queries));
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
    auto values = unpacked(state.out.read<uint16_t>());
    V local;
    for (int head = 0; head < 6; ++head) {
      V value(values.begin() + head * 512, values.begin() + (head + 1) * 512);
      auto out = matrix(value,
                        static_cast<uint16_t *>(weight.data) +
                            (head * 256 + 128) * 512,
                        128, 512);
      local.insert(local.end(), out.begin(), out.end());
    }
    return linear(p + ".o_proj", mome(gather(local), p + ".o_conv", pos), true);
  }
  V mlp(const V &x, const std::string &p, bool distributed) {
    auto gate = linear(p + ".gate_proj", x, distributed),
         up = linear(p + ".up_proj", x, distributed);
    if (!distributed) {
      int k = gate.size();
      auto &down = w(p + ".down_proj.weight");
      int n = down.descriptor.shape[0];
      auto &sc = w(p + ".down_proj.weight_scale");
      V joined = gate;
      joined.insert(joined.end(), up.begin(), up.end());
      DeviceTensor input({1, 2 * k}, ACL_BF16, 2), quant({1, k}, ACL_INT8, 1),
          scale({1}, ACL_FLOAT, 4), out({1, n}, ACL_BF16, 2);
      input.upload(packed(joined));
      View weight(down.data, {n, k}, ACL_INT8), scales(sc.data, {n}, ACL_FLOAT);
      uint64_t bytes = 0;
      aclOpExecutor *e = nullptr;
      char mode[] = "dynamic";
      checked(aclnnDequantSwigluQuantGetWorkspaceSize(
                  input.tensor, nullptr, nullptr, nullptr, nullptr, nullptr,
                  nullptr, true, mode, quant.tensor, scale.tensor, &bytes, &e),
              "expert fused activation prepare");
      run(aclnnDequantSwigluQuant, bytes, e);
      checked(aclnnQuantMatmulV5GetWorkspaceSize(
                  quant.tensor, weight.t, scale.tensor, scales.t, nullptr,
                  nullptr, nullptr, nullptr, nullptr, false, true, 0,
                  out.tensor, &bytes, &e),
              "expert down prepare");
      run(aclnnQuantMatmulV5, bytes, e);
      return unpacked(out.read<uint16_t>());
    }
    int width = gate.size();
    V joined = gate;
    joined.insert(joined.end(), up.begin(), up.end());
    DeviceTensor input({1, 2 * width}, ACL_BF16, 2),
        output({1, width}, ACL_BF16, 2);
    input.upload(packed(joined));
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
    gate = unpacked(output.read<uint16_t>());
    return linear(p + ".down_proj", gate, distributed);
  }
  V feedforward(const V &x, int layer) {
    std::string p = "model.layers." + std::to_string(layer) + ".mlp";
    if (layer < 2)
      return mlp(x, p, true);
    auto &ptr = routers[layer];
    if (!ptr) {
      ptr = std::make_unique<openpangu::Router>(1, 2560, 256, 8, stream);
      auto &gate = cpu(p + ".gate.weight");
      V transposed(gate.size());
      for (int e = 0; e < 256; ++e)
        for (int j = 0; j < 2560; ++j)
          transposed[j * 256 + e] = gate[e * 2560 + j];
      ptr->weight.upload(transposed);
      ptr->bias.upload(cpu(p + ".e_score_correction_bias"));
    }
    ptr->input.upload(x);
    ptr->launch();
    checked(aclrtSynchronizeStream(stream), "router fence");
    auto order = ptr->ids.read<int32_t>();
    auto score = ptr->scores.read<float>();
    V y(2560, 0);
    for (int i = 0; i < 8; ++i) {
      int e = order[i];
      if (e / 32 != rank)
        continue;
      auto value = mlp(x, p + ".experts." + std::to_string(e), false);
      float scale = score[i];
      for (int j = 0; j < 2560; ++j)
        y[j] += scale * value[j];
    }
    y = sum(y);
    auto shared = mlp(x, p + ".shared_experts", true);
    for (int j = 0; j < 2560; ++j)
      y[j] = roundbf(y[j] + shared[j]);
    return y;
  }
  V token(int id, int pos) {
    current_position = pos;
    if (pos < 0 || pos >= 512 || id < 0 || id >= 151552)
      throw std::runtime_error("diagnostic decoder token/context bound");
    auto &embedding = w("model.embed_tokens.weight");
    std::vector<uint16_t> b(2560);
    checked(
        aclrtMemcpy(b.data(), 5120,
                    static_cast<uint16_t *>(embedding.data) + size_t(id) * 2560,
                    5120, ACL_MEMCPY_DEVICE_TO_HOST),
        "embedding row");
    V x;
    auto row = unpacked(b);
    for (int j = 0; j < 4; ++j)
      x.insert(x.end(), row.begin(), row.end());
    for (int layer = 0; layer < 46; ++layer) {
      std::string p = "model.layers." + std::to_string(layer);
      auto h = pre(x, p + ".attn_mhc_module");
      auto a = attention(norm(h.input, p + ".input_layernorm"), layer, pos);
      x = post(x, norm(a, p + ".post_attention_layernorm"), h);
      h = pre(x, p + ".mlp_mhc_module");
      auto f = feedforward(norm(h.input, p + ".pre_mlp_layernorm"), layer);
      x = post(x, norm(f, p + ".post_mlp_layernorm"), h);
      if (bindings.count(p + ".block_post_layernorm.weight"))
        x = norm(x, p + ".block_post_layernorm");
      trace(p + ".output", x);
      for (float v : x)
        if (!std::isfinite(v))
          throw std::runtime_error("nonfinite layer " + std::to_string(layer));
      if (rank == 0)
        std::cerr << "position=" << pos << " layer=" << layer << " rms="
                  << std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(),
                                                  0.f) /
                               x.size())
                  << '\n';
    }
    return linear(
        "lm_head",
        norm(pre(x, "model.merge_mhc_module", true).input, "model.norm"), true);
  }
};
} // namespace pangu
int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "usage: openpangu_decode LOAD_PLAN_DIR CHECKPOINT NEW_TOKENS "
                 "TOKEN_ID...\n";
    return 2;
  }
  try {
    int steps = std::stoi(argv[3]);
    std::vector<int> prompt;
    for (int i = 4; i < argc; ++i)
      prompt.push_back(std::stoi(argv[i]));
    if (steps < 1 || steps > 128 || prompt.size() + steps > 512)
      throw std::runtime_error("token budget");
    openpangu::checked(aclInit(nullptr), "init");
    openpangu::checked(aclrtSetDevice(0), "bootstrap");
    HcclRootInfo info;
    openpangu::checked(HcclGetRootInfo(&info), "root");
    std::vector<std::thread> threads;
    for (int rank = 0; rank < 8; ++rank)
      threads.emplace_back([&, rank] {
        try {
          auto plan = inferfabric::initialization::Plan::read(
              std::string(argv[1]) + "/rank-" + std::to_string(rank) + ".bin");
          if (plan.rank != static_cast<uint32_t>(rank))
            throw std::runtime_error(
                "decoder rank/initialization plan mismatch");
          openpangu::checked(aclrtSetDevice(plan.device), "device");
          {
            pangu::Decoder decoder(plan, argv[2], info);
            pangu::V logits;
            int pos = 0;
            for (int token : prompt)
              logits = decoder.token(token, pos++);
            for (int i = 0; i < steps; ++i) {
              int next = int(std::max_element(logits.begin(), logits.end()) -
                             logits.begin());
              {
                openpangu::DeviceTensor selected({1}, ACL_INT32, 4),
                    peers({8}, ACL_INT32, 4);
                selected.upload(std::vector<int32_t>{next});
                openpangu::checked(HcclAllGather(selected.data, peers.data, 1,
                                                 HCCL_DATA_TYPE_INT32,
                                                 decoder.comm, decoder.stream),
                                   "token consensus");
                openpangu::checked(aclrtSynchronizeStream(decoder.stream),
                                   "consensus fence");
                for (auto value : peers.read<int32_t>())
                  if (value != next)
                    throw std::runtime_error("ranks disagree on token");
              }
              if (rank == 0) {
                double total = 0;
                for (float value : logits)
                  total += std::exp(double(value - logits[next]));
                double lse = logits[next] + std::log(total);
                std::vector<int> order(logits.size());
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(
                    order.begin(), order.begin() + 5, order.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
                std::cout << "{\"step\":" << i << ",\"token\":" << next
                          << ",\"logit\":" << logits[next] << ",\"top5\":[";
                for (int j = 0; j < 5; ++j) {
                  if (j)
                    std::cout << ',';
                  std::cout << "{\"token\":" << order[j]
                            << ",\"logprob\":" << (logits[order[j]] - lse)
                            << '}';
                }
                std::cout << "]}" << std::endl;
              }
              if (next == 148902 || i + 1 == steps)
                break;
              logits = decoder.token(next, pos++);
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

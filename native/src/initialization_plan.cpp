#include "initialization_plan.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace inferfabric::initialization {
namespace {
[[noreturn]] void fail(const char *message) {
  throw std::runtime_error(message);
}
uint64_t add(uint64_t a, uint64_t b) {
  if (a > UINT64_MAX - b)
    fail("initialization size overflow");
  return a + b;
}
uint64_t number(std::istream &stream, unsigned width) {
  uint64_t value = 0;
  for (unsigned i = 0; i < width; ++i) {
    int c = stream.get();
    if (c == EOF)
      fail("truncated initialization plan");
    value |= uint64_t(uint8_t(c)) << (8 * i);
  }
  return value;
}
uint32_t u32(std::istream &s) { return static_cast<uint32_t>(number(s, 4)); }
std::string string(std::istream &stream) {
  auto n = u32(stream);
  if (!n || n > 4096)
    fail("initialization string bound");
  std::string result(n, '\0');
  if (!stream.read(result.data(), n) || result.find('\0') != std::string::npos)
    fail("invalid initialization string");
  return result;
}
bool valid_string(const std::string &s) {
  return !s.empty() && s.size() <= 4096 && s.find('\0') == std::string::npos;
}
} // namespace
uint64_t element_bytes(DType dtype) {
  switch (dtype) {
  case DType::I8:
    return 1;
  case DType::Bf16:
    return 2;
  case DType::F32:
    return 4;
  }
  fail("unsupported initialization dtype");
}
uint64_t Parameter::bytes() const {
  if (shape.empty() || shape.size() > 8)
    fail("initialization shape rank");
  uint64_t bytes = element_bytes(dtype);
  for (auto dimension : shape) {
    if (dimension <= 0 || uint64_t(dimension) > UINT64_MAX / bytes)
      fail("initialization shape overflow");
    bytes *= uint64_t(dimension);
  }
  return bytes;
}
Plan Plan::read(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  char magic[8];
  if (!input.read(magic, 8) || std::memcmp(magic, "IFINIT01", 8))
    fail("unsupported initialization plan version");
  Plan p;
  p.rank = u32(input);
  p.device = u32(input);
  p.checkpoint_metadata_sha256 = string(input);
  auto count = u32(input);
  if (!count || count > 100000)
    fail("initialization parameter count");
  for (uint32_t i = 0; i < count; ++i) {
    Parameter parameter;
    parameter.id = u32(input);
    parameter.name = string(input);
    parameter.dtype = static_cast<DType>(u32(input));
    auto dimensions = u32(input);
    if (!dimensions || dimensions > 8)
      fail("initialization shape rank");
    for (uint32_t d = 0; d < dimensions; ++d) {
      auto value = number(input, 8);
      if (value > INT64_MAX)
        fail("initialization dimension overflow");
      parameter.shape.push_back(static_cast<int64_t>(value));
    }
    p.parameters.push_back(std::move(parameter));
  }
  if (u32(input) != count)
    fail("initialization instruction count");
  for (uint32_t i = 0; i < count; ++i) {
    if (u32(input) != 1)
      fail("unsupported initialization opcode");
    LoadSlice op;
    op.destination = u32(input);
    op.shard = string(input);
    op.offset = number(input, 8);
    op.bytes = number(input, 8);
    op.source_dtype = static_cast<DType>(u32(input));
    p.instructions.push_back(std::move(op));
  }
  if (input.peek() != EOF)
    fail("trailing initialization data");
  p.validate();
  return p;
}
void Plan::validate() const {
  if (parameters.empty() || parameters.size() > 100000 ||
      instructions.size() != parameters.size())
    fail("initialization count mismatch");
  if (checkpoint_metadata_sha256.size() != 64 ||
      checkpoint_metadata_sha256.find_first_not_of("0123456789abcdefABCDEF") !=
          std::string::npos)
    fail("initialization provenance");
  std::set<std::string> names;
  uint64_t total = 0;
  for (size_t i = 0; i < parameters.size(); ++i) {
    const auto &p = parameters[i];
    if (p.id != i || !valid_string(p.name) || !names.insert(p.name).second)
      fail("initialization parameter ID/name");
    total = add(total, p.bytes());
  }
  std::vector<bool> produced(parameters.size(), false);
  for (const auto &op : instructions) {
    if (op.destination >= parameters.size() || produced[op.destination])
      fail("initialization destination/duplicate producer");
    produced[op.destination] = true;
    if (!valid_string(op.shard) || op.shard == "." || op.shard == ".." ||
        op.shard.find_first_of("/\\:") != std::string::npos)
      fail("initialization shard path");
    const auto &p = parameters[op.destination];
    auto width = element_bytes(op.source_dtype);
    if (op.source_dtype != p.dtype &&
        !(op.source_dtype == DType::Bf16 && p.dtype == DType::F32))
      fail("unsupported initialization conversion");
    auto elements = p.bytes() / element_bytes(p.dtype);
    if (elements > UINT64_MAX / width || elements * width != op.bytes ||
        add(op.offset, op.bytes) > INT64_MAX)
      fail("initialization source range");
  }
}
ParameterId Plan::resolve(const std::string &name) const {
  for (const auto &p : parameters)
    if (p.name == name)
      return p.id;
  throw std::runtime_error("unbound parameter: " + name);
}
ResidentParameter::ResidentParameter(const Parameter &p,
                                     std::shared_ptr<DeviceMemory> backend)
    : backend_(std::move(backend)), descriptor(p), bytes(p.bytes()),
      dtype(p.dtype) {
  data = backend_->allocate(bytes);
  if (!data)
    fail("initialization allocation returned null");
}
ResidentParameter::~ResidentParameter() {
  if (data)
    backend_->release(data);
}
ResidentParameter &ParameterStore::at(ParameterId id) const {
  if (id >= parameters_.size() || !parameters_[id])
    fail("parameter is not initialized");
  return *parameters_[id];
}
ParameterStore Executor::execute(const Plan &plan, const std::string &directory,
                                 std::shared_ptr<DeviceMemory> backend,
                                 const Options &options) {
  plan.validate();
  if (!backend || !options.staging_bytes || options.staging_bytes % 4 ||
      options.staging_bytes > 64 * 1024 * 1024)
    fail("initialization backend/staging options");
  auto root = std::filesystem::canonical(directory);
  uint64_t resident = 0;
  for (const auto &p : plan.parameters)
    resident = add(resident, p.bytes());
  if (resident > options.resident_budget)
    fail("initialization resident budget exceeded");
  std::vector<std::filesystem::path> paths;
  for (const auto &op : plan.instructions) {
    auto file = std::filesystem::canonical(root / op.shard);
    if (file.parent_path() != root || !std::filesystem::is_regular_file(file) ||
        std::filesystem::file_size(file) < add(op.offset, op.bytes))
      fail("initialization shard range/path mismatch");
    paths.push_back(std::move(file));
  }
  std::vector<char> staging(options.staging_bytes);
  std::vector<float> converted;
  std::vector<char> roundtrip;
  if (options.verify_upload)
    roundtrip.resize(options.staging_bytes * 2);
  ParameterStore result;
  result.parameters_.resize(plan.parameters.size());
  for (size_t i = 0; i < plan.instructions.size(); ++i) {
    const auto &op = plan.instructions[i];
    const auto &p = plan.parameters[op.destination];
    auto tensor = std::make_unique<ResidentParameter>(p, backend);
    std::ifstream file(paths[i], std::ios::binary);
    if (!file.seekg(op.offset))
      fail("initialization seek failed");
    uint64_t done = 0, output_offset = 0;
    while (done < op.bytes) {
      size_t count = std::min<uint64_t>(staging.size(), op.bytes - done);
      if (!file.read(staging.data(), count))
        fail("short initialization read");
      const void *data = staging.data();
      size_t output = count;
      if (op.source_dtype != p.dtype) {
        converted.resize(count / 2);
        for (size_t j = 0; j < count / 2; ++j) {
          uint32_t bits = (uint32_t(uint8_t(staging[2 * j])) |
                           uint32_t(uint8_t(staging[2 * j + 1])) << 8)
                          << 16;
          std::memcpy(&converted[j], &bits, 4);
          if (!std::isfinite(converted[j]))
            fail("nonfinite initialization cast");
        }
        data = converted.data();
        output = count * 2;
      }
      auto destination = static_cast<char *>(tensor->data) + output_offset;
      backend->upload(destination, data, output);
      if (options.verify_upload) {
        backend->readback(roundtrip.data(), destination, output);
        if (std::memcmp(roundtrip.data(), data, output))
          fail("initialization upload mismatch");
      }
      done += count;
      output_offset += output;
    }
    result.source_bytes = add(result.source_bytes, op.bytes);
    result.resident_bytes = add(result.resident_bytes, tensor->bytes);
    result.parameters_[op.destination] = std::move(tensor);
  }
  return result;
}
} // namespace inferfabric::initialization

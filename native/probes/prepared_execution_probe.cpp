#include "../src/resident_tensor.h"
#include <iostream>
#include <sstream>
int main() {
  using namespace inferfabric;
  try {
    prepared::check(aclInit(nullptr), "init");
    prepared::check(aclrtSetDevice(0), "device");
    {
      prepared::Context context;
      openpangu::DeviceTensor results({128, 4}, ACL_BF16, 2);
      aclrtStream foreign = nullptr;
      prepared::check(aclrtCreateStream(&foreign), "foreign stream");
      bool rejected = false;
      try {
        context.require_stream(foreign);
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      prepared::check(aclrtDestroyStream(foreign), "foreign stream destroy");
      if (!rejected)
        throw std::runtime_error("cross-stream use accepted");
      rejected = false;
      try {
        results.upload(std::vector<uint16_t>(512));
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("implicit host upload accepted");
      // No intermediate fences. Every source vector and tensor lease is gone
      // before device completion; subsequent iterations recycle the same slots.
      for (int i = 0; i < 128; ++i) {
        context.begin_step(i);
        openpangu::DeviceTensor input({4}, ACL_FLOAT, 4),
            output({4}, ACL_BF16, 2);
        resident::control_upload(input, std::vector<float>(4, float(i + 1)));
        uint64_t bytes = 0;
        aclOpExecutor *executor = nullptr;
        prepared::check(prepared::query("cast", true, aclnnCastGetWorkspaceSize,
                                        input.tensor, ACL_BF16, output.tensor,
                                        &bytes, &executor),
                        "prepare");
        prepared::check(aclnnCast(context.workspace(bytes), bytes, executor,
                                  context.stream),
                        "cast");
        context.copy(static_cast<char *>(results.data) + i * 8, output.data, 8);
        context.end_step();
      }
      auto *host = context.output_host(results.bytes);
      {
        resident::TransferScope output(resident::TransferRole::RequestOutput);
        prepared::check(
            aclrtMemcpyAsync(host, results.bytes, results.data, results.bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST, context.stream),
            "output");
      }
      context.complete();
      for (int i = 0; i < 128; ++i) {
        float expected = float(i + 1);
        uint32_t bits;
        std::memcpy(&bits, &expected, sizeof(bits));
        for (int j = 0; j < 4; ++j)
          if (static_cast<uint16_t *>(host)[i * 4 + j] != uint16_t(bits >> 16))
            throw std::runtime_error("asynchronous lifetime/reuse mismatch");
      }
      // Same pinned slots may be rewritten only AFTER the output boundary.
      openpangu::DeviceTensor again({4}, ACL_FLOAT, 4);
      resident::control_upload(again, std::vector<float>(4, 7));
      context.complete();
      void *released = context.acquire(12);
      context.release(released);
      rejected = false;
      try {
        context.release(released);
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("double release accepted");
      std::ostringstream audit;
      context.write(audit);
      if (audit.str().find("\"replays\":127") == std::string::npos)
        throw std::runtime_error("invocation cache missed stable operands");
      std::cout << audit.str();
    }
    prepared::check(aclrtResetDevice(0), "reset");
    prepared::check(aclFinalize(), "finalize");
    std::cout << "prepared invocation, ordered reuse, pinned staging and "
                 "negative contracts passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

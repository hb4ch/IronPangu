#include "../src/resident_tensor.h"
#include "../src/resident_transfer.h"
#include <iostream>
int main() {
  using namespace inferfabric::resident;
  try {
    openpangu::checked(aclInit(nullptr), "init");
    openpangu::checked(aclrtSetDevice(0), "device");
    {
      openpangu::DeviceTensor source({4}, ACL_FLOAT, 4);
      int rejected = 0;
      try {
        source.upload(std::vector<float>{1, 2, 3, 4});
      } catch (const std::runtime_error &) {
        ++rejected;
      }
      control_upload(source, std::vector<float>{1, 2, 3, 4});
      try {
        source.read<float>();
      } catch (const std::runtime_error &) {
        ++rejected;
      }
      try {
        aclrtMemcpyAsync(nullptr, 0, nullptr, 0, ACL_MEMCPY_DEVICE_TO_HOST,
                         nullptr);
      } catch (const std::runtime_error &) {
        ++rejected;
      }
      auto owned = from_device(source.data, 4, ACL_FLOAT);
      auto shared = owned;
      owned = Tensor();
      auto tail = slice(shared, 2, 2), head = slice(shared, 0, 2);
      auto output = concat(tail, head);
      std::array<float, 4> values{};
      {
        TransferScope result(TransferRole::RequestOutput);
        openpangu::checked(aclrtMemcpy(values.data(), 16, output.data(), 16,
                                       ACL_MEMCPY_DEVICE_TO_HOST),
                           "request output");
      }
      if (values != std::array<float, 4>{3, 4, 1, 2} || rejected != 3)
        throw std::runtime_error("resident ownership/guard regression");
      // Numeric conversion contract: BF16 ties must round to nearest even.
      control_upload(source, std::vector<float>{1.00390625f, 1.01171875f,
                                                -1.00390625f, -1.01171875f});
      openpangu::DeviceTensor bf({4}, ACL_BF16, 2);
      aclrtStream stream = nullptr;
      openpangu::checked(aclrtCreateStream(&stream), "stream");
      openpangu::Operation op;
      openpangu::checked(aclnnCastGetWorkspaceSize(source.tensor, ACL_BF16,
                                                   bf.tensor, &op.bytes,
                                                   &op.executor),
                         "cast prepare");
      op.allocate();
      op.run(aclnnCast, stream);
      openpangu::checked(aclrtSynchronizeStream(stream), "cast completion");
      {
        TransferScope result(TransferRole::RequestOutput);
        auto encoded = bf.read<uint16_t>();
        if (encoded != std::vector<uint16_t>{0x3f80, 0x3f82, 0xbf80, 0xbf82})
          throw std::runtime_error("BF16 conversion");
      }
      openpangu::checked(aclrtDestroyStream(stream), "stream destruction");
      try {
        slice(shared, 3, 2);
        throw std::logic_error("slice unexpectedly accepted");
      } catch (const std::runtime_error &) {
      }
    }
    openpangu::checked(aclrtResetDevice(0), "reset");
    openpangu::checked(aclFinalize(), "finalize");
    std::cout << "device ownership, copy, slice, concat and implicit transfer "
                 "rejection passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

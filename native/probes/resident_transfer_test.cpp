#include "../src/resident_transfer.h"
#include <iostream>
#include <sstream>
int main() {
  using namespace inferfabric::resident;
  int rejected = 0;
  for (auto kind : {ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST}) {
    try {
      transfers.check(4, kind);
    } catch (const std::runtime_error &) {
      ++rejected;
    }
  }
  {
    TransferScope control(TransferRole::ControlInput);
    transfers.check(4, ACL_MEMCPY_HOST_TO_DEVICE);
    try {
      transfers.check(4, ACL_MEMCPY_DEVICE_TO_HOST);
    } catch (const std::runtime_error &) {
      ++rejected;
    }
  }
  {
    TransferScope output(TransferRole::RequestOutput);
    transfers.check(8, ACL_MEMCPY_DEVICE_TO_HOST);
    try {
      transfers.check(4, ACL_MEMCPY_HOST_TO_DEVICE);
    } catch (const std::runtime_error &) {
      ++rejected;
    }
  }
  {
    TransferScope parameter(TransferRole::ParameterPreparation);
    transfers.check(16, ACL_MEMCPY_DEVICE_TO_HOST);
    transfers.check(32, ACL_MEMCPY_HOST_TO_DEVICE);
  }
  transfers.check(4096, ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (rejected != 4 || transfers.role != TransferRole::None ||
      transfers.rejected != 4 || transfers.d2h_bytes[3] != 8 ||
      transfers.h2d_bytes[2] != 32)
    return 1;
  std::ostringstream output;
  transfers.write(output);
  if (output.str().find("request_output") == std::string::npos)
    return 1;
  std::cout << "resident transfer negative tests passed\n";
}

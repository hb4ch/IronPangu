#include "initialization_acl.h"
#include <acl/acl.h>
#include <iostream>
#include <stdexcept>
using namespace inferfabric::initialization;
void check(aclError e) {
  if (e != ACL_SUCCESS)
    throw std::runtime_error("ACL status " + std::to_string(e));
}
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: initialization_probe PLAN CHECKPOINT\n";
    return 2;
  }
  try {
    auto plan = Plan::read(argv[1]);
    check(aclInit(nullptr));
    check(aclrtSetDevice(plan.device));
    {
      Options options;
      options.verify_upload = true;
      auto weights =
          Executor::execute(plan, argv[2], acl_device_memory(), options);
      std::cout << "{\"status\":\"passed\",\"rank\":" << plan.rank
                << ",\"weights\":" << weights.size()
                << ",\"stored_bytes\":" << weights.source_bytes
                << ",\"prepared_bytes\":" << weights.resident_bytes
                << ",\"full_upload_readback\":true}\n";
    }
    check(aclrtResetDevice(plan.device));
    check(aclFinalize());
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#include "compiled_kernel.h"
#include <iostream>
#include <stdexcept>
using namespace inferfabric;
void require(bool value) {
  if (!value)
    throw std::runtime_error("kernel test failed");
}
template <class F> void rejects(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::exception &) {
    failed = true;
  }
  require(failed);
}
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "usage: compiled_kernel_test LIBRARY BAD_ABI_LIBRARY");
    rejects([&] { CompiledKernel k(argv[1], "wrong", "cpu"); });
    rejects([&] { CompiledKernel k(argv[1], "test.scale.v1", "ascend"); });
    rejects([&] { CompiledKernel k(argv[2], "test.scale.v1", "cpu"); });
    float x[] = {2, 3}, y[] = {0, 0};
    IFKernelTensorV1 input{};
    input.struct_bytes = sizeof(input);
    input.dtype = 3;
    input.rank = 1;
    input.access = 1;
    input.data = x;
    input.allocation_bytes = 8;
    input.shape[0] = 2;
    input.strides[0] = 1;
    auto output = input;
    output.data = y;
    output.access = 2;
    auto k = std::make_unique<CompiledKernel>(argv[1], "test.scale.v1", "cpu");
    auto bad = input;
    bad.allocation_bytes = 4;
    rejects([&] { k->prepare({bad}, {output}); });
    bad = input;
    bad.strides[0] = 2;
    rejects([&] { k->prepare({bad}, {output}); });
    bad = input;
    bad.dtype = 99;
    rejects([&] { k->prepare({bad}, {output}); });
    bad = input;
    bad.struct_bytes = 0;
    rejects([&] { k->prepare({bad}, {output}); });
    rejects([&] { k->prepare({input, input}, {output}); });
    auto alias = output;
    alias.data = x;
    rejects([&] { k->prepare({input}, {alias}); });
    auto prepared = k->prepare({input}, {output});
    k.reset(); // Prepared state pins its library.
    alignas(64) char workspace[128];
    rejects([&] { prepared.enqueue(workspace, 32, nullptr); });
    rejects([&] { prepared.enqueue(workspace + 1, 64, nullptr); });
    prepared.enqueue(workspace, 128, nullptr);
    require(y[0] == 4 && y[1] == 6);
    rejects([&] { prepared.enqueue(workspace, 128, nullptr); });
    std::cout << "compiled kernel ABI: contract/target, tensor "
                 "bounds/strides/aliases, workspace, retained library and "
                 "execution passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

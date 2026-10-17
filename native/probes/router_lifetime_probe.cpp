#include "../src/openpangu_ops.h"
#include <iostream>

// Exercise executor ownership independently of model loading. No graph capture,
// weight payload or host activation readback is needed for workspace preparation.
int main() {
  try {
    openpangu::checked(aclInit(nullptr),"init");
    openpangu::checked(aclrtSetDevice(0),"device");
    for(int width:{16,4096}) {
      std::cerr<<"router lifetime width="<<width<<" begin\n";
      {
        inferfabric::prepared::Context context;
        {openpangu::Router router(width,2560,256,8,context.stream);}
        std::cerr<<"router destroyed width="<<width<<'\n';
      }
      std::cerr<<"context destroyed width="<<width<<'\n';
    }
    openpangu::checked(aclrtResetDevice(0),"reset");
    openpangu::checked(aclFinalize(),"finalize");
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}

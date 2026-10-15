#include "openpangu_probe_common.h"
int main(int argc,char **argv) {
  if(argc!=4){std::cerr<<"usage: openpangu_linear_probe FIXTURE DEVICE BATCH\n";return 2;}
  try {
    int device=std::stoi(argv[2]);check(aclInit(nullptr),"init");check(aclrtSetDevice(device),"device");
    {Runner r(argv[1],std::stoi(argv[3]));r.prepare();r.run();
      std::cout<<"{\"status\":\"passed\",\"device\":"<<device<<",\"batch\":"<<r.m<<",\"input\":"<<r.k<<",\"output\":"<<r.n<<",\"cases\":3,\"graph_passes\":3,\"checked_values\":"<<r.checked<<",\"max_abs_error\":"<<r.max_error<<",\"max_quant_integer_error\":"<<r.max_quant_error<<",\"workspace_bytes\":"<<std::max(r.dq_bytes,r.mm_bytes)<<"}\n";}
    check(aclrtResetDevice(device),"reset");check(aclFinalize(),"finalize");return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}

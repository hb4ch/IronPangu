// Isolate host metadata recycling without loading model weights.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_cast.h>
#include <malloc.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

void check(int status,const char *name) {
  if(status)throw std::runtime_error(std::string(name)+": "+std::to_string(status));
}
void report(int batch) {
  auto m=mallinfo2();
  printf("{\"batch\":%d,\"allocated_arena_bytes\":%zu,\"free_arena_bytes\":%zu,\"mmap_bytes\":%zu}\n",
         batch,m.uordblks,m.fordblks,m.hblkhd);fflush(stdout);
}
int main(int argc,char **argv) try {
  if(argc!=2 || (strcmp(argv[1],"ordinary") && strcmp(argv[1],"repeatable") && strcmp(argv[1],"retained") && strcmp(argv[1],"released")))
    throw std::runtime_error("usage: host_metadata_probe ordinary|repeatable|retained|released");
  const bool repeatable=!strcmp(argv[1],"repeatable");
  const bool fresh=!strcmp(argv[1],"retained") || !strcmp(argv[1],"released");
  const bool release=!strcmp(argv[1],"released");
  check(setenv("ACLNN_CACHE_LIMIT","0",1),"cache setting");
  check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"device");
  aclrtStream stream=nullptr;check(aclrtCreateStream(&stream),"stream");
  void *input=nullptr,*output=nullptr,*workspace=nullptr;
  constexpr int64_t rows=4096,columns=192,width=64,start=128;
  constexpr size_t input_bytes=rows*columns*2,output_bytes=rows*width*2;
  check(aclrtMalloc(&input,input_bytes,ACL_MEM_MALLOC_HUGE_FIRST),"input");
  check(aclrtMalloc(&output,output_bytes,ACL_MEM_MALLOC_HUGE_FIRST),"output");
  check(aclrtMemsetAsync(input,input_bytes,0,input_bytes,stream),"initialize");
  int64_t shape[]={rows,width},stride[]={columns,1},packed[]={width,1},storage=rows*columns-start;
  auto *x=aclCreateTensor(shape,2,ACL_BF16,stride,0,ACL_FORMAT_ND,&storage,1,static_cast<char *>(input)+start*2);
  auto *y=aclCreateTensor(shape,2,ACL_BF16,packed,0,ACL_FORMAT_ND,shape,2,output);
  if(!x || !y)throw std::runtime_error("descriptors");
  uint64_t capacity=0;std::vector<aclOpExecutor *> held;
  std::vector<aclTensor *> descriptors;
  for(int batch=0;batch<17;++batch) {
    for(int i=0;i<128;++i) {
      uint64_t bytes=0;aclOpExecutor *executor=nullptr;
      auto *source=x,*dest=y;
      if(fresh) {
        source=aclCreateTensor(shape,2,ACL_BF16,stride,0,ACL_FORMAT_ND,&storage,1,static_cast<char *>(input)+start*2);
        dest=aclCreateTensor(shape,2,ACL_BF16,packed,0,ACL_FORMAT_ND,shape,2,output);
        if(!source || !dest)throw std::runtime_error("fresh descriptors");
        descriptors.push_back(source);descriptors.push_back(dest);
      }
      check(aclnnCastGetWorkspaceSize(source,ACL_BF16,dest,&bytes,&executor),"prepare");
      if(bytes>capacity) {
        if(batch || i)throw std::runtime_error("workspace grew after first query");
        check(aclrtMalloc(&workspace,bytes,ACL_MEM_MALLOC_HUGE_FIRST),"workspace");capacity=bytes;
      }
      if(repeatable){check(aclSetAclOpExecutorRepeatable(executor),"repeatable");held.push_back(executor);}
      check(aclnnCast(workspace,bytes,executor,stream),"launch");
    }
    check(aclrtSynchronizeStream(stream),"batch complete");
    for(auto *executor:held)check(aclDestroyAclOpExecutor(executor),"destroy repeatable");
    held.clear();
    if(release){for(auto *tensor:descriptors)aclDestroyTensor(tensor);descriptors.clear();}
    report(batch);
  }
  for(auto *tensor:descriptors)aclDestroyTensor(tensor);
  aclDestroyTensor(x);aclDestroyTensor(y);
  if(workspace)aclrtFree(workspace);
  aclrtFree(input);aclrtFree(output);aclrtDestroyStream(stream);
  check(aclFinalize(),"finalize");
  return 0;
} catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 1;}

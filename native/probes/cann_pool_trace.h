#pragma once
// Offline CANN 9.1 diagnostic: observe pool-client ownership separately from
// malloc backing blocks. Never inspect private headers or alter pool policy.
#include "sampled_allocations.h"
namespace sampled {
inline void *pool_symbol(const char *symbol) {
  void *library=dlopen("libnnopbase.so",RTLD_LAZY|RTLD_NOLOAD);
  void *function=library?dlsym(library,symbol):nullptr;
  if(library)dlclose(library);
  if(!function) {
    static const char message[]="CANN pool diagnostic symbol resolution failed\n";
    write(STDERR_FILENO,message,sizeof(message)-1);_exit(125);
  }
  return function;
}
}
extern "C" void *_ZN2op8internal8AllocateEm(size_t bytes) {
  using Function=void *(*)(size_t);
  static const auto original=reinterpret_cast<Function>(sampled::pool_symbol("_ZN2op8internal8AllocateEm"));
  void *result=original(bytes);sampled::record(result,bytes,1);return result;
}
extern "C" void _ZN2op8internal10DeAllocateEPv(void *pointer) {
  using Function=void (*)(void *);
  static const auto original=reinterpret_cast<Function>(sampled::pool_symbol("_ZN2op8internal10DeAllocateEPv"));
  sampled::forget(pointer,1);original(pointer);
}

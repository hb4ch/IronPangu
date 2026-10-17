#pragma once
// Linux/glibc-only offline diagnostic. Fixed storage; sampled allocation stacks,
// exact removal of recorded pointers. Not a production allocator replacement.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include <pthread.h>
#include <unistd.h>
#include <dlfcn.h>

extern "C" void *__libc_malloc(size_t);
extern "C" void *__libc_calloc(size_t,size_t);
extern "C" void *__libc_realloc(void *,size_t);
extern "C" void __libc_free(void *);
extern "C" void *__libc_memalign(size_t,size_t);

namespace sampled {
constexpr size_t shards=32, capacity=2048, depth=12;
struct Entry { void *pointer; size_t bytes; uint64_t serial; unsigned kind; int frames; void *stack[depth]; };
struct Shard { pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER; Entry entries[capacity]{}; };
inline Shard table[shards];
inline std::atomic<bool> enabled{false};
inline std::atomic<uint64_t> serial{0}, dropped{0};
inline unsigned rate=1024;
inline thread_local unsigned countdown[2]{};
inline thread_local bool internal=false;
inline size_t hash(void *p) {
  auto x=reinterpret_cast<uintptr_t>(p)>>4;
  x^=x>>23; x*=0x2127599bf4325c37ULL; return x^(x>>47);
}
inline void forget(void *p,unsigned kind=0) {
  if(!p || !enabled.load(std::memory_order_relaxed)) return;
  const auto h=hash(p)^kind;auto &s=table[h%shards];
  pthread_mutex_lock(&s.mutex);
  for(size_t j=0;j<capacity;++j) {
    auto &e=s.entries[(h/shards+j)%capacity];
    if(!e.pointer) break;
    if(e.pointer==p && e.kind==kind){e.pointer=reinterpret_cast<void *>(1);break;}
  }
  pthread_mutex_unlock(&s.mutex);
}
inline void record(void *p,size_t bytes,unsigned kind=0) {
  if(!p || internal || !enabled.load(std::memory_order_relaxed)) return;
  if(countdown[kind]){--countdown[kind];return;}
  countdown[kind]=rate-1;
  internal=true;
  Entry next{};next.pointer=p;next.bytes=bytes;next.serial=++serial;next.kind=kind;
  next.frames=backtrace(next.stack,depth);
  const auto h=hash(p)^kind;auto &s=table[h%shards];
  pthread_mutex_lock(&s.mutex);
  size_t slot=capacity;
  for(size_t j=0;j<capacity;++j) {
    const auto at=(h/shards+j)%capacity;auto &e=s.entries[at];
    if(e.pointer==reinterpret_cast<void *>(1) && slot==capacity)slot=at;
    if(!e.pointer){if(slot==capacity)slot=at;break;}
  }
  if(slot==capacity)++dropped;else s.entries[slot]=next;
  pthread_mutex_unlock(&s.mutex);
  internal=false;
}
// Snapshot retains raw module-relative return addresses, even for stripped DSOs.
// dladdr is outside table locks: loader work can allocate/free internally.
inline void snapshot(FILE *out,long long time_ns) {
  if(!enabled.load())return;
  internal=true;
  auto *copy=static_cast<Entry *>(__libc_malloc(sizeof(Entry)*capacity));
  if(!copy){++dropped;internal=false;return;}
  fprintf(out,"snapshot %lld rate=%u dropped=%llu\n",time_ns,rate,(unsigned long long)dropped.load());
  for(auto &s:table) {
    size_t count=0;
    pthread_mutex_lock(&s.mutex);
    for(auto &e:s.entries)if(e.pointer && e.pointer!=reinterpret_cast<void *>(1))copy[count++]=e;
    pthread_mutex_unlock(&s.mutex);
    for(size_t j=0;j<count;++j) {
      const auto &e=copy[j];
      fprintf(out,"allocation %llu %zu kind=%s",(unsigned long long)e.serial,e.bytes,e.kind?"cann-client":"libc");
      for(int i=0;i<e.frames;++i) {
        Dl_info info{};
        if(dladdr(e.stack[i],&info) && info.dli_fbase)
          fprintf(out," %s+0x%zx",info.dli_fname,size_t(reinterpret_cast<uintptr_t>(e.stack[i])-reinterpret_cast<uintptr_t>(info.dli_fbase)));
        else fprintf(out," unknown+%p",e.stack[i]);
      }
      fputc('\n',out);
    }
  }
  fputs("end\n",out);fflush(out);
  __libc_free(copy);internal=false;
}
}
extern "C" void *malloc(size_t n) {auto *p=__libc_malloc(n);sampled::record(p,n);return p;}
extern "C" void *calloc(size_t n,size_t w) {auto *p=__libc_calloc(n,w);if(p)sampled::record(p,n*w);return p;}
extern "C" void free(void *p) {sampled::forget(p);__libc_free(p);}
extern "C" void *realloc(void *p,size_t n) {
  if(!p || !sampled::enabled.load()) {auto *next=__libc_realloc(p,n);sampled::record(next,n);return next;}
  // Serialize address reuse with this shard. Failed realloc retains its sample.
  const auto h=sampled::hash(p);auto &s=sampled::table[h%sampled::shards];
  pthread_mutex_lock(&s.mutex);
  auto *next=__libc_realloc(p,n);
  if(next || n==0)for(size_t j=0;j<sampled::capacity;++j) {
    auto &e=s.entries[(h/sampled::shards+j)%sampled::capacity];
    if(!e.pointer)break;
    if(e.pointer==p && e.kind==0){e.pointer=reinterpret_cast<void *>(1);break;}
  }
  pthread_mutex_unlock(&s.mutex);
  if(next)sampled::record(next,n);
  return next;
}
extern "C" void *memalign(size_t a,size_t n) {auto *p=__libc_memalign(a,n);sampled::record(p,n);return p;}

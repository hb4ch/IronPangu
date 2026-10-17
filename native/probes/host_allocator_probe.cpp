// Optional LD_PRELOAD diagnostic. No ACL calls, device reads or inference hooks.
// Build in the Linux container: g++ -shared -fPIC -O2 -pthread ... -o probe.so
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include "sampled_allocations.h"

namespace {
std::atomic<bool> stopping{false};
pthread_t worker;
int output = -1;
bool started = false;
pid_t owner = 0;
FILE *stacks=nullptr;
const char *start_file=nullptr;
char request_path[4096]{},ack_path[4096]{};
unsigned long last_request=0;
long long sample_time=0;
void sample() {
  if(stacks && !sampled::enabled.load() && (!start_file || access(start_file,F_OK)==0))sampled::enabled.store(true);
  const auto memory = mallinfo2();
  timespec now{};
  clock_gettime(CLOCK_REALTIME, &now);
  sample_time=static_cast<long long>(now.tv_sec)*1000000000+now.tv_nsec;
  char line[512];
  const int count = snprintf(line, sizeof(line),
      "{\"time_ns\":%lld,\"pid\":%d,\"arena_bytes\":%zu,"
      "\"allocated_arena_bytes\":%zu,\"free_arena_bytes\":%zu,"
      "\"mmap_bytes\":%zu,\"mmap_regions\":%zu}\n",
      static_cast<long long>(now.tv_sec)*1000000000+now.tv_nsec,
      int(getpid()), memory.arena, memory.uordblks, memory.fordblks,
      memory.hblkhd, memory.hblks);
  if (count <= 0 || size_t(count) >= sizeof(line)) return;
  size_t written=0;
  while(written<size_t(count)) {
    const auto n=write(output,line+written,size_t(count)-written);
    if(n<0 && errno==EINTR) continue;
    if(n<=0) break;
    written+=size_t(n);
  }
  if(stacks)sampled::snapshot(stacks,static_cast<long long>(now.tv_sec)*1000000000+now.tv_nsec);
}
void *run(void *) {
  while(!stopping.load()) {
    sample();
    for(int i=0;i<20 && !stopping.load();++i) {
      timespec delay{0,100000000};
      nanosleep(&delay,nullptr);
      if(*request_path) {
        char value[64]{};const int fd=open(request_path,O_RDONLY|O_CLOEXEC);
        if(fd>=0) {
          const auto n=read(fd,value,sizeof(value)-1);close(fd);
          const auto request=n>0?strtoul(value,nullptr,10):0;
          if(request>last_request) {
            sample();last_request=request;
            const int ack=open(ack_path,O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0600);
            if(ack<0)_exit(125);
            const int count=snprintf(value,sizeof(value),"%lu %lld\n",request,sample_time);
            if(write(ack,value,count)!=count)_exit(125);
            close(ack);
          }
        }
      }
    }
  }
  sample();
  return nullptr;
}
__attribute__((constructor)) void begin() {
  const char *path=getenv("INFERFABRIC_HOST_ALLOCATOR_AUDIT");
  if(!path || !*path) return;
  // Vendor helpers inherit LD_PRELOAD. Never sample or reject their startup.
  const char *target=getenv("INFERFABRIC_HOST_ALLOCATOR_EXECUTABLE");
  char executable[4096];
  const auto length=readlink("/proc/self/exe",executable,sizeof(executable)-1);
  if(!target || length<0) return;
  executable[length]='\0';
  if(std::strcmp(executable,target)!=0) return;
  if(const char *stack_path=getenv("INFERFABRIC_HOST_ALLOCATION_STACKS")) {
    const int fd=open(stack_path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(fd<0 || !(stacks=fdopen(fd,"w")))_exit(125);
    start_file=getenv("INFERFABRIC_HOST_ALLOCATION_START");
    if(strlen(stack_path)+9>=sizeof(request_path))_exit(125);
    snprintf(request_path,sizeof(request_path),"%s.request",stack_path);
    snprintf(ack_path,sizeof(ack_path),"%s.ack",stack_path);
    if(const char *value=getenv("INFERFABRIC_HOST_ALLOCATION_RATE")) {
      char *end=nullptr;const auto rate=strtoul(value,&end,10);
      if(!end || *end || rate<1 || rate>1048576)_exit(125);
      sampled::rate=unsigned(rate);
    }
  }
  output=open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
  // The frontend installs sigwait after process startup. A preload thread must
  // not receive its termination signals before that policy is inherited.
  sigset_t all,previous;
  pthread_atfork(nullptr,nullptr,[]{sampled::enabled.store(false);});
  sigfillset(&all);
  const int masked=pthread_sigmask(SIG_BLOCK,&all,&previous);
  const int created=(output>=0 && masked==0)?pthread_create(&worker,nullptr,run,nullptr):-1;
  if(masked==0) pthread_sigmask(SIG_SETMASK,&previous,nullptr);
  if(output<0 || created!=0) {
    // Explicitly requested diagnostics must never silently disappear.
    static const char error[]="host allocator probe initialization failed\n";
    write(STDERR_FILENO,error,sizeof(error)-1);
    _exit(125);
  }
  started=true;
  owner=getpid();
}
__attribute__((destructor)) void end() {
  if(!started || owner!=getpid()) return;
  stopping.store(true);
  pthread_join(worker,nullptr);
  sampled::enabled.store(false);
  if(stacks)fclose(stacks);
  close(output);
}
}

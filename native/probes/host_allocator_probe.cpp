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

namespace {
std::atomic<bool> stopping{false};
pthread_t worker;
int output = -1;
bool started = false;
pid_t owner = 0;
void sample() {
  const auto memory = mallinfo2();
  timespec now{};
  clock_gettime(CLOCK_REALTIME, &now);
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
}
void *run(void *) {
  while(!stopping.load()) {
    sample();
    for(int i=0;i<20 && !stopping.load();++i) {
      timespec delay{0,100000000};
      nanosleep(&delay,nullptr);
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
  output=open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
  // The frontend installs sigwait after process startup. A preload thread must
  // not receive its termination signals before that policy is inherited.
  sigset_t all,previous;
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
  close(output);
}
}

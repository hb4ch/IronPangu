#include "inferfabric_acl.h"
#include <cstdio>
int main() {
    inferfabric_acl_session* s = nullptr;
    if (inferfabric_acl_open(1, &s)) return 1;
    inferfabric_acl_memory_stats stats{};
    uint64_t handle = 0;
    bool ok = !inferfabric_acl_set_memory_budget(s, 2ULL << 20, 256ULL << 20)
        && !inferfabric_acl_allocate(s, 1ULL << 20, &handle)
        && inferfabric_acl_allocate(s, 3ULL << 20, &handle) != 0
        && !inferfabric_acl_memory_snapshot(s, &stats)
        && stats.buffer_bytes == (1ULL << 20);
    std::printf("allocation_budget_guard=%s buffers=%llu\n", ok ? "passed" : "failed", (unsigned long long)stats.buffer_bytes);
    inferfabric_acl_close(s);
    return ok ? 0 : 1;
}

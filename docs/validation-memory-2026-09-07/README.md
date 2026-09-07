# Reproducing the allocation guard test

Inside `ironpangu-npu`, with device 1 free and the CANN environment sourced:

```sh
g++ -std=c++17 -I native/include memory_probe.cpp -L native-build -lpangu_acl \
  -Wl,-rpath,/data/p00603624/ironpangu/native-build -o memory_probe
./memory_probe
```

Observed: `allocation_budget_guard=passed buffers=1048576`.

The HTTP regression also consumed a 512-token prompt before testing overlapping admission. Its JSON summary predates a dedicated long-prompt output field; that assertion is in `frontend/src/native_smoke.rs` and fails the command on mismatch.

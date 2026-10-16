#include "../src/prepared_weight_views.h"
#include "../src/resident_tensor.h"
#include <iostream>
#include <sstream>
int main() {
  using namespace inferfabric;
  try {
    prepared::check(aclInit(nullptr), "init");
    prepared::check(aclrtSetDevice(0), "device");
    {
      const char *path="probe-workspace.txt";
      {std::ofstream out(path);out<<"IFWORKSPACE1 123 "<<prepared::ArenaLayout::fingerprint_file("/proc/self/exe")<<" 1024\n";}
      {
        prepared::Context fixed("",123,path);
        auto *address=fixed.workspace(1);
        if(address!=fixed.workspace(512) || address!=fixed.workspace(1024))
          throw std::runtime_error("planned workspace moved");
        bool rejected=false;
        try{fixed.workspace(1025);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected || address!=fixed.workspace(1))throw std::runtime_error("workspace overflow accepted or changed address");
        std::ostringstream audit;fixed.write(audit);
        if(audit.str().find("\"growth_allocations\":0")==std::string::npos)
          throw std::runtime_error("planned workspace grew");
      }
      std::remove(path);
    }
    {
      prepared::Context context;
      openpangu::DeviceTensor results({128, 4}, ACL_BF16, 2);
      resident::PreparedWeightViews weight_views;
      openpangu::DeviceTensor source_weight({2, 3, 4}, ACL_FLOAT, 4);
      std::vector<float> source_values(24);
      for (int i = 0; i < 24; ++i)
        source_values[i] = float(i);
      resident::control_upload(source_weight, source_values);
      auto *packed = weight_views.get(source_weight.data, 24, {2, 4, 2},
                                      {12, 1, 4}, ACL_FLOAT);
      if (packed != weight_views.get(source_weight.data, 24, {2, 4, 2},
                                     {12, 1, 4}, ACL_FLOAT))
        throw std::runtime_error("immutable layout cache missed");
      // Validate materialization through a device operation, including a gap
      // between batches and a transposed inner dimension.
      openpangu::DeviceTensor packed_result({2, 4, 2}, ACL_FLOAT, 4);
      uint64_t packed_bytes = 0;
      aclOpExecutor *packed_executor = nullptr;
      prepared::check(
          aclnnCastGetWorkspaceSize(packed, ACL_FLOAT, packed_result.tensor,
                                    &packed_bytes, &packed_executor),
          "packed copy prepare");
      prepared::check(aclnnCast(context.workspace(packed_bytes), packed_bytes,
                                packed_executor, context.stream),
                      "packed copy");
      auto *packed_host = context.output_host(packed_result.bytes);
      {
        resident::TransferScope output(resident::TransferRole::RequestOutput);
        prepared::check(
            aclrtMemcpyAsync(packed_host, packed_result.bytes,
                             packed_result.data, packed_result.bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST, context.stream),
            "packed view output");
      }
      context.complete();
      for (int b = 0; b < 2; ++b)
        for (int col = 0; col < 4; ++col)
          for (int row = 0; row < 2; ++row)
            if (static_cast<float *>(packed_host)[b * 8 + col * 2 + row] !=
                b * 12 + row * 4 + col)
              throw std::runtime_error(
                  "immutable strided materialization mismatch");
      auto *strided = context.strided_descriptor(results.data, 512, {2, 3, 4},
                                                 {256, 8, 1}, ACL_BF16);
      if (strided != context.strided_descriptor(results.data, 512, {2, 3, 4},
                                                {256, 8, 1}, ACL_BF16) ||
          strided == context.strided_descriptor(results.data, 512, {2, 3, 4},
                                                {128, 8, 1}, ACL_BF16))
        throw std::runtime_error("strided descriptor cache key mismatch");
      for (auto strides :
           {std::vector<int64_t>{512, 8, 1}, std::vector<int64_t>{256, 0, 1},
            std::vector<int64_t>{256, 8}}) {
        bool bounds_rejected = false;
        try {
          context.strided_descriptor(results.data, 512, {2, 3, 4}, strides,
                                     ACL_BF16);
        } catch (const std::runtime_error &) {
          bounds_rejected = true;
        }
        if (!bounds_rejected)
          throw std::runtime_error("invalid strided view accepted");
      }
      aclrtStream foreign = nullptr;
      prepared::check(aclrtCreateStream(&foreign), "foreign stream");
      bool rejected = false;
      try {
        context.require_stream(foreign);
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      prepared::check(aclrtDestroyStream(foreign), "foreign stream destroy");
      if (!rejected)
        throw std::runtime_error("cross-stream use accepted");
      rejected = false;
      try {
        results.upload(std::vector<uint16_t>(512));
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("implicit host upload accepted");
      // No intermediate fences. Every source vector and tensor lease is gone
      // before device completion; subsequent iterations recycle the same slots.
      for (int i = 0; i < 128; ++i) {
        context.begin_step(i);
        openpangu::DeviceTensor input({4}, ACL_FLOAT, 4),
            output({4}, ACL_BF16, 2);
        resident::control_upload(input, std::vector<float>(4, float(i + 1)));
        uint64_t bytes = 0;
        aclOpExecutor *executor = nullptr;
        prepared::check(prepared::query("cast", true, aclnnCastGetWorkspaceSize,
                                        input.tensor, ACL_BF16, output.tensor,
                                        &bytes, &executor),
                        "prepare");
        prepared::check(aclnnCast(context.workspace(bytes), bytes, executor,
                                  context.stream),
                        "cast");
        context.copy(static_cast<char *>(results.data) + i * 8, output.data, 8);
        context.end_step();
      }
      auto *host = context.output_host(results.bytes);
      {
        resident::TransferScope output(resident::TransferRole::RequestOutput);
        prepared::check(
            aclrtMemcpyAsync(host, results.bytes, results.data, results.bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST, context.stream),
            "output");
      }
      context.complete();
      for (int i = 0; i < 128; ++i) {
        float expected = float(i + 1);
        uint32_t bits;
        std::memcpy(&bits, &expected, sizeof(bits));
        for (int j = 0; j < 4; ++j)
          if (static_cast<uint16_t *>(host)[i * 4 + j] != uint16_t(bits >> 16))
            throw std::runtime_error("asynchronous lifetime/reuse mismatch");
      }
      // Aliases and nested views retain the whole pool lease after their
      // producer descriptor is destroyed; a same-sized allocation cannot steal
      // it.
      resident::Tensor alias;
      void *original = nullptr;
      {
        openpangu::DeviceTensor source({8}, ACL_FLOAT, 4);
        original = source.data;
        resident::control_upload(source,
                                 std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
        alias =
            resident::Tensor::alias(source, ACL_FLOAT).view(2, 5).view(1, 2);
      }
      {
        openpangu::DeviceTensor competing({8}, ACL_FLOAT, 4);
        if (competing.data == original)
          throw std::runtime_error("live view allocation reused");
        resident::control_upload(competing, std::vector<float>(8, -1));
        rejected = false;
        try {
          alias.view(1, 2);
        } catch (const std::runtime_error &) {
          rejected = true;
        }
        if (!rejected)
          throw std::runtime_error("view overflow accepted");
        auto *output = context.output_host(alias.bytes());
        {
          resident::TransferScope permission(
              resident::TransferRole::RequestOutput);
          prepared::check(aclrtMemcpyAsync(output, alias.bytes(), alias.data(),
                                           alias.bytes(),
                                           ACL_MEMCPY_DEVICE_TO_HOST,
                                           context.stream),
                          "view output");
        }
        context.complete();
        if (static_cast<float *>(output)[0] != 3 ||
            static_cast<float *>(output)[1] != 4)
          throw std::runtime_error("view offset/lifetime mismatch");
      }
      alias = {};
      void *recycled = context.acquire(32);
      if (recycled != original)
        throw std::runtime_error("last view did not release allocation");
      context.release(recycled);
      // Same pinned slots may be rewritten only AFTER the output boundary.
      openpangu::DeviceTensor again({4}, ACL_FLOAT, 4);
      resident::control_upload(again, std::vector<float>(4, 7));
      context.complete();
      void *released = context.acquire(12);
      context.release(released);
      rejected = false;
      try {
        context.release(released);
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("double release accepted");
      std::ostringstream audit;
      context.write(audit);
      if (audit.str().find("\"replays\":127") == std::string::npos)
        throw std::runtime_error("invocation cache missed stable operands");
      std::cout << audit.str();
    }
    {
      const char *path="prepared-arena-test.txt";
      {std::ofstream manifest(path);manifest<<"IFARENA1 123 "<<prepared::ArenaLayout::fingerprint_file("/proc/self/exe")<<" 516 2\n0 4\n512 4\nbindings 2\n-1 1 0\n7 2 0 1\n";}
      const char *workspace_path="prepared-workspace-test.txt";
      {std::ofstream manifest(workspace_path);manifest<<"IFWORKSPACE1 123 "<<prepared::ArenaLayout::fingerprint_file("/proc/self/exe")<<" 0\n";}
      prepared::Context context(path,123,workspace_path);
      std::remove(workspace_path);
      void *a=context.acquire(4);context.instruction(7,true);void *b=context.acquire(4);
      if(static_cast<char *>(b)-static_cast<char *>(a)!=512)throw std::runtime_error("arena offsets not followed");
      context.zero(a,4);context.copy(b,a,4);
      auto reject=[&](auto fn){bool rejected=false;try{fn();}catch(const std::runtime_error&){rejected=true;}if(!rejected)throw std::runtime_error("unplanned arena operation accepted");};
      reject([&]{context.acquire(8);});
      context.instruction(7);reject([&]{context.copy(b,a,4);});
      context.release(b);context.instruction(8);reject([&]{context.acquire(4);});
      context.instruction(-1);context.release(a);context.complete();
      std::remove(path);
    }
    prepared::check(aclrtResetDevice(0), "reset");
    prepared::check(aclFinalize(), "finalize");
    std::cout << "prepared invocation, ordered reuse, pinned staging and "
                 "negative contracts passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

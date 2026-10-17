#include "token_executor.h"
namespace pangu { using Decoder = inferfabric::token::Decoder; using V = inferfabric::token::V; }
int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "usage: openpangu_decode LOAD_PLAN_DIR CHECKPOINT NEW_TOKENS "
                 "TOKEN_ID...\n";
    return 2;
  }
  try {
    if (std::getenv("IF_PANGU_TRACE_DIR"))
      throw std::runtime_error(
          "activation host tracing is forbidden by device residency");
    int steps = std::stoi(argv[3]);
    std::vector<int> prompt;
    for (int i = 4; i < argc; ++i)
      prompt.push_back(std::stoi(argv[i]));
    if (steps < 1 || steps > 128 || prompt.size() + steps > 512)
      throw std::runtime_error("token budget");
    auto execution = inferfabric::execution::Plan::read(std::string(argv[1]) +
                                                        "/execution.bin");
    inferfabric::execution::validate_kernels(execution);
#ifndef IF_CACHE_KERNELS
    for(const auto &n:execution.nodes)if(n.op=="cache_quantize_append" || n.op=="packed_attention_metadata")
      throw std::runtime_error("plan requires integer cache kernels; build native/kernels first");
#endif
    if(!execution.physical) throw std::runtime_error("execution requires a physical IFTOKN02 program");
    bool frozen_bundle=std::filesystem::exists(std::filesystem::path(argv[1])/"arena.json");
    for(int rank=0;rank<8;++rank) frozen_bundle=frozen_bundle || std::filesystem::exists(std::filesystem::path(argv[1])/("arena-rank-"+std::to_string(rank)+".txt"));
    if(!frozen_bundle && !std::getenv("INFERFABRIC_PREPARE_ARENA"))
      throw std::runtime_error("unbound device arena: run explicit INFERFABRIC_PREPARE_ARENA=1 hardware preparation, then bind-prepared-arena.py");
    if(frozen_bundle) for(int rank=0;rank<8;++rank)
      if(!std::filesystem::exists(std::filesystem::path(argv[1])/("arena-rank-"+std::to_string(rank)+".txt")))
        throw std::runtime_error("incomplete frozen arena bundle");

    if (prompt.size() + steps > execution.context)
      throw std::runtime_error("execution plan context budget");
    std::vector<inferfabric::initialization::Plan> initialization;
    std::set<uint32_t> devices;
    for (int rank = 0; rank < 8; ++rank) {
      auto plan = inferfabric::initialization::Plan::read(
          std::string(argv[1]) + "/rank-" + std::to_string(rank) + ".bin");
      if (plan.rank != uint32_t(rank) || !devices.insert(plan.device).second)
        throw std::runtime_error("initialization rank/device mismatch");
      inferfabric::execution::validate_bindings(execution, plan);
      initialization.push_back(std::move(plan));
    }
    openpangu::checked(aclInit(nullptr), "init");
    openpangu::checked(aclrtSetDevice(initialization[0].device), "bootstrap");
    HcclRootInfo info;
    openpangu::checked(HcclGetRootInfo(&info), "root");
    inferfabric::profiling::Capture capture(
        std::vector<uint32_t>(devices.begin(), devices.end()));
    std::vector<std::thread> threads;
    for (int rank = 0; rank < 8; ++rank)
      threads.emplace_back([&, rank] {
        try {
          const auto &plan = initialization[rank];
          openpangu::checked(aclrtSetDevice(plan.device), "device");
          {
            size_t memory_free_before=0,memory_total=0;
            openpangu::checked(aclrtGetMemInfo(ACL_HBM_MEM,&memory_free_before,&memory_total),"memory baseline");
            auto arena_path=std::filesystem::path(argv[1]) / ("arena-rank-"+std::to_string(rank)+".txt");
            auto workspace_path=std::filesystem::path(argv[1]) / ("workspace-rank-"+std::to_string(rank)+".txt");
            pangu::Decoder decoder(plan, argv[2], info, execution,
                (std::filesystem::exists(std::filesystem::path(argv[1])/"arena.json") || std::filesystem::exists(arena_path))?arena_path.string():"",
                inferfabric::prepared::ArenaLayout::fingerprint_file(std::string(argv[1])+"/execution.bin"),
                std::filesystem::exists(workspace_path)?workspace_path.string():"");
            pangu::V logits;
            int pos = 0;
            for (int token : prompt) {
              const bool prepare_decode =
                  pos > 0 && pos + 1 == int(prompt.size());
              logits = decoder.token(token, pos++, prepare_decode);
            }
            for (int i = 0; i < steps; ++i) {
              int next = decoder.emit(logits, i);
              if (i > 0 && i <= 2 &&
                  (i == 2 || next == int(execution.eos) || i + 1 == steps) &&
                  !capture.directory.empty())
                capture.rendezvous(false);
              if (next == int(execution.eos) || i + 1 == steps)
                break;
              if (i == 0 && !capture.directory.empty())
                capture.rendezvous(true);
              inferfabric::profiling::state.active =
                  !capture.directory.empty() && i < 2;
              {
                inferfabric::profiling::Scope timing("decode_total");
                logits = decoder.token(next, pos++, true);
              }
              inferfabric::profiling::state.active = false;
            }
            if (!capture.directory.empty())
              inferfabric::profiling::write(capture.directory + "/rank-" +
                                            std::to_string(rank) + ".csv");
            if (const char *dir =
                    std::getenv("INFERFABRIC_PREPARED_AUDIT_DIR")) {
              std::filesystem::create_directories(dir);
              std::ofstream out(std::filesystem::path(dir) /
                                ("rank-" + std::to_string(rank) + ".json"));
              decoder.execution.write(out);
              size_t memory_free_after=0,total_after=0;
              openpangu::checked(aclrtGetMemInfo(ACL_HBM_MEM,&memory_free_after,&total_after),"memory snapshot");
              std::ofstream memory(std::filesystem::path(dir)/("memory-rank-"+std::to_string(rank)+".json"));
              memory << "{\"rank\":" << rank << ",\"device\":" << plan.device
                     << ",\"hbm_total_bytes\":" << total_after
                     << ",\"hbm_free_before_decoder_bytes\":" << memory_free_before
                     << ",\"hbm_free_after_generation_bytes\":" << memory_free_after
                     << ",\"hbm_used_after_generation_bytes\":" << total_after-memory_free_after
                     << ",\"hbm_growth_bytes\":" << int64_t(memory_free_before)-int64_t(memory_free_after) << "}\n";
              if(!memory)throw std::runtime_error("memory audit write");
              if (!out)
                throw std::runtime_error("prepared audit write");
            }
            if (const char *dir =
                    std::getenv("INFERFABRIC_TRANSFER_AUDIT_DIR")) {
              std::filesystem::create_directories(dir);
              std::ofstream audit(std::filesystem::path(dir) /
                                  ("rank-" + std::to_string(rank) + ".json"));
              inferfabric::resident::transfers.write(audit);
              if (!audit)
                throw std::runtime_error("transfer audit write");
            }
          }
          openpangu::checked(aclrtResetDevice(plan.device), "reset");
        } catch (const std::exception &e) {
          std::cerr << "rank=" << rank << " " << e.what() << std::endl;
          std::_Exit(1);
        }
      });
    for (auto &t : threads)
      t.join();
    capture.close();
    openpangu::checked(aclFinalize(), "finalize");
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}

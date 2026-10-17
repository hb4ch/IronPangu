#include "token_executor.h"
#include "../include/program_engine.h"
#include "../include/paged_state_contract.h"
#include <condition_variable>
#include <mutex>
#include <atomic>

namespace {
thread_local std::string last_error;
std::atomic<bool> owned{false};
enum class Command { Reset, Token, Prefill, Batch, BindPages, SelectResident, ResetResident, PlanStateCopy, ExecuteStateCopy };
struct Engine {
  std::string bundle, checkpoint;
  inferfabric::execution::Plan program;
  std::vector<inferfabric::initialization::Plan> loads;
  HcclRootInfo root{};
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::thread> workers;
  size_t ready = 0, finished = 0;
  uint64_t generation = 0;
  Command command = Command::Reset;
  uint32_t input = 0, position = 0, sample = 0, output = 0;
  uint32_t active_resident = 0;
  std::vector<uint32_t> next_positions;
  std::vector<uint32_t> chunk_tokens;
  std::vector<inferfabric::execution::SequenceSlice> batch_members;
  std::vector<uint32_t> batch_sample_rows,batch_outputs;
  uint32_t vocabulary = 0;
  std::string binding_component;
  std::vector<int32_t> binding_pages;
  bool preparation;
  // Opt-in offline diagnostics expose only the sampler's existing request
  // outputs. No activation readback, new kernels or completion boundaries.
  bool output_diagnostics = std::getenv("INFERFABRIC_OUTPUT_DIAGNOSTICS") != nullptr;
  bool stopping = false, acl_initialized = false;
  std::string failure;

  Engine(const char *path, const char *weights, bool prepare) : bundle(path), checkpoint(weights),
      program(inferfabric::execution::Plan::read(bundle + "/execution.bin")), preparation(prepare) {
    if(program.uncaptured_prefill) {
      // CANN 9.1's implicit executor-cache teardown crashed after the full
      // variable-query regression. Our explicit repeatable invocations and
      // decode graphs retain their own lifetime; disable only the vendor cache.
      const char *cache=std::getenv("ACLNN_CACHE_LIMIT");
      if(cache && std::string(cache)!="0")
        throw std::runtime_error("uncaptured prefill requires ACLNN_CACHE_LIMIT=0; nonzero vendor cache is unqualified");
      if(!cache && setenv("ACLNN_CACHE_LIMIT","0",0)!=0)
        throw std::runtime_error("cannot configure uncaptured operator cache policy");
      std::cerr<<"program uncaptured prefill ACLNN_CACHE_LIMIT=0\n";
    }
    inferfabric::execution::validate_kernels(program);
    const char *experimental=std::getenv("INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT");
    if(program.context>512 && (!experimental || std::string(experimental)!="1"))
      throw std::runtime_error("context above 512 is unqualified; explicit INFERFABRIC_EXPERIMENTAL_LONG_CONTEXT=1 required for qualification runs");
    next_positions.resize(program.resident_sequences,0);
    if (!program.physical || program.context < 3)
      throw std::runtime_error("persistent engine requires a physical token program with context >=3");
    if (!std::getenv("INFERFABRIC_DECODE_GRAPH"))
      throw std::runtime_error("persistent token engine requires explicitly enabled graph execution");
    for (const auto &n : program.nodes) if (n.op == "embedding") vocabulary = n.integer("vocab");
    if (!vocabulary) throw std::runtime_error("missing vocabulary contract");
    std::set<uint32_t> devices;
    for (uint32_t rank = 0; rank < 8; ++rank) {
      auto plan = inferfabric::initialization::Plan::read(bundle + "/rank-" + std::to_string(rank) + ".bin");
      if (plan.rank != rank || !devices.insert(plan.device).second)
        throw std::runtime_error("invalid rank/device bindings");
      inferfabric::execution::validate_bindings(program, plan);
      if (!preparation && (!std::filesystem::exists(sidecar("arena", rank)) || !std::filesystem::exists(sidecar("workspace", rank))))
        throw std::runtime_error("persistent engine requires complete frozen arena and workspace plans");
      loads.push_back(std::move(plan));
    }
  }
  std::string sidecar(const char *name, uint32_t rank) const {
    return bundle + "/" + name + "-rank-" + std::to_string(rank) + ".txt";
  }
  void start() {
    openpangu::checked(aclInit(nullptr), "engine ACL init");
    acl_initialized = true;
    openpangu::checked(aclrtSetDevice(loads[0].device), "engine bootstrap");
    openpangu::checked(HcclGetRootInfo(&root), "engine HCCL root");
    for (size_t rank = 0; rank < loads.size(); ++rank)
      workers.emplace_back([this, rank] { worker(rank); });
    std::unique_lock<std::mutex> lock(mutex);
    changed.wait(lock, [&] { return ready == loads.size() || !failure.empty(); });
    if (!failure.empty()) throw std::runtime_error(failure);
  }
  void worker(size_t rank) noexcept {
    try {
      openpangu::checked(aclrtSetDevice(loads[rank].device), "engine rank device");
      {
        size_t free_before = 0, total_before = 0;
        openpangu::checked(aclrtGetMemInfo(ACL_HBM_MEM, &free_before, &total_before), "engine memory baseline");
        Dl_info library{};
        if (!dladdr(reinterpret_cast<void *>(&inferfabric_program_abi_version), &library) || !library.dli_fname)
          throw std::runtime_error("native backend identity unavailable");
        auto present = [&](const char *kind) { auto p = sidecar(kind, rank); return std::filesystem::exists(p) ? p : std::string{}; };
        if(rank==0)std::cerr<<"program preparation rank=0 phase=load_parameters\n";
        inferfabric::token::Decoder decoder(loads[rank], checkpoint, root, program,
            present("arena"), inferfabric::prepared::ArenaLayout::fingerprint_file(bundle + "/execution.bin"),
            present("workspace"), inferfabric::prepared::ArenaLayout::fingerprint_file(library.dli_fname));
        // Same preparation order as the qualified diagnostic client. Warmup
        // constructs all parameter views, graph addresses and output invocations.
        if(rank==0)std::cerr<<"program preparation rank=0 phase=decode\n";
        auto warm = decoder.token(0, 0, false);
        warm = decoder.token(0, 1, false);
        warm = decoder.token(0, 2, true);
        decoder.emit(warm, 0, false);
        warm = {};
        if(program.uncaptured_prefill) {
          std::vector<uint32_t> chunk(program.prefill_tokens,0);
          for(int pass=0;pass<2;++pass) {
            if(rank==0)std::cerr<<"program preparation rank=0 phase=uncaptured_prefill query_tokens="<<chunk.size()<<" pass="<<pass<<'\n';
            decoder.reset_request();
            auto logits=decoder.prefill(chunk,0,false);
            decoder.emit(logits,int(chunk.size())-1,false);
          }
        }
        for(auto count:program.prefill_variants) {
          std::vector<uint32_t> chunk(count,0);
          for(int pass=0;pass<3;++pass) {
            decoder.reset_request();
            auto logits=decoder.prefill(chunk,0,pass==2);
            decoder.emit(logits,int(count)-1,false);
          }
        }
        if(program.uncaptured_prefill) {
          // Instantiate each exact decode batch before sealing. Extend remains
          // uncaptured, including long prompts and maximum-query warmup.
          const auto max_sequences=std::min(program.resident_sequences,program.prefill_tokens);
          for(uint32_t count=1;count<=max_sequences;++count) {
            if(rank==0)std::cerr<<"program preparation rank=0 phase=sequence_decode sequences="<<count<<'\n';
            decoder.reset_request();
            std::vector<uint32_t> tokens(count,0),rows(count);
            std::iota(rows.begin(),rows.end(),0);
            for(uint32_t pass=0;pass<3;++pass) {
              std::vector<inferfabric::execution::SequenceSlice> members;
              for(uint32_t slot=0;slot<count;++slot)members.push_back({slot,pass,1,true});
              auto logits=decoder.sequence_batch(tokens,members,pass==2);
              decoder.emit_rows(logits,rows);
            }
          }
          if(max_sequences>1) {
            // Bound control tables and extend temporaries with all resident
            // rows present, without ever capturing an extend graph.
            decoder.reset_request();
            std::vector<uint32_t> tokens(program.prefill_tokens,0);
            std::vector<inferfabric::execution::SequenceSlice> members;
            for(uint32_t slot=0;slot<max_sequences;++slot)
              members.push_back({slot,0,slot==0?program.prefill_tokens-max_sequences+1:1,false});
            auto logits=decoder.sequence_batch(tokens,members,false);
            decoder.execution.complete();
          }
        }
        decoder.seal_graphs();
        if(rank==0)std::cerr<<"program preparation rank=0 phase=complete\n";
        if (const char *dir = std::getenv("INFERFABRIC_PREPARED_AUDIT_DIR")) {
          std::filesystem::create_directories(dir);
          std::ofstream audit(std::filesystem::path(dir) / ("rank-" + std::to_string(rank) + ".json"));
          decoder.execution.write(audit);
          if (!audit) throw std::runtime_error("preparation audit write");
        }
        decoder.reset_request();
        decoder.release_preparation_pages();
        uint64_t seen = 0;
        {
          std::lock_guard<std::mutex> lock(mutex);
          ++ready;
          changed.notify_all();
        }
        for (;;) {
          Command cmd;
          uint32_t token, pos, want;
          {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [&] { return generation != seen || stopping; });
            if (stopping) break;
            seen = generation; cmd = command; token = input; pos = position; want = sample;
          }
          uint32_t next = 0;
          std::vector<uint32_t> next_batch;
          if (cmd == Command::Reset) decoder.reset_request();
          else if (cmd == Command::SelectResident) decoder.select_resident(token,int(pos)-1);
          else if (cmd == Command::ResetResident) decoder.reset_resident(token);
          else if (cmd == Command::PlanStateCopy) decoder.plan_continuation_copy(token,pos,want);
          else if (cmd == Command::ExecuteStateCopy) decoder.execute_continuation_copy();
          else if (cmd == Command::BindPages && rank == token)
            decoder.bind_cache_pages(binding_component, binding_pages);
          else if (cmd == Command::Token) {
            auto logits = decoder.token(int(token), int(pos), true);
            if (want) next = uint32_t(decoder.emit(logits, int(pos), output_diagnostics));
            else decoder.execution.complete();
          } else if(cmd == Command::Prefill) {
            auto logits=decoder.prefill(chunk_tokens,int(pos),!program.uncaptured_prefill);
            if(want)next=uint32_t(decoder.emit(logits,int(pos+chunk_tokens.size()-1),output_diagnostics));
            else decoder.execution.complete();
          } else if(cmd == Command::Batch) {
            auto logits=decoder.sequence_batch(chunk_tokens,batch_members);
            next_batch=decoder.emit_rows(logits,batch_sample_rows);
          }
          {
            std::lock_guard<std::mutex> lock(mutex);
            if (rank == 0) {output = next;if(cmd==Command::Batch)batch_outputs=std::move(next_batch);}
            ++finished;
            changed.notify_all();
          }
        }
        if (const char *dir = std::getenv("INFERFABRIC_PREPARED_AUDIT_DIR")) {
          std::ofstream audit(std::filesystem::path(dir) / ("rank-" + std::to_string(rank) + ".json"));
          decoder.execution.write(audit);
          size_t free_after = 0, total_after = 0;
          openpangu::checked(aclrtGetMemInfo(ACL_HBM_MEM, &free_after, &total_after), "engine memory final");
          std::ofstream memory(std::filesystem::path(dir) / ("memory-rank-" + std::to_string(rank) + ".json"));
          memory << "{\"rank\":" << rank << ",\"hbm_used_bytes\":" << total_after - free_after
                 << ",\"hbm_growth_bytes\":" << int64_t(free_before)-int64_t(free_after) << "}\n";
          if (!audit || !memory) throw std::runtime_error("engine final audit write");
        }
        if (const char *dir = std::getenv("INFERFABRIC_TRANSFER_AUDIT_DIR")) {
          std::filesystem::create_directories(dir);
          std::ofstream audit(std::filesystem::path(dir) / ("rank-" + std::to_string(rank) + ".json"));
          inferfabric::resident::transfers.write(audit);
          if (!audit) throw std::runtime_error("engine transfer audit write");
        }
      }
      openpangu::checked(aclrtResetDevice(loads[rank].device), "engine rank reset");
    } catch (const std::exception &e) {
      std::lock_guard<std::mutex> lock(mutex);
      if (failure.empty()) failure = "rank " + std::to_string(rank) + ": " + e.what();
      stopping = true;
      changed.notify_all();
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex);
      failure = "native rank failure"; stopping = true; changed.notify_all();
    }
  }
  uint32_t submit(Command cmd, uint32_t token = 0, uint32_t pos = 0, uint32_t want = 0) {
    std::unique_lock<std::mutex> lock(mutex);
    if (stopping || !failure.empty()) throw std::runtime_error("worker group unavailable: " + failure);
    command = cmd; input = token; position = pos; sample = want; finished = 0;
    ++generation; changed.notify_all();
    changed.wait(lock, [&] { return finished == loads.size() || !failure.empty(); });
    if (!failure.empty()) throw std::runtime_error(failure);
    return output;
  }
  ~Engine() {
    { std::lock_guard<std::mutex> lock(mutex); stopping = true; changed.notify_all(); }
    for (auto &w : workers) if (w.joinable()) w.join();
    if (acl_initialized) aclFinalize();
  }
};
template<class F> int call(F f) noexcept {
  try { f(); last_error.clear(); return 0; }
  catch (const std::exception &e) { last_error = e.what(); return -1; }
  catch (...) { last_error = "native program engine failure"; return -1; }
}
Engine &engine(void *p) {
  if (!p) throw std::runtime_error("null engine");
  return *static_cast<Engine *>(p);
}
}
extern "C" uint32_t inferfabric_program_abi_version() { return 1; }
extern "C" const char *inferfabric_program_error() { return last_error.c_str(); }
extern "C" int inferfabric_program_open(const char *bundle, const char *checkpoint, uint32_t preparation, void **out) {
  return call([&] {
    if (!bundle || !checkpoint || !out || preparation > 1) throw std::runtime_error("invalid engine arguments");
    *out = nullptr;
    if (owned.exchange(true)) throw std::runtime_error("only one ACL program group per process");
    try {
      auto e = std::make_unique<Engine>(bundle, checkpoint, preparation != 0);
      e->start(); *out = e.release();
    } catch (...) { owned = false; throw; }
  });
}
extern "C" int inferfabric_program_reset(void *p) {
  return call([&] { auto &e = engine(p); e.submit(Command::Reset);
    std::fill(e.next_positions.begin(),e.next_positions.end(),0); e.active_resident=0; });
}
extern "C" int inferfabric_program_resident_capacity(void *p, uint32_t *count) {
  return call([&] { if(!count)throw std::runtime_error("null resident capacity"); *count=engine(p).program.resident_sequences; });
}
extern "C" int inferfabric_program_select_resident(void *p, uint32_t slot) {
  return call([&] { auto &e=engine(p);
    if(slot>=e.next_positions.size())throw std::runtime_error("resident selection bounds");
    if(slot!=e.active_resident)e.submit(Command::SelectResident,slot,e.next_positions[slot]);
    e.active_resident=slot;
  });
}
extern "C" int inferfabric_program_reset_resident(void *p, uint32_t slot) {
  return call([&] { auto &e=engine(p);
    if(slot>=e.next_positions.size())throw std::runtime_error("resident reset bounds");
    e.submit(Command::ResetResident,slot); e.next_positions[slot]=0;
  });
}
extern "C" int inferfabric_program_copy_resident(void *p,uint32_t source,uint32_t destination,uint32_t boundary) {
  return call([&] { auto &e=engine(p);
    if(source==destination || source>=e.next_positions.size() || destination>=e.next_positions.size() ||
       !boundary || boundary>e.program.context || e.next_positions[source]!=boundary || e.next_positions[destination])
      throw std::runtime_error("state copy requires completed source and reset destination");
    e.submit(Command::PlanStateCopy,source,destination,boundary);
    e.submit(Command::ExecuteStateCopy);
    e.next_positions[destination]=boundary;
  });
}
extern "C" int inferfabric_program_bind_pages(void *p, uint32_t rank, const char *component, const int32_t *pages, uint32_t count) {
  return call([&] {
    auto &e=engine(p);
    if (e.next_positions[e.active_resident] || rank>=e.loads.size() || !component || !pages)
      throw std::runtime_error("invalid page binding arguments");
    const inferfabric::execution::Node *found=nullptr;
    for(const auto &node:e.program.nodes)
      if(node.name==component && inferfabric::execution::is_paged_state(node)) {
        if(found)throw std::runtime_error("ambiguous named cache state");
        found=&node;
      }
    if(!found)throw std::runtime_error("unregistered named cache state");
    inferfabric::execution::paged_state_contract(e.program,*found).validate_count(count,e.program.shared_pages);
    e.binding_component=component;
    e.binding_pages.assign(pages,pages+count);
    e.submit(Command::BindPages,rank);
  });
}
extern "C" int inferfabric_program_info(void *p, uint32_t *context, uint32_t *vocabulary, uint32_t *eos) {
  return call([&] { auto &e = engine(p); if (!context || !vocabulary || !eos) throw std::runtime_error("null info output");
    *context = e.program.context; *vocabulary = e.vocabulary; *eos = e.program.eos; });
}
extern "C" int inferfabric_program_token(void *p, uint32_t token, uint32_t position, uint32_t sample, uint32_t *output) {
  return call([&] {
    auto &e = engine(p);
    if (!output || sample > 1 || position != e.next_positions[e.active_resident] || position >= e.program.context || token >= e.vocabulary)
      throw std::runtime_error("invalid program token/position/output");
    *output = e.submit(Command::Token, token, position, sample); ++e.next_positions[e.active_resident];
  });
}
extern "C" int inferfabric_program_close(void *p) {
  return call([&] { if (!p) throw std::runtime_error("null engine"); delete static_cast<Engine *>(p); owned = false; });
}
extern "C" int inferfabric_program_prefill_capacity(void *p,uint32_t *count) {
  return call([&]{if(!count)throw std::runtime_error("null prefill capacity");*count=engine(p).program.prefill_tokens;});
}
extern "C" int inferfabric_program_prefill_mode(void *p,uint32_t *mode) {
  return call([&]{if(!mode)throw std::runtime_error("null prefill mode");*mode=engine(p).program.uncaptured_prefill?1:0;});
}
extern "C" int inferfabric_program_prefill_variants(void *p,uint32_t *mask) {
  return call([&]{if(!mask)throw std::runtime_error("null prefill variants");uint32_t bits=0;for(auto n:engine(p).program.prefill_variants){if(n>=32)throw std::runtime_error("prefill widths require exact-size ABI");bits|=uint32_t(1)<<n;}*mask=bits;});
}
extern "C" int inferfabric_program_prefill_sizes(void *p,uint32_t *sizes,uint32_t capacity,uint32_t *count) {
  return call([&]{const auto &variants=engine(p).program.prefill_variants;
    if(!count || variants.size()>capacity || (!sizes && !variants.empty()))throw std::runtime_error("prefill sizes output capacity");
    for(size_t i=0;i<variants.size();++i)sizes[i]=variants[i];
    *count=uint32_t(variants.size());
  });
}
extern "C" int inferfabric_program_shared_pages(void *p,uint32_t *enabled) {
  return call([&]{if(!enabled)throw std::runtime_error("null page policy output");*enabled=engine(p).program.shared_pages?1:0;});
}
extern "C" int inferfabric_program_bind_state(void *p,uint32_t rank,uint32_t logical_node,const int32_t *pages,uint32_t count) {
  return call([&]{
    auto &e=engine(p);
    const inferfabric::execution::Node *found=nullptr;
    for(const auto &n:e.program.nodes)
      if(n.logical_node==logical_node && inferfabric::execution::is_paged_state(n)) {
        if(found)throw std::runtime_error("ambiguous cache state binding");
        found=&n;
      }
    if(!found || !e.program.shared_pages)throw std::runtime_error("unregistered shared cache state");
    if(e.next_positions[e.active_resident] || rank>=e.loads.size() || !pages)
      throw std::runtime_error("invalid shared page binding");
    inferfabric::execution::paged_state_contract(e.program,*found).validate_count(count,true);
    e.binding_component=found->name;e.binding_pages.assign(pages,pages+count);
    e.submit(Command::BindPages,rank);
  });
}
extern "C" int inferfabric_program_prefill(void *p,const uint32_t *tokens,uint32_t count,uint32_t position,uint32_t sample,uint32_t *output) {
  return call([&]{
    auto &e=engine(p);
    if(!tokens || !output || sample>1 || count<=1 || !e.program.supports_prefill(count) ||
       position!=e.next_positions[e.active_resident] || position>e.program.context || count>e.program.context-position)
      throw std::runtime_error("invalid program prefill shape/position/output");
    for(uint32_t i=0;i<count;++i)if(tokens[i]>=e.vocabulary)throw std::runtime_error("invalid prefill token");
    e.chunk_tokens.assign(tokens,tokens+count);
    *output=e.submit(Command::Prefill,0,position,sample);
    e.next_positions[e.active_resident]+=count;
  });
}
extern "C" int inferfabric_program_batch(void *p,const IFProgramSequenceV1 *sequences,uint32_t sequence_count,
    const uint32_t *tokens,uint32_t token_count,uint32_t *outputs,uint32_t output_capacity,uint32_t *output_count) {
  return call([&]{
    auto &e=engine(p);
    if(!e.preparation)throw std::runtime_error("multi-sequence arena qualification pending; preparation mode required");
    if(!sequences || !tokens || !output_count || !sequence_count || sequence_count>e.program.resident_sequences ||
       !token_count || token_count>e.program.prefill_tokens || !e.program.uncaptured_prefill)
      throw std::runtime_error("invalid sequence batch arguments");
    std::vector<inferfabric::execution::SequenceSlice> members;
    for(uint32_t i=0;i<sequence_count;++i) {
      const auto &s=sequences[i];
      if(s.sample>1 || s.resident>=e.next_positions.size() || s.position!=e.next_positions[s.resident])
        throw std::runtime_error("invalid sequence continuation");
      members.push_back({s.resident,s.position,s.token_count,s.sample!=0});
    }
    auto batch=inferfabric::execution::SequenceBatch::make(members,e.program.resident_sequences,e.program.prefill_tokens,e.program.context);
    if(batch.positions.size()!=token_count || batch.sample_rows.size()>output_capacity || (!outputs && !batch.sample_rows.empty()))
      throw std::runtime_error("batch token/output extent");
    for(uint32_t i=0;i<token_count;++i)if(tokens[i]>=e.vocabulary)throw std::runtime_error("invalid batch token");
    e.batch_members=std::move(members);e.batch_sample_rows=batch.sample_rows;e.chunk_tokens.assign(tokens,tokens+token_count);
    e.submit(Command::Batch);
    if(e.batch_outputs.size()!=batch.sample_rows.size())throw std::runtime_error("batch sampler output count");
    for(size_t i=0;i<e.batch_outputs.size();++i)outputs[i]=e.batch_outputs[i];
    for(const auto &s:e.batch_members)e.next_positions[s.resident]+=s.tokens;
    *output_count=e.batch_outputs.size();
  });
}
extern "C" int inferfabric_program_batch_capacity(void *p,uint32_t *capacity) {
  return call([&]{if(!capacity)throw std::runtime_error("null batch capacity");auto &e=engine(p);
    *capacity=e.preparation && e.program.uncaptured_prefill?e.program.prefill_tokens:0;});
}

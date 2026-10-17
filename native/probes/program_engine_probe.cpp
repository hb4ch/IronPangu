// Offline client of the same persistent ABI used by the Rust frontend.
#include "program_engine.h"
#include "token_execution_plan.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
static void check(int status) { if (status) throw std::runtime_error(inferfabric_program_error()); }
int main(int argc, char **argv) {
  void *engine = nullptr;
  try {
    if (argc < 4) throw std::runtime_error("usage: program_engine_probe PLAN CHECKPOINT REQUESTS.txt | PLAN CHECKPOINT NEW_TOKENS TOKEN...");
    struct Request { unsigned steps; std::vector<unsigned> prompt; };
    std::vector<Request> requests;
    if (argc == 4) {
      std::ifstream input(argv[3]);
      if (!input) throw std::runtime_error("request fixture open");
      std::string line;
      while (std::getline(input, line)) {
        std::istringstream row(line); Request r{}; unsigned token;
        if (!(row >> r.steps)) throw std::runtime_error("request fixture output count");
        while (row >> token) r.prompt.push_back(token);
        if (!row.eof()) throw std::runtime_error("invalid fixture token");
        requests.push_back(std::move(r));
      }
    } else {
      Request r{unsigned(std::stoul(argv[3])), {}};
      for (int i = 4; i < argc; ++i) r.prompt.push_back(std::stoul(argv[i]));
      requests.assign(3, r);
    }
    if (requests.empty()) throw std::runtime_error("empty fixture");
    struct Binding { uint32_t rank; std::string component; std::vector<int32_t> pages; };
    std::vector<Binding> bindings;
    if (const char *path=std::getenv("INFERFABRIC_PAGE_BINDINGS")) {
      std::ifstream file(path); if(!file)throw std::runtime_error("page bindings open");
      std::string line;
      while(std::getline(file,line)) {
        std::istringstream row(line); Binding b{}; int32_t page;
        if(!(row>>b.rank>>b.component))throw std::runtime_error("page binding header");
        while(row>>page)b.pages.push_back(page);
        if(!row.eof() || b.pages.empty())throw std::runtime_error("page binding entries");
        bindings.push_back(std::move(b));
      }
      if(bindings.empty())throw std::runtime_error("empty page bindings");
    }
    for (const auto &r : requests) if (!r.steps || r.steps > 128 || r.prompt.empty()) throw std::runtime_error("probe request bound");
    check(inferfabric_program_open(argv[1], argv[2], std::getenv("INFERFABRIC_PREPARE_ARENA") ? 1 : 0, &engine));
    if(std::getenv("INFERFABRIC_TEST_BINDING_LIMITS")) {
      if(bindings.empty())throw std::runtime_error("binding-limit test requires manifest-derived bindings");
      const auto plan=inferfabric::execution::Plan::read(std::string(argv[1])+"/execution.bin");
      size_t rejected=0;
      for(const auto &binding:bindings) {
        const inferfabric::execution::Node *node=nullptr;
        for(const auto &candidate:plan.nodes)if(candidate.name==binding.component)node=&candidate;
        if(!node)throw std::runtime_error("binding test component missing from plan");
        auto extra=binding.pages;extra.push_back(0);
        for(auto count:{uint32_t(extra.size()),UINT32_MAX}) {
          if(!inferfabric_program_bind_pages(engine,binding.rank,binding.component.c_str(),extra.data(),count))
            throw std::runtime_error("oversized named binding accepted");
          ++rejected;
          if(plan.shared_pages) {
            if(!inferfabric_program_bind_state(engine,binding.rank,node->logical_node,extra.data(),count))
              throw std::runtime_error("oversized logical binding accepted");
            ++rejected;
          }
        }
        // Failure must not poison the worker group; a valid binding still works.
        check(inferfabric_program_bind_pages(engine,binding.rank,binding.component.c_str(),binding.pages.data(),uint32_t(binding.pages.size())));
      }
      std::cerr<<"binding limits: "<<rejected<<" invalid requests rejected; valid retries passed\n";
    }
    uint32_t chunk=1,variant_mask=0;
    if(std::getenv("INFERFABRIC_PACKED_PREFILL")) {
      check(inferfabric_program_prefill_capacity(engine,&chunk));
      check(inferfabric_program_prefill_variants(engine,&variant_mask));
      if(chunk<=1)throw std::runtime_error("packed prefill requires a compiled chunk variant");
    }
    auto choose_chunk=[&](size_t limit) {
      for(auto n:{16u,8u,4u,2u})if(n<=limit && (variant_mask&(1u<<n)))return n;
      return 1u;
    };
    if (std::getenv("INFERFABRIC_INTERLEAVE_RESIDENTS")) {
      uint32_t residents=0;
      check(inferfabric_program_resident_capacity(engine,&residents));
      if(residents<2)throw std::runtime_error("interleaving requires at least two compiled residents");
      struct Live { size_t request; unsigned position=0, generated=0, next=0; std::chrono::steady_clock::time_point previous; };
      std::vector<Live> live(residents);
      size_t admitted=0, completed=0;
      auto admit=[&](unsigned slot) {
        check(inferfabric_program_reset_resident(engine,slot));
        live[slot]={admitted++,0,0,0,std::chrono::steady_clock::now()};
        check(inferfabric_program_select_resident(engine,slot));
        for(const auto &b:bindings) {
          auto pages=b.pages;
          for(auto &page:pages)page+=slot*pages.size();
          check(inferfabric_program_bind_pages(engine,b.rank,b.component.c_str(),pages.data(),pages.size()));
        }
      };
      for(unsigned slot=0;slot<residents;++slot) {
        live[slot].request=requests.size();
        if(admitted<requests.size())admit(slot);
      }
      while(completed<requests.size())for(unsigned slot=0;slot<residents;++slot) {
        auto &r=live[slot];
        if(r.request>=requests.size())continue;
        const auto &request=requests[r.request];
        check(inferfabric_program_select_resident(engine,slot));
        auto token=r.position<request.prompt.size()?request.prompt[r.position]:r.next;
        auto count=r.position<request.prompt.size() ? choose_chunk(request.prompt.size()-r.position) : 1;
        auto sample=r.position+count>=request.prompt.size();
        if(count>1)check(inferfabric_program_prefill(engine,request.prompt.data()+r.position,count,r.position,sample,&r.next));
        else check(inferfabric_program_token(engine,token,r.position,sample,&r.next));
        r.position+=count;
        if(sample) {
          auto now=std::chrono::steady_clock::now();
          auto us=std::chrono::duration_cast<std::chrono::microseconds>(now-r.previous).count();
          std::cout<<"{\"request\":"<<r.request<<",\"resident\":"<<slot<<",\"step\":"<<r.generated
                   <<",\"token\":"<<r.next<<",\"elapsed_us\":"<<us<<"}"<<std::endl;
          r.previous=now;
          if(++r.generated==request.steps) {
            ++completed;
            r.request=requests.size();
            if(admitted<requests.size())admit(slot);
          }
        }
      }
      check(inferfabric_program_close(engine)); engine=nullptr;
      return 0;
    }
    for (unsigned request = 0; request < requests.size(); ++request) {
      const auto &prompt = requests[request].prompt;
      auto steps = requests[request].steps;
      auto began = std::chrono::steady_clock::now();
      check(inferfabric_program_reset(engine));
      for(const auto &b:bindings) {
        auto pages=b.pages;
        if(std::getenv("INFERFABRIC_ALTERNATE_PAGE_LAYOUTS") && request%2==0)
          for(size_t i=0;i<pages.size();++i)pages[i]=int32_t(i);
        check(inferfabric_program_bind_pages(engine,b.rank,b.component.c_str(),pages.data(),pages.size()));
      }
      unsigned pos = 0, next = 0;
      while(pos<prompt.size()) {
        auto count=choose_chunk(prompt.size()-pos);
        if(count>1)check(inferfabric_program_prefill(engine,prompt.data()+pos,count,pos,pos+count==prompt.size(),&next));
        else check(inferfabric_program_token(engine, prompt[pos], pos, pos + 1 == prompt.size(), &next));
        pos+=count;
      }
      auto previous = began;
      for (unsigned i = 0; i < steps; ++i) {
        if (i) { check(inferfabric_program_token(engine, next, pos, 1, &next)); ++pos; }
        auto now = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(now - previous).count();
        std::cout << "{\"request\":" << request << ",\"step\":" << i << ",\"token\":" << next << ",\"elapsed_us\":" << us << "}" << std::endl;
        previous = now;
      }
    }
    check(inferfabric_program_close(engine)); engine = nullptr;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    if (engine) inferfabric_program_close(engine);
    return 1;
  }
}

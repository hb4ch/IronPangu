// Offline client of the same persistent ABI used by the Rust frontend.
#include "program_engine.h"
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
    for (unsigned request = 0; request < requests.size(); ++request) {
      const auto &prompt = requests[request].prompt;
      auto steps = requests[request].steps;
      auto began = std::chrono::steady_clock::now();
      check(inferfabric_program_reset(engine));
      for(const auto &b:bindings)
        check(inferfabric_program_bind_pages(engine,b.rank,b.component.c_str(),b.pages.data(),b.pages.size()));
      unsigned pos = 0, next = 0;
      for (auto token : prompt) {
        check(inferfabric_program_token(engine, token, pos, pos + 1 == prompt.size(), &next));
        ++pos;
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

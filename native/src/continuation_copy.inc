// Included inside Decoder. Complete continuation state is registered by op,
// never by model/layer names. Control tables are rebuilt from destination ownership.
struct ContinuationCopy {
  void *base;
  uint32_t logical_node;
  const char *field;
  state::StateCopyRange range;
};
std::vector<ContinuationCopy> continuation_copies;
uint32_t copy_destination=0,copy_boundary=0;
void plan_continuation_copy(uint32_t source,uint32_t destination,uint32_t boundary) {
  if(source==destination || source>=program.resident_sequences || destination>=program.resident_sequences ||
      !boundary || boundary>program.context)
    throw std::runtime_error("continuation copy resident/boundary");
  std::vector<ContinuationCopy> planned;
  size_t recurrent_count=0,attention_count=0,index_count=0;
  auto append=[&](const execution::Node &node,const char *field,openpangu::DeviceTensor &storage,
                  const std::vector<state::StateCopyRange> &ranges) {
    for(const auto &range:ranges) {
      execution.validate_state_copy(storage.data,range.source,range.destination,range.bytes);
      planned.push_back({storage.data,node.logical_node,field,range});
    }
  };
  for(const auto &node:program.nodes) {
    if(node.op=="mome" || node.op=="mome_inplace") {
      auto &storage=mome_states.at(node.name)->state;
      if(storage.bytes%program.resident_sequences)throw std::runtime_error("recurrent field partition");
      const uint64_t bytes=storage.bytes/program.resident_sequences;
      append(node,"history",storage,{{source*bytes,destination*bytes,bytes}});
      ++recurrent_count;
    } else if(node.op=="index_cache_append") {
      auto &cache=*index_states.at(node.name);
      append(node,"keys",cache.keys,cache.pages.copy_plan(source,destination,boundary,false,cache.keys.bytes));
      if(cache.scales)
        append(node,"scales",*cache.scales,cache.pages.copy_plan(source,destination,boundary,false,cache.scales->bytes));
      ++index_count;
    } else if(execution::is_paged_state(node)) {
      auto &cache=*attention_states.at(node.name);
      if(cache.scales || (!cache.packed_c8 && cache.cache_bits!=16))
        throw std::runtime_error("unregistered continuation cache layout");
      append(node,"cache",cache.key,cache.page_binding.copy_plan(source,destination,boundary,
          cache.logical_context!=0,cache.key.bytes));
      ++attention_count;
    }
  }
  if(planned.empty() || recurrent_count!=mome_states.size() || attention_count!=attention_states.size() ||
     index_count!=index_states.size())throw std::runtime_error("incomplete continuation state registry");
  // This bounded audit contains the latest plan, not unbounded per-request data.
  // Logical IDs resolve through the compiled plan; no device pointer is exposed.
  if(const char *dir=std::getenv("INFERFABRIC_STATE_COPY_AUDIT_DIR")) {
    std::filesystem::create_directories(dir);
    std::ofstream audit(std::filesystem::path(dir)/("rank-"+std::to_string(rank)+".json"));
    audit<<"{\"format\":\"inferfabric.state-copy.v1\",\"source\":"<<source
         <<",\"destination\":"<<destination<<",\"boundary\":"<<boundary<<",\"operations\":[";
    bool first=true;
    for(const auto &copy:planned) {
      if(!first)audit<<',';
      first=false;
      audit<<"{\"logical_node\":"<<copy.logical_node<<",\"field\":\""<<copy.field
           <<"\",\"primitive\":\"D2D\",\"source_offset\":"<<copy.range.source
           <<",\"destination_offset\":"<<copy.range.destination<<",\"bytes\":"<<copy.range.bytes<<'}';
    }
    audit<<"]}\n";
    if(!audit)throw std::runtime_error("state copy plan audit write");
  }
  continuation_copies=std::move(planned);copy_destination=destination;copy_boundary=boundary;
}
void execute_continuation_copy() {
  if(continuation_copies.empty())throw std::runtime_error("missing continuation copy plan");
  for(const auto &copy:continuation_copies)
    execution.copy_state(copy.base,copy.range.source,copy.range.destination,copy.range.bytes);
  execution.complete(); // owning stream only; the rank group publishes afterward
  if(active_resident==copy_destination)current_position=int(copy_boundary)-1;
  continuation_copies.clear();
}

#include "openpangu_probe_common.h"
#include "../src/openpangu_ops.h"
#include "cache_quantization.h"
#include <numeric>

int main() {
  try {
    check(aclInit(nullptr),"init");check(aclrtSetDevice(0),"device");
    size_t checked_values=0,launches=0;
    {
      aclrtStream stream=nullptr;check(aclrtCreateStream(&stream),"stream");
      for(int rows:{1,2,17,127,4096}) {
        openpangu::DeviceTensor positions({rows},ACL_INT64,8),
            indices({rows,512},ACL_INT32,4),lengths({rows},ACL_INT32,4);
        for(int start:{0,507,65536-rows}) {
          std::vector<int64_t> input(rows);std::iota(input.begin(),input.end(),start);
          positions.upload(input);
          // The production batch entry is called once, regardless of row count.
          if_packed_attention_metadata_batch(positions.data,indices.data,lengths.data,512,rows,stream);
          ++launches;check(aclrtSynchronizeStream(stream),"metadata completion");
          auto actual=indices.read<int32_t>(),lens=lengths.read<int32_t>();
          for(int row=0;row<rows;++row) {
            if(lens[row]!=input[row]+1)throw std::runtime_error("metadata length mismatch rows="+std::to_string(rows)+" start="+std::to_string(start)+" row="+std::to_string(row)+" actual="+std::to_string(lens[row])+" expected="+std::to_string(input[row]+1));
            for(int column=0;column<512;++column) {
              if(actual[row*512+column]!=(column<=input[row]?column:-1))
                throw std::runtime_error("metadata causal index mismatch");
              ++checked_values;
            }
          }
        }
      }
      check(aclrtDestroyStream(stream),"stream destroy");
    }
    check(aclrtResetDevice(0),"reset");check(aclFinalize(),"finalize");
    std::cout<<"{\"status\":\"passed\",\"batch_launches\":"<<launches
             <<",\"checked_indices\":"<<checked_values<<",\"max_query_tokens\":4096}\n";
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}

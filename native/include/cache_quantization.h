#pragma once
#include <cstdint>
extern "C" void if_cache_quantize(void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,void*);
extern "C" void if_cache_dequantize(void*,void*,void*,void*,uint32_t,uint32_t,uint32_t,void*);

extern "C" void if_packed_attention_metadata(void*,void*,void*,uint32_t,void*);
extern "C" void if_packed_attention_metadata_batch(void*,void*,void*,uint32_t,uint32_t,void*);

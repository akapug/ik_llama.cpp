#include "common.cuh"

// Split-K decode attention for head size 512 on Pascal (sm_6x, no tensor cores): one K/V head shared by many Q heads
// (DeepSeek V4: 64 Q heads, K == V, F32 sinks), few Q rows. See fattn-dec-sinks.cu.

// true when the op is a decode shape this kernel takes (it is enabled, the Q rows are few enough and K == V)
bool ggml_cuda_fattn_dec_sinks_wants(const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_dec_sinks(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

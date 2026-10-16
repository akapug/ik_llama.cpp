#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
struct llama_batch;
struct llama_context;
struct ggml_tensor;

// Per-step speculative checkpoints keep one delta row per verified token for each compressor-state ring. The capture
// reads the graph's per-token state rows by batch index, and the restore replays the accepted rows in batch order into
// their ring slots (the last writer of a slot wins), so a verification batch may hold more tokens than a ring has slots.
// Capacity: DSV4_PER_STEP_DEFAULT_MAX_ROWS rows, overridable with the environment variable LLAMA_DSV4_PER_STEP_MAX_ROWS.
// A per-step batch must also fit one ubatch (n_ubatch): capture and restore use ubatch-local row indices.
static constexpr int DSV4_PER_STEP_DEFAULT_MAX_ROWS = 64;
int llama_dsv4_per_step_max_rows();

bool llama_prepare_dsv4_graph_inputs(llama_context & lctx, const llama_batch & batch, bool set_tensors, bool reserve_plan);
void llama_reset_dsv4_state(llama_context * ctx, int32_t seq_id = -1);
bool llama_dsv4_spec_ckpt_prepare(llama_context * ctx, int mode, int max_tokens);
bool llama_dsv4_spec_ckpt_save(llama_context * ctx, bool use_gpu);
bool llama_dsv4_spec_ckpt_capture_rows(llama_context * ctx);
// true while the current batch runs with a per-step speculative checkpoint (the graph then keeps the per-token
// compressor state rows the capture reads)
bool llama_dsv4_per_step_capture_active(const llama_context & ctx);
// true when the graph of a batch with n_rows compressor-state rows must keep its per-token state rows for the capture
bool llama_dsv4_per_step_capture_batch(const llama_context & ctx, size_t n_rows);
// rows of the allocated per-step checkpoint (0 when none is allocated)
int llama_dsv4_per_step_capacity(const llama_context & ctx);
enum llama_spec_ckpt_restore_result llama_dsv4_spec_ckpt_restore(llama_context * ctx, bool use_gpu, int accepted_step);
void llama_dsv4_spec_ckpt_discard(llama_context * ctx);
ggml_tensor * llama_dsv4_spec_ckpt_delta(llama_context * ctx, ggml_tensor * state_tensor);
void llama_dsv4_spec_ckpt_record_plan(llama_context * ctx);

int64_t llama_dsv4_idx_score_chunk(int64_t n_lid, int64_t n_indexer_head, int64_t n_stream);
int64_t llama_dsv4_idx_chunk_nodes(int64_t n_tokens, int64_t n_lid, int64_t n_indexer_head, int64_t n_stream, int64_t n_unfused);

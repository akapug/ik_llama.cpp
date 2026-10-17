// DeepSeek-V4 speculative checkpoint check: greedy decoding without speculation vs speculative decoding with long
// synthetic drafts, token for token, plus the compressor-state rings after every verification step.
//
// The synthetic drafter proposes the next tokens of the no-speculation reference with one token changed at a
// pseudo-random position (or none), so every verification batch has a known, partial acceptance length. A checkpoint
// mode that restores the compressor state wrongly shows up as a ring mismatch at the restored position and usually as a
// changed token stream.
//
// Exactness: the rings must match bit for bit (--tol 0) when the verification batch and the single-token reference use
// the same matmul kernels. On CPU, iqk switches Q8_0/MXFP4 matmuls to a repacked path at >= 32 rows
// (ggml/src/iqk/iqk_mul_mat.cpp, is_dequant_better), so a 33-row batch differs from single-token decoding in the last
// bits whatever the checkpoint mode; use --draft 30 for the exact check and a tolerance for longer drafts. Ring checks
// stop at the first token that differs from the reference (the drafts no longer follow the run after that); the
// reference top-1/top-2 logit margin at that token is printed to tell a numeric near-tie from a broken restore.
// Measured on a 4-layer DeepSeek-V4-Flash cut requantized to Q6_K (all matmul weights; attn_kv via --custom-q): the
// builtin prompts stay bit-exact for 256 tokens with 33-row batches in per-step and gpu-fallback mode. A prompt that
// flipped a token at a 30+ row batch in every checkpoint mode alike (rings still bit-exact) is not in the list.
// The ring oracle (below) is exact at any batch size and kernel path.
//
// usage: test-dsv4-spec-ckpt MODEL [--mode per-step|gpu-fallback|cpu|auto] [--expect-mode MODE|none] [--draft N]
//            [--n-gen N] [--ubatch N] [--threads N] [--first-prompt I] [--n-prompts N] [--tol X] [--bench N] [--dump FILE]
//   MODEL can also come from LLAMACPP_TEST_MODELFILE; without a model the test is skipped (exit code 77).
//   --expect-mode: the checkpoint mode llama_spec_ckpt_init must pick for draft+1 tokens; "none" = it must refuse.
//   --bench N: time N per-step captures and restores at the first partially accepted step with >= 8 accepted drafts.
//   --dump FILE: append the generated tokens of each prompt (one line per prompt) to compare runs of different modes.

#include "llama.h"
#include "common.h"

#include "llama-context.h"
#include "llama-dsv4.h"

#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

static const char * k_prompts[] = {
    "Write a Python function that parses a CSV file of orders (id, customer, amount, date), groups the orders by "
    "customer, and prints the total amount per customer sorted from largest to smallest. Then explain the code.",

    "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox "
    "jumps over the lazy dog. Repeat the sentence above ten more times, numbering each line.",

    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\n"
    "# Rewrite fib above so it runs in linear time, add type hints and a docstring, and write three unit tests.\n",

    "Here is a shopping list: milk, eggs, bread, butter, cheese, apples, bananas, rice, beans, coffee. Write the list "
    "again in alphabetical order, then once more grouped by the aisle of a typical supermarket, with a short note on "
    "each item.",

    "Summarize the plot of a heist movie in which a retired locksmith, a nervous accountant and a talkative pilot try "
    "to steal a painting from a museum during a snowstorm. Use exactly five numbered paragraphs.",
};

static int mode_from_name(const std::string & s) {
    if (s == "per-step")     return LLAMA_SPEC_CKPT_PER_STEP;
    if (s == "gpu-fallback") return LLAMA_SPEC_CKPT_GPU_FALLBACK;
    if (s == "cpu")          return LLAMA_SPEC_CKPT_CPU;
    if (s == "auto")         return LLAMA_SPEC_CKPT_AUTO;
    if (s == "none")         return LLAMA_SPEC_CKPT_NONE;
    fprintf(stderr, "unknown mode %s\n", s.c_str());
    exit(2);
}

static const char * mode_name(int m) {
    switch (m) {
        case LLAMA_SPEC_CKPT_PER_STEP:     return "per-step";
        case LLAMA_SPEC_CKPT_GPU_FALLBACK: return "gpu-fallback";
        case LLAMA_SPEC_CKPT_CPU:          return "cpu";
        case LLAMA_SPEC_CKPT_AUTO:         return "auto";
        default:                           return "none";
    }
}

static llama_token argmax(const float * logits, int n_vocab, float * margin = nullptr) {
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    if (margin != nullptr) {
        float second = -INFINITY;
        for (int i = 0; i < n_vocab; ++i) {
            if (i != best && logits[i] > second) {
                second = logits[i];
            }
        }
        *margin = logits[best] - second;
    }
    return best;
}

// every compressor-state ring of the context, concatenated
static std::vector<float> ring_snapshot(llama_context * ctx) {
    std::vector<float> out;
    const auto add = [&](const std::vector<ggml_tensor *> & tensors) {
        for (ggml_tensor * t : tensors) {
            if (t == nullptr) {
                continue;
            }
            GGML_ASSERT(t->type == GGML_TYPE_F32);
            const size_t off = out.size();
            out.resize(off + (size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, out.data() + off, 0, ggml_nbytes(t));
        }
    };
    const auto & c = ctx->dsv4.cache;
    add(c.csa_state_kv);
    add(c.csa_state_score);
    add(c.hca_state_kv);
    add(c.hca_state_score);
    add(c.lid_state_kv);
    add(c.lid_state_score);
    return out;
}

static std::vector<ggml_tensor *> ring_tensors(llama_context * ctx) {
    std::vector<ggml_tensor *> out;
    const auto & c = ctx->dsv4.cache;
    for (const auto * group : { &c.csa_state_kv, &c.csa_state_score, &c.hca_state_kv, &c.hca_state_score,
                                &c.lid_state_kv, &c.lid_state_score }) {
        for (ggml_tensor * t : *group) {
            if (t != nullptr && std::find(out.begin(), out.end(), t) == out.end()) {
                out.push_back(t);
            }
        }
    }
    return out;
}

static std::vector<float> tensor_data(ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

// Independent oracle of a per-step restore: the saved ring with, for tokens 0..accepted of the verification batch in
// batch order, the token's state row (as the graph computed it) written to slot (pos % ring rows). The rows come from
// the graph's per-token state tensors, not from the checkpoint's delta buffers, so this checks the capture/restore copy
// paths bit for bit at any batch size.
struct ring_oracle {
    std::vector<ggml_tensor *> rings;
    std::vector<std::vector<float>> base;
    std::vector<std::vector<float>> rows; // per ring: n_tokens rows of the verification batch (empty: not captured)

    void save(llama_context * ctx) {
        rings = ring_tensors(ctx);
        base.clear();
        for (ggml_tensor * t : rings) {
            base.push_back(tensor_data(t));
        }
    }

    bool read_rows(llama_context * ctx, int n_tokens) {
        rows.assign(rings.size(), {});
        for (size_t i = 0; i < rings.size(); ++i) {
            const auto it = ctx->dsv4.capture_rows.find(rings[i]);
            if (it == ctx->dsv4.capture_rows.end()) {
                if (ctx->dsv4.persisted_rings.count(rings[i]) != 0) {
                    return false; // a ring the graph writes must have per-token rows
                }
                continue;
            }
            ggml_tensor * r = it->second;
            const size_t row_bytes = (size_t) rings[i]->ne[0] * sizeof(float);
            if (r->type != GGML_TYPE_F32 || r->ne[0] != rings[i]->ne[0] || r->ne[1] < n_tokens || r->nb[0] != sizeof(float)) {
                return false;
            }
            rows[i].resize((size_t) n_tokens * (size_t) rings[i]->ne[0]);
            for (int j = 0; j < n_tokens; ++j) {
                ggml_backend_tensor_get(r, rows[i].data() + (size_t) j * rings[i]->ne[0], (size_t) j * r->nb[1], row_bytes);
            }
        }
        return true;
    }

    // number of rings whose data differs from the oracle after accepting tokens 0..accepted at positions base_pos..
    int check(int base_pos, int accepted) const {
        int n_bad = 0;
        for (size_t i = 0; i < rings.size(); ++i) {
            std::vector<float> want = base[i];
            const int64_t ne0 = rings[i]->ne[0], n_rows = rings[i]->ne[1];
            if (!rows[i].empty()) {
                for (int j = 0; j <= accepted; ++j) {
                    const int64_t slot = (base_pos + j) % n_rows;
                    std::copy_n(rows[i].begin() + (size_t) j * ne0, ne0, want.begin() + (size_t) slot * ne0);
                }
            }
            const std::vector<float> got = tensor_data(rings[i]);
            n_bad += std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) != 0;
        }
        return n_bad;
    }
};

struct ring_diff {
    double max_abs = 0.0;
    double max_excess = 0.0; // max of |a-b| - tol*(1+|b|); > 0 = outside tolerance
    double max_rel = 0.0;    // max of |a-b|/(1+|b|)
};

static ring_diff compare_rings(const std::vector<float> & a, const std::vector<float> & b, double tol) {
    ring_diff d;
    if (a.size() != b.size()) {
        d.max_abs = d.max_excess = INFINITY;
        return d;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = std::fabs((double) a[i] - (double) b[i]);
        d.max_abs = std::max(d.max_abs, e);
        d.max_excess = std::max(d.max_excess, e - tol*(1.0 + std::fabs((double) b[i])));
        d.max_rel = std::max(d.max_rel, e/(1.0 + std::fabs((double) b[i])));
    }
    return d;
}

static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static constexpr int k_mode_unset = -1000;

struct options {
    std::string model;
    int mode = LLAMA_SPEC_CKPT_PER_STEP;
    int expect_mode = k_mode_unset; // unset: the requested mode (auto: per-step)
    int draft = 32;
    int n_gen = 256;
    int ubatch = 512;
    int threads = 8;
    int first_prompt = 0;
    int n_prompts = (int) (sizeof(k_prompts)/sizeof(k_prompts[0]));
    double tol = 1e-3;
    int bench = 0;
    std::string dump;
};

static llama_context * new_context(llama_model * model, const options & o) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = std::max(512, o.draft + 1);
    cp.n_ubatch = o.ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = o.threads;
    cp.n_threads_batch = o.threads;
    cp.flash_attn = true;
    return llama_init_from_model(model, cp);
}

static bool decode_prompt(llama_context * ctx, const std::vector<llama_token> & prompt, llama_batch & batch) {
    common_batch_clear(batch);
    for (size_t i = 0; i < prompt.size(); ++i) {
        common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
    }
    return llama_decode(ctx, batch) == 0;
}

struct run_stats {
    int verify = 0;
    int restores = 0;
    int full_accept = 0;
    int accepted_ge8 = 0;   // restores with >= 8 accepted drafts (past the old 8-row limit)
    int max_restored_step = -1;
    int ring_checks = 0;
    int ring_fail = 0;
    int oracle_checks = 0;
    int oracle_fail = 0;
    double ring_max_abs = 0.0;
    double ring_max_rel = 0.0;
    double bench_capture_us = -1.0;
    double bench_restore_us = -1.0;
};

int main(int argc, char ** argv) {
    options o;
    const char * env_model = getenv("LLAMACPP_TEST_MODELFILE");
    if (env_model != nullptr) {
        o.model = env_model;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if      (a == "--mode")        o.mode = mode_from_name(next());
        else if (a == "--expect-mode") o.expect_mode = mode_from_name(next());
        else if (a == "--draft")       o.draft = std::stoi(next());
        else if (a == "--n-gen")       o.n_gen = std::stoi(next());
        else if (a == "--ubatch")      o.ubatch = std::stoi(next());
        else if (a == "--threads")     o.threads = std::stoi(next());
        else if (a == "--n-prompts")   o.n_prompts = std::stoi(next());
        else if (a == "--first-prompt") o.first_prompt = std::stoi(next());
        else if (a == "--tol")         o.tol = std::stod(next());
        else if (a == "--bench")       o.bench = std::stoi(next());
        else if (a == "--dump")        o.dump = next();
        else if (a[0] != '-')          o.model = a;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (o.model.empty()) {
        fprintf(stderr, "SKIP: no model (pass a DeepSeek-V4 GGUF or set LLAMACPP_TEST_MODELFILE)\n");
        return 77;
    }
    if (o.expect_mode == k_mode_unset) {
        o.expect_mode = o.mode == LLAMA_SPEC_CKPT_AUTO ? LLAMA_SPEC_CKPT_PER_STEP : o.mode;
    }
    const int n_builtin = (int) (sizeof(k_prompts)/sizeof(k_prompts[0]));
    o.first_prompt = std::max(0, std::min(o.first_prompt, n_builtin - 1));
    o.n_prompts = std::min(o.n_prompts, n_builtin - o.first_prompt);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(o.model.c_str(), mp);
    if (model == nullptr) {
        fprintf(stderr, "FAIL: cannot load %s\n", o.model.c_str());
        return 1;
    }
    if (!llama_model_is_deepseek4(model)) {
        fprintf(stderr, "SKIP: %s is not a DeepSeek-V4 model\n", o.model.c_str());
        return 77;
    }
    const int n_vocab = llama_n_vocab(model);
    const int max_tokens = o.draft + 1;

    // 1. the checkpoint mode llama_spec_ckpt_init picks for this draft length and ubatch
    {
        llama_context * ctx = new_context(model, o);
        const int got = llama_spec_ckpt_init(ctx, o.mode, max_tokens);
        llama_spec_ckpt_discard(ctx);
        llama_free(ctx);
        printf("MODE requested=%s max_tokens=%d n_ubatch=%d -> %s (expected %s)\n",
                mode_name(o.mode), max_tokens, o.ubatch, mode_name(got), mode_name(o.expect_mode));
        if (got != o.expect_mode) {
            printf("RESULT FAIL: checkpoint mode %s, expected %s\n", mode_name(got), mode_name(o.expect_mode));
            llama_free_model(model);
            return 1;
        }
        if (got == LLAMA_SPEC_CKPT_NONE) {
            printf("RESULT PASS (checkpoint refused as expected)\n");
            llama_free_model(model);
            return 0;
        }
    }

    llama_batch batch = llama_batch_init(std::max(4096, max_tokens), 0, 1);
    bool all_ok = true;
    run_stats total;

    for (int ip = o.first_prompt; ip < o.first_prompt + o.n_prompts; ++ip) {
        const std::vector<llama_token> prompt = common_tokenize(model, k_prompts[ip], true, false);
        const llama_pos n_prompt = (llama_pos) prompt.size();

        // reference: greedy, one token per decode; ring snapshot after each processed position
        std::vector<llama_token> ref;
        std::vector<float> ref_margin; // top-1 minus top-2 logit of each reference token
        std::map<llama_pos, std::vector<float>> ref_rings;
        {
            llama_context * ctx = new_context(model, o);
            if (!decode_prompt(ctx, prompt, batch)) {
                fprintf(stderr, "FAIL: reference prompt decode\n");
                return 1;
            }
            ref_rings[n_prompt - 1] = ring_snapshot(ctx);
            ref_margin.push_back(0.0f);
            ref.push_back(argmax(llama_get_logits_ith(ctx, -1), n_vocab, &ref_margin.back()));
            while ((int) ref.size() < o.n_gen) {
                const llama_pos pos = n_prompt + (llama_pos) ref.size() - 1;
                common_batch_clear(batch);
                common_batch_add(batch, ref.back(), pos, { 0 }, true);
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "FAIL: reference decode at %d\n", (int) pos);
                    return 1;
                }
                ref_rings[pos] = ring_snapshot(ctx);
                ref_margin.push_back(0.0f);
                ref.push_back(argmax(llama_get_logits_ith(ctx, -1), n_vocab, &ref_margin.back()));
            }
            llama_free(ctx);
        }

        // speculative: drafts from the reference with one changed token
        run_stats st;
        std::vector<llama_token> out;
        {
            llama_context * ctx = new_context(model, o);
            const int got = llama_spec_ckpt_init(ctx, o.mode, max_tokens);
            llama_spec_ckpt_discard(ctx);
            if (got != o.expect_mode) {
                fprintf(stderr, "FAIL: spec context picked %s\n", mode_name(got));
                return 1;
            }
            if (!decode_prompt(ctx, prompt, batch)) {
                fprintf(stderr, "FAIL: spec prompt decode\n");
                return 1;
            }
            out.push_back(argmax(llama_get_logits_ith(ctx, -1), n_vocab));
            uint64_t rng = mix64(0x5eed + (uint64_t) ip);

            while ((int) out.size() < o.n_gen) {
                const int k = (int) out.size();
                const int n_draft = std::min(o.draft, o.n_gen - k);
                std::vector<llama_token> draft(ref.begin() + k, ref.begin() + k + n_draft);
                rng = mix64(rng);
                const int bad = (int) (rng % (uint64_t) (n_draft + 1)); // == n_draft: no changed token
                if (bad < n_draft) {
                    draft[bad] = (draft[bad] + 1 + (llama_token) ((rng >> 32) % 97)) % n_vocab;
                }

                const llama_token sampled = out.back();
                const int mode = llama_spec_ckpt_init(ctx, o.mode, n_draft + 1);
                if (mode != o.expect_mode || !llama_spec_ckpt_save(ctx, 0)) {
                    fprintf(stderr, "FAIL: checkpoint init/save (mode %s)\n", mode_name(mode));
                    return 1;
                }
                const bool use_oracle = mode == LLAMA_SPEC_CKPT_PER_STEP;
                ring_oracle oracle;
                if (use_oracle) {
                    oracle.save(ctx);
                }

                // the last sampled token (out[k-1]) sits at position n_prompt + k - 1, the drafts follow it
                const llama_pos base = n_prompt + k - 1;
                common_batch_clear(batch);
                common_batch_add(batch, sampled, base, { 0 }, true);
                for (int j = 0; j < n_draft; ++j) {
                    common_batch_add(batch, draft[j], base + 1 + j, { 0 }, true);
                }
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "FAIL: verify decode\n");
                    return 1;
                }
                st.verify++;
                const bool oracle_rows = use_oracle && oracle.read_rows(ctx, batch.n_tokens);
                if (use_oracle && !oracle_rows) {
                    if (st.oracle_fail < 5) {
                        printf("  oracle: per-token state rows of the verification graph are missing (prompt %d pos %d)\n",
                                ip, (int) base);
                    }
                    st.oracle_fail++;
                }

                int acc = 0;
                llama_token next = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
                while (acc < n_draft && next == draft[acc]) {
                    ++acc;
                    next = argmax(llama_get_logits_ith(ctx, acc), n_vocab);
                }

                if (acc < n_draft) {
                    if (o.bench > 0 && mode == LLAMA_SPEC_CKPT_PER_STEP && acc >= 8 && st.bench_capture_us < 0) {
                        const auto t0 = std::chrono::steady_clock::now();
                        for (int b = 0; b < o.bench; ++b) {
                            llama_dsv4_spec_ckpt_capture_rows(ctx);
                        }
                        llama_synchronize(ctx);
                        const auto t1 = std::chrono::steady_clock::now();
                        for (int b = 0; b < o.bench; ++b) {
                            llama_dsv4_spec_ckpt_restore(ctx, true, acc);
                        }
                        llama_synchronize(ctx);
                        const auto t2 = std::chrono::steady_clock::now();
                        st.bench_capture_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / o.bench;
                        st.bench_restore_us = std::chrono::duration<double, std::micro>(t2 - t1).count() / o.bench;
                    }
                    const auto res = llama_spec_ckpt_restore_ex(ctx, 0, base, acc);
                    if (res == LLAMA_SPEC_CKPT_RESTORE_FAILED) {
                        fprintf(stderr, "FAIL: restore at step %d\n", acc);
                        return 1;
                    }
                    if (res == LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED) {
                        common_batch_clear(batch);
                        common_batch_add(batch, sampled, base, { 0 }, acc == 0);
                        for (int j = 0; j < acc; ++j) {
                            common_batch_add(batch, draft[j], base + 1 + j, { 0 }, j == acc - 1);
                        }
                        if (llama_decode(ctx, batch) != 0) {
                            fprintf(stderr, "FAIL: replay decode\n");
                            return 1;
                        }
                    }
                    st.restores++;
                    st.accepted_ge8 += acc >= 8;
                    st.max_restored_step = std::max(st.max_restored_step, acc);
                } else {
                    st.full_accept++;
                }
                llama_spec_ckpt_discard(ctx);

                if (oracle_rows) {
                    const int n_bad = oracle.check(base, acc);
                    st.oracle_checks++;
                    if (n_bad != 0) {
                        if (st.oracle_fail < 5) {
                            printf("  oracle mismatch prompt %d pos %d (accepted %d of %d): %d of %zu rings differ\n",
                                    ip, (int) base, acc, n_draft, n_bad, oracle.rings.size());
                        }
                        st.oracle_fail++;
                    }
                }

                for (int j = 0; j < acc; ++j) {
                    out.push_back(draft[j]);
                }
                out.push_back(next);
                const bool on_ref = std::equal(out.begin(), out.begin() + std::min(out.size(), ref.size()), ref.begin());

                // while the run follows the reference, the rings must equal it after the last accepted position
                const llama_pos last = base + acc;
                const auto it = ref_rings.find(last);
                if (on_ref && it != ref_rings.end()) {
                    const ring_diff d = compare_rings(ring_snapshot(ctx), it->second, o.tol);
                    st.ring_checks++;
                    st.ring_max_abs = std::max(st.ring_max_abs, d.max_abs);
                    st.ring_max_rel = std::max(st.ring_max_rel, d.max_rel);
                    if (d.max_excess > 0.0) {
                        if (st.ring_fail < 5) {
                            printf("  ring mismatch prompt %d pos %d (accepted %d of %d): max |diff| %.3g\n",
                                    ip, (int) last, acc, n_draft, d.max_abs);
                        }
                        st.ring_fail++;
                    }
                }
            }
            llama_free(ctx);
        }
        out.resize(o.n_gen);

        int first_diff = -1;
        for (int i = 0; i < o.n_gen; ++i) {
            if (out[i] != ref[i]) {
                first_diff = i;
                break;
            }
        }
        const bool ok = first_diff < 0 && st.ring_fail == 0 && st.oracle_fail == 0;
        all_ok = all_ok && ok;
        std::string diff_note;
        if (first_diff >= 0) {
            char buf[96];
            snprintf(buf, sizeof(buf), " (reference top-1/top-2 margin there %.4g)", ref_margin[first_diff]);
            diff_note = std::to_string(first_diff) + buf;
        }
        if (!o.dump.empty()) {
            FILE * f = fopen(o.dump.c_str(), "a");
            if (f != nullptr) {
                fprintf(f, "%d", ip);
                for (llama_token t : out) {
                    fprintf(f, " %d", t);
                }
                fprintf(f, "\n");
                fclose(f);
            }
        }
        printf("prompt %d: n_prompt=%d n_gen=%d tokens %s%s | verify=%d restores=%d (>=8 accepted: %d, max step %d) "
               "full=%d | rings checked=%d fail=%d max|diff|=%.3g rel=%.3g | oracle checked=%d fail=%d",
                ip, (int) n_prompt, o.n_gen, first_diff < 0 ? "MATCH" : "DIFF at ",
                diff_note.c_str(),
                st.verify, st.restores, st.accepted_ge8, st.max_restored_step, st.full_accept,
                st.ring_checks, st.ring_fail, st.ring_max_abs, st.ring_max_rel, st.oracle_checks, st.oracle_fail);
        if (st.bench_capture_us >= 0) {
            printf(" | bench capture=%.1f us restore=%.1f us", st.bench_capture_us, st.bench_restore_us);
        }
        printf("\n");
        fflush(stdout);

        total.verify += st.verify;
        total.restores += st.restores;
        total.accepted_ge8 += st.accepted_ge8;
        total.full_accept += st.full_accept;
        total.ring_checks += st.ring_checks;
        total.ring_fail += st.ring_fail;
        total.oracle_checks += st.oracle_checks;
        total.oracle_fail += st.oracle_fail;
        total.ring_max_abs = std::max(total.ring_max_abs, st.ring_max_abs);
        total.ring_max_rel = std::max(total.ring_max_rel, st.ring_max_rel);
        total.max_restored_step = std::max(total.max_restored_step, st.max_restored_step);
    }

    llama_batch_free(batch);
    llama_free_model(model);
    llama_backend_free();

    printf("TOTAL mode=%s max_tokens=%d verify=%d restores=%d (>=8 accepted: %d, max step %d) full=%d "
           "rings checked=%d fail=%d max|diff|=%.3g rel=%.3g tol=%g oracle checked=%d fail=%d\n",
            mode_name(o.expect_mode), max_tokens, total.verify, total.restores, total.accepted_ge8,
            total.max_restored_step, total.full_accept, total.ring_checks, total.ring_fail, total.ring_max_abs,
            total.ring_max_rel, o.tol, total.oracle_checks, total.oracle_fail);
    printf("RESULT %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

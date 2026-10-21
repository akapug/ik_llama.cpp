// Checks LLAMA_EXPERT_PLACEMENT page binding on fake expert tensors and on dense tensors named by a regex or a name with
// a group count (no model file needed).
// Needs a NUMA node other than node 0. Not in CMake; build against build-cuda:
//   g++ -std=c++17 -Iinclude -Isrc -Iggml/include -Iggml/src tests/test-expert-place.cpp -o test-expert-place
//       -Lbuild-cuda/src -Lbuild-cuda/ggml/src -lllama -lggml -Wl,-rpath,$PWD/build-cuda/src:$PWD/build-cuda/ggml/src
//   ./test-expert-place <second-node> [prebind]
// prebind: bind each tensor with llama_expert_numa_prebind before the first touch (the deferred --no-mmap loader), then
// run llama_expert_numa_apply after it, which must find every page already placed
#include "llama-expert-place.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

static int node_of(const void * p) {
    void * pages[1] = { (void *) p };
    int status[1] = { -1 };
    if (syscall(SYS_move_pages, 0, 1, pages, nullptr, status, 0) != 0) return -1000;
    return status[0];
}

int main(int argc, char ** argv) {
    const int pmem = argc > 1 ? atoi(argv[1]) : 2;
    const size_t page = sysconf(_SC_PAGESIZE);

    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    // up: 32 KiB per expert (page multiple); down: 12000 B per expert (straddles pages)
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 8, 16);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1000, 3, 16);
    ggml_set_name(up, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");
    // dense: q_b 64 rows of 4 KiB (rows:0,N by regex -> rows 0-31 on node 0, 32-63 on N); wo_a 4 groups of 8 rows
    // ("/4" -> each group's rows 0-3 on node 0, 4-7 on N); norm is matched by no rule and stays where it falls
    ggml_tensor * qb   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 64);
    ggml_tensor * woa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 32);
    ggml_tensor * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_set_name(qb, "blk.0.attn_q_b.weight");
    ggml_set_name(woa, "blk.0.attn_output_a.weight");
    ggml_set_name(norm, "blk.0.attn_norm.weight");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    llama_model model;
    model.layers.resize(1);
    model.layers[0].ffn_up_exps = up;
    model.layers[0].ffn_down_exps = down;
    for (ggml_tensor * t : { up, down, qb, woa, norm }) model.tensors_by_name.emplace_back(ggml_get_name(t), t);

    const std::set<int> cold = { 1, 3, 5, 6, 7, 15 };
    const std::string path = "/tmp/test-expert-place.tsv";
    {
        std::ofstream f(path);
        f << "# test\nblk.0\t*\t0\nblk.0\t1,3,5-7,15\t" << pmem << "\n"
          << "re:attn_q_b\t*\trows:0," << pmem << "\nblk.0.attn_output_a.weight\t/4\trows:0," << pmem << "\n";
    }
    setenv("LLAMA_EXPERT_PLACEMENT", path.c_str(), 1);
    setenv("LLAMA_EXPERT_PLACEMENT_VERIFY", "1", 1);
    const bool prebind = argc > 2 && !strcmp(argv[2], "prebind");
    if (prebind) {
        for (ggml_tensor * t : { up, down, qb, woa, norm }) llama_expert_numa_prebind(model, t);
    } else {
        llama_expert_numa_apply(model, false);
    }
    memset(up->data, 1, ggml_nbytes(up));       // first touch after the policy, as the loader does
    memset(down->data, 1, ggml_nbytes(down));
    memset(qb->data, 1, ggml_nbytes(qb));
    memset(woa->data, 1, ggml_nbytes(woa));
    if (!prebind) llama_expert_numa_report(model);   // prebind: the checks below run before any apply could migrate a page

    int bad = 0, checked = 0;
    for (ggml_tensor * t : { up, down }) {
        const uintptr_t base = (uintptr_t) t->data, end = base + ggml_nbytes(t);
        for (int e = 0; e < (int) t->ne[2]; ++e) {
            // pages fully inside expert e must sit on its node
            uintptr_t a = (base + e * t->nb[2] + page - 1) / page * page;
            uintptr_t b = std::min<uintptr_t>(base + (e + 1) * t->nb[2], end) / page * page;
            for (uintptr_t p = a; p < b; p += page) {
                const int want = cold.count(e) ? pmem : 0;
                const int got = node_of((void *) p);
                ++checked;
                if (got != want) {
                    if (bad < 10) fprintf(stderr, "%s expert %d page %#lx: node %d, want %d\n", t->name, e, (unsigned long) p, got, want);
                    ++bad;
                }
            }
        }
    }
    // dense: one row = 4 KiB, so the page holding a row's middle is mostly that row's and takes its slice's node
    for (auto [t, groups] : { std::pair<ggml_tensor *, int>{ qb, 1 }, { woa, 4 } }) {
        const int rows = (int) t->ne[1] / groups;
        for (int g = 0; g < groups; ++g) {
            for (int r = 0; r < rows; ++r) {
                const char * mid = (const char *) t->data + ((size_t) (g * rows + r) * t->nb[1]) + t->nb[1] / 2;
                const uintptr_t pg = (uintptr_t) mid / page * page, base = (uintptr_t) t->data;
                if (pg < base || pg + page > base + ggml_nbytes(t)) continue;   // edge pages are left to the process policy
                const int want = r < rows / 2 ? 0 : pmem;
                const int got = node_of(mid);
                ++checked;
                if (got != want) {
                    if (bad < 10) fprintf(stderr, "%s group %d row %d: node %d, want %d\n", t->name, g, r, got, want);
                    ++bad;
                }
            }
        }
    }
    if (prebind) {   // then the after-load apply, as the deferred loader does: it must find the pages placed
        llama_expert_numa_apply(model, false);
        llama_expert_numa_report(model);
    }
    printf("checked %d whole pages, %d misplaced%s\n", checked, bad, prebind ? " before the after-load apply" : "");
    model.layers[0].ffn_up_exps = model.layers[0].ffn_down_exps = nullptr;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return bad ? 1 : 0;
}

// Checks LLAMA_EXPERT_PLACEMENT page binding on fake expert tensors (no model file needed).
// Needs a NUMA node other than node 0. Not in CMake; build against build-cuda:
//   g++ -std=c++17 -Iinclude -Isrc -Iggml/include -Iggml/src tests/test-expert-place.cpp -o test-expert-place
//       -Lbuild-cuda/src -Lbuild-cuda/ggml/src -lllama -lggml -Wl,-rpath,$PWD/build-cuda/src:$PWD/build-cuda/ggml/src
//   ./test-expert-place <second-node>
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

    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    // up: 32 KiB per expert (page multiple); down: 12000 B per expert (straddles pages)
    ggml_tensor * up   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 8, 16);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1000, 3, 16);
    ggml_set_name(up, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    llama_model model;
    model.layers.resize(1);
    model.layers[0].ffn_up_exps = up;
    model.layers[0].ffn_down_exps = down;

    const std::set<int> cold = { 1, 3, 5, 6, 7, 15 };
    const std::string path = "/tmp/test-expert-place.tsv";
    {
        std::ofstream f(path);
        f << "# test\nblk.0\t*\t0\nblk.0\t1,3,5-7,15\t" << pmem << "\n";
    }
    setenv("LLAMA_EXPERT_PLACEMENT", path.c_str(), 1);
    setenv("LLAMA_EXPERT_PLACEMENT_VERIFY", "1", 1);
    llama_expert_numa_apply(model, false);
    memset(up->data, 1, ggml_nbytes(up));       // first touch after the policy, as the loader does
    memset(down->data, 1, ggml_nbytes(down));
    llama_expert_numa_report(model);

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
    printf("checked %d whole pages, %d misplaced\n", checked, bad);
    model.layers[0].ffn_up_exps = model.layers[0].ffn_down_exps = nullptr;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return bad ? 1 : 0;
}

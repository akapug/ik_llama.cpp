// LLAMA_NO_HOST_WEIGHTS: CPU-side model weights take the plain CPU buffer type (never pinned) instead of the GPU host type.
// The switch is read once per process, so each mode is its own run:
//   test-no-host-weights 0            -> weights use the default host type (CUDA_Host in a CUDA build with a device)
//   LLAMA_NO_HOST_WEIGHTS=1 test-no-host-weights 1   -> weights use CPU; the default host type is unchanged for other buffers
// Not in CMake; built beside test-expert-place (build-nohost.sh).
#include "llama-impl.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char ** argv) {
    const bool want_cpu = argc > 1 && atoi(argv[1]) != 0;
    const char * weights = ggml_backend_buft_name(llama_weight_buffer_type_cpu());
    const char * host    = ggml_backend_buft_name(llama_default_buffer_type_cpu(true));
    const char * cpu     = ggml_backend_buft_name(ggml_backend_cpu_buffer_type());
    const bool ok = llama_no_host_weights() == want_cpu && strcmp(weights, want_cpu ? cpu : host) == 0;
    printf("no_host %d: weights %s, host buffers %s, cpu %s -> %s\n", (int) llama_no_host_weights(), weights, host, cpu, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

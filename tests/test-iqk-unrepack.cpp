// Checks iqk_unrepack: the byte-exact inverse of the run-time repack (-rtr) for the expert types that GPU
// hot-expert banks copy from repacked host tensors (MXFP4_R8, Q8_0_R8, Q2_K_R4, Q3_K_R4, Q4_K_R4).
//
// For each type: a 3D expert tensor of random bytes is repacked in place with iqk_repack_tensor (the -rtr path),
// then every expert slice is un-repacked on its own (as the bank fill does) and must equal the original bytes.
// Random bytes reach every bit of every field (scales, mins, high bits), so a dropped or misplaced bit shows up.
// A second pass checks that the dequantized values of the un-repacked rows equal those of the original rows.

#include "ggml.h"
#include "iqk/iqk_quantize.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char * what, ggml_type t) {
    if (!ok) {
        ++g_failures;
        fprintf(stderr, "FAIL %s: %s\n", ggml_type_name(t), what);
    }
}

static void run(ggml_type base, ggml_type repacked, int64_t ne0, int64_t ne1, int64_t ne2, std::mt19937 & rng) {
    ggml_init_params ip = { ggml_tensor_overhead() + ggml_row_size(base, ne0)*ne1*ne2 + 256, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_3d(ctx, base, ne0, ne1, ne2);
    ggml_set_name(t, "blk.0.ffn_up_exps.weight");
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> orig(nbytes);
    for (auto & b : orig) b = (uint8_t) rng();
    memcpy(t->data, orig.data(), nbytes);

    iqk_repack_tensor(t);
    check(t->type == repacked, "iqk_repack_tensor did not produce the expected repacked type", base);
    check(memcmp(t->data, orig.data(), nbytes) != 0, "repack left the bytes unchanged (test would prove nothing)", base);
    check(iqk_unrepacked_type(t->type) == (int) base, "iqk_unrepacked_type", base);
    check(t->nb[2] == ggml_row_size(base, ne0)*ne1, "repacked expert stride differs from the base layout", base);

    std::vector<uint8_t> back(nbytes, 0xA5);
    for (int64_t e = 0; e < ne2; ++e) {
        const size_t off = e*t->nb[2];
        check(iqk_unrepack(t->type, ne1, ne0, (const char *) t->data + off, back.data() + off), "iqk_unrepack returned false", base);
    }
    size_t first_bad = nbytes;
    for (size_t i = 0; i < nbytes && first_bad == nbytes; ++i) if (back[i] != orig[i]) first_bad = i;
    if (first_bad != nbytes) {
        fprintf(stderr, "FAIL %s: byte %zu differs (orig %02x, unrepacked %02x)\n", ggml_type_name(base), first_bad,
                orig[first_bad], back[first_bad]);
        ++g_failures;
    }

    // the dequantized values must match too (also covers the row order across the interleave)
    const auto tr = ggml_internal_get_type_traits(base);
    if (tr.to_float) {
        const size_t rs = ggml_row_size(base, ne0);
        std::vector<float> a(ne0), b(ne0);
        int nbad = 0;
        for (int64_t r = 0; r < ne1*ne2; ++r) {
            tr.to_float(orig.data() + r*rs, a.data(), ne0);
            tr.to_float(back.data() + r*rs, b.data(), ne0);
            for (int64_t i = 0; i < ne0; ++i) {
                if (!(a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i])))) ++nbad;
            }
        }
        check(nbad == 0, "dequantized values differ", base);
    }

    // a row count that is not a multiple of the interleave is refused
    check(!iqk_unrepack(t->type, ne1 - 1, ne0, t->data, back.data()), "accepted a partial row group", base);
    printf("%-8s -> %-8s ne=[%lld,%lld,%lld] %zu bytes: %s\n", ggml_type_name(base), ggml_type_name(repacked),
            (long long) ne0, (long long) ne1, (long long) ne2, nbytes, first_bad == nbytes ? "ok" : "MISMATCH");
    ggml_free(ctx);
}

int main() {
    std::mt19937 rng(20261017);
    struct { ggml_type base, repacked; } cases[] = {
        { GGML_TYPE_MXFP4, GGML_TYPE_MXFP4_R8 },
        { GGML_TYPE_Q8_0,  GGML_TYPE_Q8_0_R8  },
        { GGML_TYPE_Q2_K,  GGML_TYPE_Q2_K_R4  },
        { GGML_TYPE_Q3_K,  GGML_TYPE_Q3_K_R4  },
        { GGML_TYPE_Q4_K,  GGML_TYPE_Q4_K_R4  },
    };
    for (const auto & c : cases) {
        run(c.base, c.repacked, 512, 16, 3, rng);
        run(c.base, c.repacked, 1024, 8, 5, rng);
    }
    // types with no inverse are refused and report themselves
    uint8_t dummy[4096] = {};
    if (iqk_unrepack(GGML_TYPE_IQ4_K_R4, 4, 256, dummy, dummy + 2048) || iqk_unrepacked_type(GGML_TYPE_IQ4_K_R4) != GGML_TYPE_IQ4_K_R4 ||
            iqk_unrepacked_type(GGML_TYPE_MXFP4) != GGML_TYPE_MXFP4) {
        fprintf(stderr, "FAIL: a type without an inverse was accepted\n");
        ++g_failures;
    }
    if (g_failures) {
        fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    printf("all unrepack round trips are byte-exact\n");
    return 0;
}

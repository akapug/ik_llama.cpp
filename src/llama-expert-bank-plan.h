#pragma once

#include "llama-expert-place.h"
#include "ggml.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

// A pre-partitioned artifact contract, not automatic model-loader synthesis.
// Callers supply the current backend allocator and device descriptor limits.
struct llama_expert_bank_limits {
    size_t allocation;
    size_t descriptor;
    size_t alignment;
};
struct llama_expert_bank_span {
    int64_t first;
    int64_t count;
};

inline std::vector<llama_expert_bank_span> llama_expert_bank_plan(
        const std::vector<const ggml_tensor *> & tensors, int64_t experts,
        llama_expert_bank_limits limits, size_t occupied_tags = 0) {
    if (tensors.empty() || experts <= 0 || !limits.allocation || !limits.descriptor ||
            !limits.alignment || occupied_tags >= LLAMA_EXPERT_PART_MAX) {
        throw std::runtime_error("expert bank plan: invalid geometry, limits or artifact tag budget");
    }
    const auto checked_add = [](size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max() - a) {
            throw std::runtime_error("expert bank plan: byte arithmetic overflow");
        }
        return a + b;
    };
    size_t capacity = size_t(experts);
    for (const auto * t : tensors) {
        if (!t || t->ne[2] != experts || t->ne[3] != 1 || t->ne[0] <= 0 || t->ne[1] <= 0 ||
                t->ne[0] % ggml_blck_size(t->type) ||
                t->nb[1] != ggml_row_size(t->type, t->ne[0])) {
            throw std::runtime_error("expert bank plan: expected ordinary packed expert rows");
        }
        const size_t row = ggml_row_size(t->type, t->ne[0]);
        if (size_t(t->ne[1]) > std::numeric_limits<size_t>::max() / row) {
            throw std::runtime_error("expert bank plan: expert extent overflow");
        }
        const size_t plane = row * size_t(t->ne[1]);
        if (t->nb[2] < plane) {
            throw std::runtime_error("expert bank plan: overlapping expert planes");
        }
        // The actual GGML descriptor span includes holes between expert planes;
        // allocator padding must also fit. No power-of-two alignment assumption.
        const size_t allocation = limits.allocation - limits.allocation % limits.alignment;
        const size_t usable = std::min(allocation, limits.descriptor);
        if (plane > usable) {
            throw std::runtime_error("expert bank plan: one expert cannot fit both limits");
        }
        capacity = std::min(capacity, checked_add((usable - plane) / t->nb[2], 1));
    }
    const size_t count = 1 + (size_t(experts) - 1) / capacity;
    if (count > LLAMA_EXPERT_PART_MAX - occupied_tags) {
        throw std::runtime_error("expert bank plan: required banks exceed remaining artifact tags");
    }
    std::vector<llama_expert_bank_span> plan;
    for (int64_t first = 0; first < experts;) {
        const int64_t n = std::min<int64_t>(capacity, experts - first);
        plan.push_back({first, n});
        first += n;
    }
    return plan;
}

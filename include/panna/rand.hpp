#pragma once

#include <random>

namespace panna {
    static std::mt19937_64 _GLOBAL_RNG;

    static std::mt19937_64 get_global_rng() {
        return _GLOBAL_RNG;
    }

    static void seed_global_rng(const uint64_t seed) {
        _GLOBAL_RNG.seed(seed);
    }
}

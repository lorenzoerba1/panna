#pragma once

#include <random>

namespace panna {

    static std::mt19937_64& get_global_rng() {
        static std::mt19937_64 rng;
        return rng;
    }

    static void seed_global_rng( const uint64_t seed ) {
        auto& rng = get_global_rng();
        rng.seed( seed );
    }

    static float sample_random_normal() {
        static std::normal_distribution<float> normal(0, 1);
        return normal(get_global_rng());
    }
} // namespace panna

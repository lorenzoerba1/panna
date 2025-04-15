#pragma once

#include <catch2/catch_test_macros.hpp>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/rand.hpp"
#include "panna/trieindex.hpp"

namespace panna {
    TEST_CASE( "angular distance trie index" ) {
        using Hasher = Simhash<24, UnitNormPoints, CosineDistance>;

        const int NUM_SAMPLES = 100;
        const size_t dimensions = 100;
        const size_t n = 10000;
        const size_t repetitions = 128;

        seed_global_rng(1344);

        std::vector<float> deltas = { 0.5, 0.8, 0.95 };
        std::vector<unsigned int> ks = { 1, 10 };

        SimhashBuilder<24, UnitNormPoints, CosineDistance> builder( dimensions );
        Index<UnitNormPoints, Hasher, CosineDistance> index( dimensions, builder, repetitions );
        for ( size_t i = 0; i < n; i++ ) {
            std::vector<float> point = sample_random_normal_vector(dimensions);
            index.insert(point);
        }
        index.rebuild();

        for ( auto k : ks ) {
            for ( auto delta : deltas ) {
                int num_correct = 0;

                float expected_correct = ( 1 - delta ) * k * NUM_SAMPLES;
                for ( int sample = 0; sample < NUM_SAMPLES; sample++ ) {
                    std::vector<float> query = sample_random_normal_vector(dimensions);
                    std::vector<std::pair<float, uint32_t>> output_exact;
                    std::vector<std::pair<float, uint32_t>> output_approx;
                    index.search_brute_force( query, k, output_exact );
                    index.search( query, k, delta, output_approx );

                    REQUIRE( output_approx.size() == k );
                    for ( auto pair_exact : output_exact ) {
                        // Each expected value is returned once.
                        for (auto pair_approx : output_approx) {
                            if (pair_exact.second == pair_approx.second) {
                                num_correct++;
                                break;
                            }
                        }
                    }
                }
                dbg(num_correct, expected_correct);
                // Only fail if the recall is far away from the expectation.
                REQUIRE( num_correct >= 0.9 * expected_correct );
            }
        }
    }
} // namespace panna

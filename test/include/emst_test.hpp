#pragma once

#include <catch2/catch_test_macros.hpp>
#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/emst.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/rand.hpp"

namespace panna {

    TEST_CASE( "EMST" ) {
        using Dataset = EuclideanPoints;
        using Distance = EuclideanDistance;
        using Hasher = E2LSH<12, Dataset, Distance>;
        const size_t dimensions = 10;

        const size_t n = 10000;
        std::vector<std::vector<float>> data;
        for (size_t i=0; i<n; i++) {
            data.push_back(sample_random_normal_vector(dimensions));
        }

        Hasher::Builder builder(2.0, dimensions);

        EMST<Dataset, Hasher, Distance> emst(dimensions, 200, builder, data, 0.01, 0.0);

        auto exact = emst.exact_tree();
        auto exact_with_fp = emst.find_tree();

        REQUIRE(exact.first == exact_with_fp.first);
    }

  
}

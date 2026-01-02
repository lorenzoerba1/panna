#pragma once

#include <catch2/catch_test_macros.hpp>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/emst.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/rand.hpp"

namespace panna {

    TEST_CASE( "EMST e2lsh" ) {
        using Dataset = EuclideanPoints;
        using Distance = EuclideanDistance;
        using Hasher = E2LSH<12, Dataset, Distance>;
        const size_t dimensions = 10;

        const std::vector<size_t> sizes = { 100, 1000, 10000 };
        for ( size_t n : sizes ) {
            std::vector<std::vector<float>> data;
            for ( size_t i = 0; i < n; i++ ) {
                data.push_back( sample_random_normal_vector( dimensions ) );
            }

            EMST<Dataset, Hasher, Distance> emst( dimensions, 200, data, 0.001, 0.0 );

            auto exact = emst.exact_tree();
            auto exact_with_fp = emst.find_tree();

            REQUIRE( exact.first == exact_with_fp.first );
        }
    }

    TEST_CASE( "EMST LatticeLSH" ) {
        using Dataset = EuclideanPoints;
        using Distance = EuclideanDistance;
        using Hasher = LatticeLSH<4, Dataset, Distance>;
        const size_t dimensions = 8;

        const std::vector<size_t> sizes = { 10000 };
        for ( size_t n : sizes ) {
            std::vector<std::vector<float>> data;
            EuclideanPoints pts( dimensions );
            for ( size_t i = 0; i < n; i++ ) {
                auto v = sample_random_normal_vector( dimensions );
                data.push_back( v );
                pts.push_back( v.begin(), v.end() );
            }

            EMST<Dataset, Hasher, Distance> emst( dimensions, 512, data, 0.001, 0.0 );

            auto exact = emst.exact_tree();
            auto exact_with_fp = emst.find_tree();

            REQUIRE( exact.first == exact_with_fp.first );
        }
    }

    TEST_CASE( "EMST mutual reachability distance" ) {
        using Dataset = EuclideanPoints;
        using Distance = EuclideanDistance;
        using Hasher = LatticeLSH<4, Dataset, Distance>;
        const size_t dimensions = 8;

        const std::vector<size_t> sizes = { 10000 };
        for ( size_t n : sizes ) {
            std::vector<std::vector<float>> data;
            EuclideanPoints pts( dimensions );
            for ( size_t i = 0; i < n; i++ ) {
                auto v = sample_random_normal_vector( dimensions );
                data.push_back( v );
                pts.push_back( v.begin(), v.end() );
            }

            EMST<Dataset, Hasher, Distance> emst( dimensions, 512, data, 0.001, 0.0 );

            const size_t num_neighbors = 5;
            auto exact = emst.exact_mutual_reachability_distance_tree( num_neighbors );
            auto probabilistic = emst.find_tree_mutual_reachability_distance( num_neighbors );
            float probabilistic_weight = 0.0;
            for ( auto& edge : probabilistic.first ) {
                probabilistic_weight += edge.weight;
            }

            REQUIRE( exact.first == probabilistic_weight );
        }
    }
} // namespace panna

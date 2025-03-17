#pragma once

#include <catch2/catch_test_macros.hpp>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/rand.hpp"

namespace panna {

    template <typename Dataset, typename Distance, typename HasherBuilder>
    void
    test_hash_collision_probability( HasherBuilder builder,
                                     unsigned int dimensions,
                                     unsigned int repetitions,
                                     unsigned int num_experiments = 1000 ) {
        using Hasher = typename HasherBuilder::Output;
        const float ACCEPTED_DEVIATION = 0.05;
        seed_global_rng( 1234 );

        Hasher hasher = builder.build( repetitions );

        std::vector<typename Hasher::Value> output_a;
        std::vector<typename Hasher::Value> output_b;
        for ( unsigned int i = 0; i < num_experiments; i++ ) {
            Dataset dataset( dimensions );
            dataset.push_back_random_normal();
            dataset.push_back_random_normal();

            typename Dataset::PointHandle vec_a = dataset[0];
            typename Dataset::PointHandle vec_b = dataset[1];

            hasher.hash( vec_a, output_a );
            hasher.hash( vec_b, output_b );
            float dist = Distance::compute( vec_a, vec_b );
            float prob = hasher.collision_probability( dist );
            float empirical = 0.0;
            REQUIRE( output_a.size() == repetitions );
            REQUIRE( output_b.size() == repetitions );
            for ( size_t i = 0; i < repetitions; i++ ) {
                if ( output_a[i] == output_b[i] ) { empirical += 1; }
            }
            empirical /= repetitions;
            REQUIRE( std::abs( empirical - prob ) <= ACCEPTED_DEVIATION );
        }
    }

    TEST_CASE( "Simhash collision probability" ) {
        using Dataset = UnitNormPoints;
        for ( size_t dimensions : { 10, 100, 200 } ) {
            SimhashBuilder<1, Dataset> builder( dimensions );
            test_hash_collision_probability<Dataset,
                                            AngularDistance,
                                            SimhashBuilder<1, Dataset>>(
                builder, dimensions, 4096 );
        }
    }

    TEST_CASE( "E2LSH collision probability" ) {
        using Dataset = NormedPoints;
        float r = 1.0;
        for ( size_t dimensions : { 10, 100, 200 } ) {
            E2LSHBuilder<1, Dataset> builder( r, dimensions );
            test_hash_collision_probability<Dataset,
                                            EuclideanDistance,
                                            E2LSHBuilder<1, Dataset>>(
                builder, dimensions, 4096 );
        }
    }

} // namespace panna

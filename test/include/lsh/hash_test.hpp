#pragma once

#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/minhash.hpp"
#include "panna/lsh/predicates.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/lsh/tensoring.hpp"
#include "panna/rand.hpp"

namespace panna {

    template <typename Dataset, typename Distance, typename HasherBuilder>
    static void test_hash_collision_probability( HasherBuilder builder,
                                                 unsigned int dimensions,
                                                 unsigned int repetitions,
                                                 unsigned int num_experiments = 1000,
                                                 float accepted_deviation = 0.05 ) {
        using Hasher = typename HasherBuilder::Output;
        seed_global_rng( 1234 );

        Hasher hasher = builder.build( repetitions );

        float total_expected = 0;
        float total_actual = 0;
        std::vector<typename Hasher::Value> output_a;
        std::vector<typename Hasher::Value> output_b;
        for ( unsigned int i = 0; i < num_experiments; i++ ) {
            Dataset dataset( dimensions );
            dataset.push_back_random();
            dataset.push_back_random();

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
                if ( output_a[i] == output_b[i] ) {
                    empirical += 1;
                }
            }
            empirical /= repetitions;
            total_actual += empirical;
            total_expected += prob;
        }
        REQUIRE(std::abs(total_actual - total_expected) / num_experiments <= accepted_deviation);
    }

    TEST_CASE( "Simhash collision probability" ) {
        using Dataset = UnitNormPoints;
        for ( size_t dimensions : { 10, 100, 200 } ) {
            SimhashBuilder<1, Dataset, CosineDistance> builder( dimensions );
            test_hash_collision_probability<Dataset,
                                            CosineDistance,
                                            SimhashBuilder<1, Dataset, CosineDistance>>(
                builder, dimensions, 4096 );
        }
    }

    TEST_CASE( "Simhash collision probability (angular distance)" ) {
        using Dataset = UnitNormPoints;
        using Distance = AngularDistance;
        for ( size_t dimensions : { 10, 100, 200 } ) {
            SimhashBuilder<1, Dataset, Distance> builder( dimensions );
            test_hash_collision_probability<Dataset,
                                            Distance,
                                            SimhashBuilder<1, Dataset, Distance>>(
                builder, dimensions, 4096 );
        }
    }

    TEST_CASE( "CrossPolytope collision probability" ) {
        using Dataset = UnitNormPoints;
        using Distance = CosineDistance;
        for ( size_t dimensions : { 10, 100, 200 } ) {
            CrossPolytopeBuilder<1, Dataset, Distance> builder( dimensions, 8192 );
            test_hash_collision_probability<Dataset,
                                            Distance,
                                            CrossPolytopeBuilder<1, Dataset, Distance>>(
                builder, dimensions, 4096 );
        }
    }

    TEST_CASE( "E2LSH collision probability" ) {
        using Dataset = EuclideanPoints;
        float r = 1.0;
        for ( size_t dimensions : { 10, 100, 200 } ) {
            E2LSHBuilder<1, Dataset, EuclideanDistance> builder( r, dimensions );
            test_hash_collision_probability<Dataset, EuclideanDistance, E2LSHBuilder<1, Dataset, EuclideanDistance>>(
                builder, dimensions, 4096 );
        }
    }

    TEST_CASE( "MinHash collision probability" ) {
        using Dataset = SparseSets;
        for ( size_t dimensions : { 20000 } ) {
            MinhashBuilder<1, Dataset> builder;
            test_hash_collision_probability<Dataset, JaccardDistance, MinhashBuilder<1, Dataset>>(
                builder, dimensions, 4096 );
        }
    }

    TEST_CASE( "MinHash1Bit collision probability" ) {
        using Dataset = SparseSets;
        for ( size_t dimensions : { 200000 } ) {
            Minhash1BitBuilder<1, Dataset> builder;
            test_hash_collision_probability<Dataset,
                                            JaccardDistance,
                                            Minhash1BitBuilder<1, Dataset>>(
                builder, dimensions, 4096, 10 );
        }
    }

    template <typename Dataset, typename Distance, typename HasherBuilder>
    static void test_failure_probability( HasherBuilder builder,
                                          unsigned int dimensions,
                                          unsigned int repetitions,
                                          unsigned int num_experiments = 1,
                                          size_t samples_per_experiment = 1000 ) {

        using Hasher = typename HasherBuilder::Output;
        seed_global_rng( 1234 );

        Hasher hasher = builder.build( repetitions );
        size_t max_concats = hasher.get_concatenations();

        std::vector<typename Hasher::Value> output_a;
        std::vector<typename Hasher::Value> output_b;
        for ( unsigned int experiment = 0; experiment < num_experiments; experiment++ ) {
            Dataset dataset( dimensions );
            dataset.push_back_random();
            dataset.push_back_random();
            typename Dataset::PointHandle vec_a = dataset[0];
            typename Dataset::PointHandle vec_b = dataset[1];
            float dist = Distance::compute( vec_a, vec_b );

            std::vector<std::vector<float>> counters;
            for ( size_t k = 1; k <= max_concats; k++ ) {
                std::vector<float> counters_row( repetitions );
                std::fill( counters_row.begin(), counters_row.end(), 0.0 );
                counters.push_back( counters_row );
            }

            std::vector<std::vector<float>> expected;
            for ( size_t k = 1; k <= max_concats; k++ ) {
                std::vector<float> expected_row;
                for ( size_t rep = 1; rep <= repetitions; rep++ ) {
                    float fp =
                        failure_probability( hasher, dist, k, rep, hasher.get_repetitions() );
                    expected_row.push_back( fp );
                }
                expected.push_back( expected_row );
            }

            for ( size_t sample = 0; sample < samples_per_experiment; sample++ ) {
                hasher = builder.build( repetitions );
                hasher.hash( vec_a, output_a );
                hasher.hash( vec_b, output_b );
                for ( size_t k = 1; k <= max_concats; k++ ) {
                    for ( size_t rep = 0; rep < repetitions; rep++ ) {
                        bool found = false;
                        for ( size_t i = 0; i < rep; i++ ) {
                            found |= output_a[i].prefix_eq( output_b[i], k );
                            if ( found ) {
                                break;
                            }
                        }
                        if ( k < max_concats && !found ) {
                            for ( size_t i = rep; i < repetitions; i++ ) {
                                found |= output_a[i].prefix_eq( output_b[i], k + 1 );
                                if ( found ) {
                                    break;
                                }
                            }
                        }
                        if ( !found ) {
                            counters[k - 1][rep] += 1;
                        }
                    }
                }
            }

            for ( size_t k = 1; k <= max_concats; k++ ) {
                for ( size_t rep = 0; rep < repetitions; rep++ ) {
                    float actual = counters[k - 1][rep] / samples_per_experiment;
                    float exp = expected[k - 1][rep];
                    // We check that the actual failure probability is lower than the expected one
                    REQUIRE( actual < exp + 0.03 );
                }
            }
        }
    }

    TEST_CASE( "Failure probability independent" ) {
        using Dataset = UnitNormPoints;
        for ( size_t dimensions : { 10 } ) {
            SimhashBuilder<24, Dataset, CosineDistance> builder( dimensions );
            test_failure_probability<Dataset,
                                     CosineDistance,
                                     SimhashBuilder<24, Dataset, CosineDistance>>(
                builder, dimensions, 128 );
        }
    }

    TEST_CASE( "Failure probability tensoring" ) {
        using Dataset = UnitNormPoints;
        for ( size_t dimensions : { 10 } ) {
            SimhashBuilder<12, Dataset, CosineDistance> inner_builder( dimensions );
            TensoringBuilder<SimhashBuilder<12, Dataset, CosineDistance>, Dataset> builder(
                inner_builder );
            test_failure_probability<
                Dataset,
                CosineDistance,
                TensoringBuilder<SimhashBuilder<12, Dataset, CosineDistance>, Dataset>>(
                builder, dimensions, 128 );
        }
    }

    TEST_CASE( "Tensoring" ) {
        using Dataset = UnitNormPoints;
        using Distance = CosineDistance;
        using Hasher = CrossPolytope<2, Dataset, Distance>;
        using TensoredHasher = Tensoring<Hasher, Dataset>;

        const uint8_t K_HALF = 2;
        const size_t dimensions = 128;
        CrossPolytopeBuilder<K_HALF, Dataset, Distance> builder( dimensions );

        TensoredHasher tensored( builder, 4096 );
        static_assert( tensored.get_concatenations() == 2 * K_HALF,
                       "the tensored dataset should have twice the concatenations" );

        Dataset dataset( dimensions );
        dataset.push_back_random();
        std::vector<TensoredHasher::Value> hash_a;
        std::vector<TensoredHasher::Value> hash_b;
        tensored.hash( dataset[0], hash_a );
        tensored.hash( dataset[0], hash_b );

        REQUIRE( hash_a == hash_b );

        // for ( float d : { 0.1, 0.2, 0.3 } ) {
        //     float fp = failure_probability( tensored, d, 4, 1, tensored.get_repetitions() );
        // }
    }

} // namespace panna

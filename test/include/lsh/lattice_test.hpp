#pragma once
#include <catch2/catch_test_macros.hpp>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/lattice.hpp"
#include "panna/lsh/values.hpp"
#include "panna/rand.hpp"

namespace panna {

    TEST_CASE( "Lattice decoding" ) {
        // here all the coordinates of the lattice are multiplied by 2, so that D_8 + 1/2 still has integer coordinates

        // examples from Jégou et al. https://inria.hal.science/inria-00318614/document section 2.2
        std::array<float, 8> x{ 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.8, 1.4 };
        auto decoded = decode_d8( x, 0.0 );
        std::array<float, 8> expected{ 1, 1, 1, 1, 1, 1, 2, 2 };
        REQUIRE( decoded == expected );

        decoded = decode_d8(x, -0.5);
        expected = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
        REQUIRE( decoded == expected );

        // examples from Conway and Sloane http://neilsloane.com/doc/Me83.pdf section 6
        x = { 0.1, 0.1, 0.8, 1.3, 2.2, -0.6, -0.7, 0.9 };
        decoded = decode_d8( x, 0.0 );
        expected = { 0, 0, 1, 1, 2, 0, -1, 1 };
        REQUIRE( decoded == expected );

        decoded = decode_d8(x, -0.5);
        expected = { -0.5, 0.5, 0.5, 1.5, 2.5, -0.5, -0.5, 0.5 };
        REQUIRE( decoded == expected );
    }

    TEST_CASE( "LatticeLSH empirical collision probabilities" ) {
        using HashFamily = LatticeLSH<1, EuclideanPoints, EuclideanDistance>;
        const float scaling_factor = 1.0;
        const size_t samples = 1e5;

        for (size_t dimensions : {8, 16, 128}) {
            LOG_INFO("msg", "new test", "dimension", dimensions);
            const size_t repetitions = samples;
            std::vector<HashFamily::Value> h1;
            std::vector<HashFamily::Value> h2;
            float prev_p = 1.0;
            float prev_e2lsh = 1.0;
            float distance = 0.5;
            while ( prev_p > 0 ) {
                EuclideanPoints pts( dimensions );
                auto x = sample_random_normal_vector( dimensions );
                auto direction = sample_random_normal_vector( dimensions );
                normalize( direction );
                rescale( direction, distance );
                auto y = add( x, direction );

                pts.push_back( x.begin(), x.end() );
                pts.push_back( y.begin(), y.end() );
                const float d = EuclideanDistance::compute( pts[0], pts[1] );

                E2LSH<1, EuclideanPoints, EuclideanDistance> e2lsh( scaling_factor, dimensions, 1 );
                HashFamily lsh( 0, 1, dimensions, repetitions );
                lsh.hash( pts[0], h1 );
                lsh.hash( pts[1], h2 );
                size_t collisions = 0;
                for ( size_t rep = 0; rep < repetitions; rep++ ) {
                    if ( h1[rep] == h2[rep] ) {
                        collisions++;
                    }
                }
                float empirical_p = ( (float)collisions ) / repetitions;
                float e2lsh_p = e2lsh.collision_probability( d );
                float rho = std::log( prev_p ) / std::log( empirical_p );
                float rho_e2lsh = std::log( prev_e2lsh ) / std::log( e2lsh_p );
                LOG_INFO( "d",
                          d,
                          "empirical",
                          empirical_p,
                          "e2lsh",
                          e2lsh_p,
                          "rho",
                          rho,
                          "rho-e2lsh",
                          rho_e2lsh
                          );
                distance *= 2;
                prev_p = empirical_p;
                prev_e2lsh = e2lsh_p;
            }
        }
    }
} // namespace panna

#pragma once

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>

#include "dbg.h"
#include "panna/data.hpp"
#include "panna/linalg.hpp"
#include "panna/rand.hpp"

namespace panna {

    //! A dummy storage for points, only for testing purposes.
    class DummyPoints {
        using PointHandle = std::vector<float>;

        size_t dimensions;
        std::vector<std::vector<float>> points;

    public:
        DummyPoints( size_t dimensions ): dimensions( dimensions ) {
        }

        void push_back_random() {
            std::vector<float> values;
            for ( unsigned int i = 0; i < dimensions; i++ ) {
                values.push_back( sample_random_normal() );
            }

            points.push_back( values );
        }

        void push_back( std::vector<float> v ) {
            points.push_back( v );
        }

        size_t size() const {
            return points.size();
        }

        PointHandle operator[]( size_t i ) {
            assert( i < points.size() );
            return points[i];
        }
    };

    TEST_CASE( "UnitNormPoints round trip" ) {
        const float TOLERANCE = 0.0001;
        float v = -1.0;
        while ( v <= 1.0 ) {
            REQUIRE( std::abs( from_16bit_fixed_point( to_16bit_fixed_point( v ) ) - v ) <
                     TOLERANCE );
            v += TOLERANCE;
        }

        size_t dims = 20;
        UnitNormPoints data( dims );
        for ( size_t i = 0; i < 100; i++ ) {
            std::vector<float> x = sample_random_normal_vector( dims );
            normalize( x );
            data.push_back( x.begin(), x.end() );
            UnitNormPointHandle handle = data[i];
            std::vector<float> check( dims );
            handle.into_vec( check );
            for ( size_t dim = 0; dim < dims; dim++ ) {
                REQUIRE( std::abs( x.at( dim ) - check.at( dim ) ) < TOLERANCE );
            }
        }
    }

    TEST_CASE( "UnitNormPoints::to_16bit_fixed_point" ) {
        REQUIRE( to_16bit_fixed_point( 0.99999 ) == INT16_MAX );
        REQUIRE( to_16bit_fixed_point( 1.0 ) == INT16_MAX );
        REQUIRE( to_16bit_fixed_point( -1.0 ) == INT16_MIN );
        REQUIRE( to_16bit_fixed_point( 0.0 ) == 0 );
        REQUIRE( to_16bit_fixed_point( 0.5 ) == 0x4000 );
        REQUIRE( to_16bit_fixed_point( -0.5 ) == (int16_t)0xc000 );
    }

    TEST_CASE( "UnitNormPoints::from_16bit_fixed_point" ) {
        REQUIRE( from_16bit_fixed_point( 0x0000 ) == 0.0 );
        REQUIRE( from_16bit_fixed_point( 0x4000 ) == 0.5 );
        REQUIRE( from_16bit_fixed_point( 0xa000 ) == -0.75 );
        REQUIRE( from_16bit_fixed_point( 0x8000 ) == -1.0 );
        REQUIRE( from_16bit_fixed_point( 0x7fff ) ==
                 ( (float)INT16_MAX ) / ( ( (float)INT16_MAX ) + 1 ) );
    }

    TEST_CASE( "pad_dimensions" ) {
        REQUIRE( UnitNormPoints( 0 ).get_padding() == 0 );
        REQUIRE( UnitNormPoints( 1 ).get_padding() == 15 );
        REQUIRE( UnitNormPoints( 16 ).get_padding() == 0 );
        REQUIRE( UnitNormPoints( 17 ).get_padding() == 15 );
    }

    TEST_CASE( "chunks per point" ) {
        REQUIRE( UnitNormPoints( 0 ).get_chunks_per_point() == 0 );
        REQUIRE( UnitNormPoints( 1 ).get_chunks_per_point() == 1 );
        REQUIRE( UnitNormPoints( 16 ).get_chunks_per_point() == 1 );
        REQUIRE( UnitNormPoints( 17 ).get_chunks_per_point() == 2 );
    }
} // namespace panna

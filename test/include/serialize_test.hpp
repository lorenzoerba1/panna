#pragma once

#include <catch2/catch_test_macros.hpp>

#include "cereal/archives/binary.hpp"
#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/rand.hpp"
#include "panna/trieindex.hpp"

namespace panna {

    template <typename T>
    void check_round_trip( T val ) {
        std::ostringstream os;
        {
            cereal::BinaryOutputArchive oar( os );
            oar( val );
        }

        T i_val;
        std::istringstream is( os.str() );
        {
            cereal::BinaryInputArchive iar( is );
            iar( i_val );
        }

        REQUIRE( val == i_val );
    }

    TEST_CASE( "serialization" ) {
        UnitNormPoints pts( 10 );
        pts.push_back_random();
        pts.push_back_random();
        pts.push_back_random();
        check_round_trip( pts );

        CrossPolytope<3, UnitNormPoints, CosineDistance> cp( 10, 128 );
        check_round_trip( cp );

        using Distance = CosineDistance;
        using Dataset = UnitNormPoints;
        using HasherBuilder = CrossPolytopeBuilder<3, Dataset, Distance>;
        using Hasher = HasherBuilder::Output;

        size_t dimensions = 100;
        HasherBuilder hbuilder( dimensions );

        Index<Dataset, Hasher, Distance> index( dimensions, hbuilder, 256 );
        for ( size_t i = 0; i < 100; i++ ) {
            std::vector<float> point = sample_random_normal_vector(dimensions);
            index.insert( point.begin(), point.end() );
        }
        check_round_trip( index );
    }
} // namespace panna

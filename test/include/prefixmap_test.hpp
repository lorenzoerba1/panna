#pragma once

#include <catch2/catch_test_macros.hpp>
#include "panna/logging.hpp"
#include "panna/prefixmap.hpp"

namespace panna {

    TEST_CASE( "cartesian index" ) {

        CartesianIndex idx( 0, 4, 10, 14 );
        std::vector<std::pair<uint32_t, uint32_t>> out;
        while (true) {
            auto maybe_pair = idx.next();
            if (!maybe_pair) {
                break;
            }
            LOG_INFO("i", maybe_pair->first, "j", maybe_pair->second);
            out.push_back(*maybe_pair);
        }

        std::vector<std::pair<uint32_t, uint32_t>> expected;
        for ( uint32_t i = 0; i < 4; i++ ) {
            for ( uint32_t j = 10; j < 14; j++ ) {
                expected.push_back( std::make_pair( i, j ) );
            }
        }

        REQUIRE( out == expected );
    }

    TEST_CASE( "chained index" ) {
        ChainedIndex<CartesianIndex> idx(
            { CartesianIndex( 0, 4, 10, 14 ), CartesianIndex( 20, 24, 30, 34 ) } );
        std::vector<std::pair<uint32_t, uint32_t>> out;
        while (true) {
            auto maybe_pair = idx.next();
            if (!maybe_pair) {
                break;
            }
            LOG_INFO("i", maybe_pair->first, "j", maybe_pair->second);
            out.push_back(*maybe_pair);
        }

        std::vector<std::pair<uint32_t, uint32_t>> expected;
        for ( uint32_t i = 20; i < 24; i++ ) {
            for ( uint32_t j = 30; j < 34; j++ ) {
                expected.push_back( std::make_pair( i, j ) );
            }
        }
        for ( uint32_t i = 0; i < 4; i++ ) {
            for ( uint32_t j = 10; j < 14; j++ ) {
                expected.push_back( std::make_pair( i, j ) );
            }
        }

        REQUIRE( out == expected );
    }

} // namespace panna

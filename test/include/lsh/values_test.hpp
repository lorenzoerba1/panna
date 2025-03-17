#pragma once
#include <catch2/catch_test_macros.hpp>

#include "panna/lsh/values.hpp"

TEST_CASE( "Bit hashes interleaving" ) {
    using namespace panna;
    BitwiseLshValue<16> a = BitwiseLshValue<16>::make( 0xaaaa );
    BitwiseLshValue<16> b = BitwiseLshValue<16>::make( 0xaaaa );
    BitwiseLshValue<32> expected = BitwiseLshValue<32>::make( 0xcccccccc );
    REQUIRE( BitwiseLshValue<32>::interleave( a, b ) == expected );
}

TEST_CASE( "Byte hashes interleaving" ) {
    using namespace panna;
    std::array<uint8_t, 4> bytes_a = { 0, 2, 4, 6 };
    std::array<uint8_t, 4> bytes_b = { 1, 3, 5, 7 };
    std::array<uint8_t, 8> bytes_expected = { 0, 1, 2, 3, 4, 5, 6, 7 };
    BytewiseLshValue<4> a = BytewiseLshValue<4>::make( bytes_a );
    BytewiseLshValue<4> b = BytewiseLshValue<4>::make( bytes_b );
    BytewiseLshValue<8> expected = BytewiseLshValue<8>::make( bytes_expected );
    REQUIRE( BytewiseLshValue<8>::interleave( a, b ) == expected );
}

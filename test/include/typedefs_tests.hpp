#include <catch2/catch_test_macros.hpp>

#include "panna/typedefs.hpp"

TEST_CASE( "Bit hashes interleaving" ) {
    using namespace panna;
    BitwiseLshValue<16> a = BitwiseLshValue<16>::make( 0xaaaa );
    BitwiseLshValue<16> b = BitwiseLshValue<16>::make( 0xaaaa );
    BitwiseLshValue<32> expected = BitwiseLshValue<32>::make( 0xcccccccc );
    REQUIRE( a.interleave( b ) == expected );
}

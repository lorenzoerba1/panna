#pragma once

#include <catch2/catch_test_macros.hpp>

#include "panna/data.hpp"
#include "panna/linalg.hpp"

namespace panna {

    TEST_CASE( "dot_product_i16 versions equal" ) {
        size_t reps = 100;
        for ( size_t dims : { 50, 100, 128, 200, 256 } ) {

            for ( unsigned i = 0; i < reps; i++ ) {
                UnitNormPoints dataset( dims );
                dataset.push_back_random_normal();
                dataset.push_back_random_normal();
                UnitNormPointHandle a = dataset[0];
                UnitNormPointHandle b = dataset[1];

                int16_t simple = dot_product_chunks16_simple( a, b );
#ifdef __AVX2__
                int16_t avx2 = dot_product_chunks16_avx2( a, b );
                REQUIRE( simple == avx2 );
#endif
            }
        }
    }

    // TEST_CASE("l2_distance_float versions equal") {
    //     unsigned reps = 100;
    //     unsigned dims = 100;
    //     Dataset<RealVectorFormat> dataset(dims);

    //     for (unsigned i=0; i < reps; i++) {
    //         auto a = RealVectorFormat::generate_random(dims);
    //         auto b = RealVectorFormat::generate_random(dims);
    //         auto sa = to_stored_type<RealVectorFormat>(a,
    //         dataset.get_description()); auto sb =
    //         to_stored_type<RealVectorFormat>(b, dataset.get_description());

    //         float simple = l2_distance_float_simple(sa.get(), sb.get(),
    //         dims); #ifdef __AVX__
    //             float avx = l2_distance_float_avx(sa.get(), sb.get(), dims);
    //             // Order of operations differ, so small error is accetable.
    //             REQUIRE(simple == Approx(avx).epsilon(0.0001));
    //         #endif
    //     }
    // }
} // namespace panna

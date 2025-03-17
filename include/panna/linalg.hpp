#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "panna/data.hpp"

#if defined( __AVX2__ ) || defined( __AVX__ )
    #include <immintrin.h>
#endif

namespace panna {

    template <typename T>
    static float dot_product( T a, T b );

    static float dummy_dot( std::vector<float> a, std::vector<float> b ) {
        assert( a.size() == b.size() );
        float sum = 0.0;
        for ( size_t i = 0; i < a.size(); i++ ) {
            sum += a[i] * b[i];
        }
        return sum;
    }

#ifdef __AVX2__
    inline static int16_t reduce_sum( __m256i values ) {
        const static unsigned int VALUES_PER_CHUNK = 16;
        alignas( 32 ) int16_t stored[VALUES_PER_CHUNK];
        _mm256_store_si256( (__m256i*)stored, values );
        int16_t ret = 0;
        for ( unsigned i = 0; i < VALUES_PER_CHUNK; i++ ) {
            ret += stored[i];
        }
        return ret;
    }

    inline static int16_t dot_product_chunks16_avx2( UnitNormPointHandle lhs,
                                                     UnitNormPointHandle rhs ) {
        assert( lhs.num_chunks == rhs.num_chunks );
        __m256i res = _mm256_mulhrs_epi16(
            _mm256_load_si256( (__m256i*)lhs.chunks[0].chunk ),
            _mm256_load_si256( (__m256i*)rhs.chunks[0].chunk ) );
        for ( size_t i = 1; i < lhs.num_chunks; i += 1 ) {
            __m256i tmp = _mm256_mulhrs_epi16(
                _mm256_load_si256( (__m256i*)lhs.chunks[i].chunk ),
                _mm256_load_si256( (__m256i*)rhs.chunks[i].chunk ) );
            res = _mm256_add_epi16( res, tmp );
        }
        return reduce_sum(res);
    }
#endif

    inline static int16_t
    dot_product_chunks16_simple( UnitNormPointHandle lhs,
                                 UnitNormPointHandle rhs ) {
        assert( lhs.num_chunks == rhs.num_chunks );
        const static unsigned int VALUES_PER_CHUNK = 16;

        int16_t res = 0;
        for ( size_t chunk_idx = 0; chunk_idx < lhs.num_chunks; chunk_idx++ ) {
            for ( size_t i = 0; i < VALUES_PER_CHUNK; i++ ) {
                int32_t precise =
                    static_cast<int32_t>( lhs.chunks[chunk_idx].chunk[i] ) *
                    static_cast<int32_t>( rhs.chunks[chunk_idx].chunk[i] );
                res += static_cast<int16_t>( ( ( precise >> 14 ) + 1 ) >> 1 );
            }
        }
        return res;
    }

    static inline int16_t dot_product_chunks16( UnitNormPointHandle lhs,
                                                UnitNormPointHandle rhs ) {
#ifdef __AVX2__
        return dot_product_chunks16_avx2( lhs, rhs );
#else
    #pragma message( \
        "there is no AVX instruction set on this machine, performance is severely limited" )
        return dot_product_chunks16_simple( lhs, rhs );
#endif
    }

    template <>
    float dot_product( UnitNormPointHandle a, UnitNormPointHandle b ) {
        return from_16bit_fixed_point( dot_product_chunks16( a, b ) );
    }

} // namespace panna

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ffht/fht_header_only.h"
#include "panna/data.hpp"

#if defined( __AVX2__ ) || defined( __AVX__ )
    #include <immintrin.h>
#endif

namespace panna {

    template <typename T>
    static T add( T &a, T &b );

    template<>
    std::vector<float> add(std::vector<float> &a, std::vector<float>&b) {
        expect(a.size() == b.size());
        std::vector<float> out(a.size());
        for (size_t i=0; i<a.size(); i++) {
            out.at(i) = a.at(i) + b.at(i);
        }
        return out;
    }

    

    template <typename T>
    static float dot_product( T a, T b );

    template<>
    float dot_product( std::vector<float> a, std::vector<float> b ) {
        assert( a.size() == b.size() );
        float sum = 0.0;
        for ( size_t i = 0; i < a.size(); i++ ) {
            sum += a.at(i) * b.at(i);
        }
        return sum;
    }

    template<>
    float dot_product( EuclideanPointHandle a, EuclideanPointHandle b ) {
        expect( a.dimensions == b.dimensions );
        float sum = 0.0;
        for ( size_t i = 0; i < a.dimensions; i++ ) {
            sum += a.vector[i] * b.vector[i];
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
        __m256i res =
            _mm256_mulhrs_epi16( _mm256_load_si256( (__m256i*)lhs.chunks[0].chunk.data() ),
                                 _mm256_load_si256( (__m256i*)rhs.chunks[0].chunk.data() ) );
        for ( size_t i = 1; i < lhs.num_chunks; i += 1 ) {
            __m256i tmp =
                _mm256_mulhrs_epi16( _mm256_load_si256( (__m256i*)lhs.chunks[i].chunk.data() ),
                                     _mm256_load_si256( (__m256i*)rhs.chunks[i].chunk.data() ) );
            res = _mm256_add_epi16( res, tmp );
        }
        return reduce_sum( res );
    }
#endif

    inline static int16_t dot_product_chunks16_simple( UnitNormPointHandle lhs,
                                                       UnitNormPointHandle rhs ) {
        assert( lhs.num_chunks == rhs.num_chunks );
        const static unsigned int VALUES_PER_CHUNK = 16;

        int16_t res = 0;
        for ( size_t chunk_idx = 0; chunk_idx < lhs.num_chunks; chunk_idx++ ) {
            for ( size_t i = 0; i < VALUES_PER_CHUNK; i++ ) {
                int32_t precise = static_cast<int32_t>( lhs.chunks[chunk_idx].chunk[i] ) *
                                  static_cast<int32_t>( rhs.chunks[chunk_idx].chunk[i] );
                res += static_cast<int16_t>( ( ( precise >> 14 ) + 1 ) >> 1 );
            }
        }
        return res;
    }

    static inline int16_t dot_product_chunks16( UnitNormPointHandle lhs, UnitNormPointHandle rhs ) {
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

    template <>
    float dot_product( NormedPointHandle a, NormedPointHandle b ) {
        float inner_dot = from_16bit_fixed_point( dot_product_chunks16( a.inner, b.inner ) );
        return std::sqrt( a.squared_norm() ) * std::sqrt( b.squared_norm() ) * inner_dot;
    }

    static void normalize(std::vector<float> & point) {
        float norm = std::sqrt(dot_product(point, point));
        if (norm == 0) {
            //throw std::runtime_error("Cannot normalize a zero vector");
            return;
        }
        for (size_t i=0; i<point.size(); i++) {
            point.at(i) /= norm;
        }
    }

    static void rescale(std::vector<float> & point, float factor) {
        for (size_t i=0; i<point.size(); i++) {
            point.at(i) *= factor;
        }
    }

    inline static float euclidean_naive( const float* a, const float* b, std::size_t n ) {
        float d = 0;
        for ( size_t i = 0; i < n; i++ ) {
            float diff = a[i] - b[i];
            d += diff * diff;
        }
        return std::sqrt( d );
    }

#ifdef __AVX2__
    inline static float euclidean_avx2( const float* a, const float* b, std::size_t n ) {
        if ( n == 0 ) {
            return 0.0f;
        }

        // Use eight accumulators for better instruction-level parallelism.
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        std::size_t i = 0;
        // 8x unrolled loop processing 64 floats per iteration.
        for ( ; i + 64 <= n; i += 64 ) {
            __m256 va0 = _mm256_loadu_ps( a + i + 0 );
            __m256 vb0 = _mm256_loadu_ps( b + i + 0 );
            __m256 diff0 = _mm256_sub_ps( va0, vb0 );
            acc0 = _mm256_fmadd_ps( diff0, diff0, acc0 );

            __m256 va1 = _mm256_loadu_ps( a + i + 8 );
            __m256 vb1 = _mm256_loadu_ps( b + i + 8 );
            __m256 diff1 = _mm256_sub_ps( va1, vb1 );
            acc1 = _mm256_fmadd_ps( diff1, diff1, acc1 );

            __m256 va2 = _mm256_loadu_ps( a + i + 16 );
            __m256 vb2 = _mm256_loadu_ps( b + i + 16 );
            __m256 diff2 = _mm256_sub_ps( va2, vb2 );
            acc2 = _mm256_fmadd_ps( diff2, diff2, acc2 );

            __m256 va3 = _mm256_loadu_ps( a + i + 24 );
            __m256 vb3 = _mm256_loadu_ps( b + i + 24 );
            __m256 diff3 = _mm256_sub_ps( va3, vb3 );
            acc3 = _mm256_fmadd_ps( diff3, diff3, acc3 );

            __m256 va4 = _mm256_loadu_ps( a + i + 32 );
            __m256 vb4 = _mm256_loadu_ps( b + i + 32 );
            __m256 diff4 = _mm256_sub_ps( va4, vb4 );
            acc4 = _mm256_fmadd_ps( diff4, diff4, acc4 );

            __m256 va5 = _mm256_loadu_ps( a + i + 40 );
            __m256 vb5 = _mm256_loadu_ps( b + i + 40 );
            __m256 diff5 = _mm256_sub_ps( va5, vb5 );
            acc5 = _mm256_fmadd_ps( diff5, diff5, acc5 );

            __m256 va6 = _mm256_loadu_ps( a + i + 48 );
            __m256 vb6 = _mm256_loadu_ps( b + i + 48 );
            __m256 diff6 = _mm256_sub_ps( va6, vb6 );
            acc6 = _mm256_fmadd_ps( diff6, diff6, acc6 );

            __m256 va7 = _mm256_loadu_ps( a + i + 56 );
            __m256 vb7 = _mm256_loadu_ps( b + i + 56 );
            __m256 diff7 = _mm256_sub_ps( va7, vb7 );
            acc7 = _mm256_fmadd_ps( diff7, diff7, acc7 );
        }

        // Sum the accumulators.
        acc0 = _mm256_add_ps( acc0, acc4 );
        acc1 = _mm256_add_ps( acc1, acc5 );
        acc2 = _mm256_add_ps( acc2, acc6 );
        acc3 = _mm256_add_ps( acc3, acc7 );
        acc0 = _mm256_add_ps( acc0, acc2 );
        acc1 = _mm256_add_ps( acc1, acc3 );
        __m256 acc = _mm256_add_ps( acc0, acc1 );

        // --- AVX2 + FMA vectorised loop (8 floats / iteration) for remaining elements ---
        for ( ; i + 8 <= n; i += 8 ) {
            __m256 va = _mm256_loadu_ps( a + i );  // load 8 floats from a (unaligned)
            __m256 vb = _mm256_loadu_ps( b + i );  // load 8 floats from b (unaligned)
            __m256 diff = _mm256_sub_ps( va, vb ); // diff = a - b

            // FMA: acc = diff * diff + acc  (single rounding, no extra temp register)
            acc = _mm256_fmadd_ps( diff, diff, acc );
        }

        // --- Horizontal reduction of the 8-lane accumulator ---
        // Fold high 128 bits into low 128 bits.
        __m128 lo = _mm256_castps256_ps128( acc );   // lanes 0-3
        __m128 hi = _mm256_extractf128_ps( acc, 1 ); // lanes 4-7
        __m128 sum = _mm_add_ps( lo, hi );           // pairwise sum → 4 lanes

        // Fold 4 lanes → 2 → 1
        sum = _mm_add_ps( sum, _mm_movehl_ps( sum, sum ) );     // add lanes [2,3] into [0,1]
        sum = _mm_add_ss( sum, _mm_shuffle_ps( sum, sum, 1 ) ); // add lane 1 into lane 0

        float result = _mm_cvtss_f32( sum ); // extract scalar sum

        // --- Scalar tail loop (handles n % 8 remaining elements) ---
        for ( ; i < n; ++i ) {
            float diff = a[i] - b[i];
            result += diff * diff;
        }

        return std::sqrt( result );
    }
#endif

    template <typename T>
    static float euclidean( T a, T b );

    template <>
    float euclidean( NormedPointHandle a, NormedPointHandle b ) {
        float dot = dot_product( a, b );
        return std::sqrt( a.squared_norm() + b.squared_norm() - 2 * dot );
    }

    template <>
    float euclidean( EuclideanPointHandle a, EuclideanPointHandle b ) {
#ifdef __AVX2__
        return euclidean_avx2( a.vector, b.vector, a.dimensions );
#else
        return euclidean_naive( a.vector, b.vector, a.dimensions );
#endif
    }

    template <>
    float euclidean(std::vector<float> a, std::vector<float> b) {
#ifdef __AVX2__
        return euclidean_avx2( a.data(), b.data(), a.size() );
#else
        return euclidean_naive( a.data(), b.data(), a.size() );
#endif
    }

    template <size_t D>
    float euclidean(std::array<float, D> a, std::array<float, D> b) {
#ifdef __AVX2__
        return euclidean_avx2( a.data(), b.data(), D );
#else
        return euclidean_naive( a.data(), b.data(), D );
#endif
    }

    template <size_t D>
    float euclidean(std::array<float, D> a, std::array<long, D> b) {
        float d = 0;
        for (size_t i=0; i<a.size(); i++) {
            float diff = a.at(i) - ((float) b.at(i));
            d += diff*diff;
        }
        return std::sqrt(d);
    }


    constexpr static unsigned int ceil_log( unsigned int value ) {
        unsigned int log = 0;
        unsigned int power_of_two = 1;
        while ( power_of_two < value ) {
            log++;
            power_of_two *= 2;
        }
        return log;
    }

    // Perform the dot product with many random vetors at once using the
    // Fast Hadamard Transform
    class RandomDotProducts {
        size_t num_products;
        size_t log_num_products;
        std::vector<float> random_signs[3];

    public:
        RandomDotProducts() {}

        RandomDotProducts( size_t num_products ):
            num_products( 1 << ceil_log( num_products ) ),
            log_num_products( ceil_log( num_products ) ) {

            random_signs[0].reserve(1 << log_num_products);
            random_signs[1].reserve(1 << log_num_products);
            random_signs[2].reserve(1 << log_num_products);

            std::uniform_int_distribution<int8_t> bernoulli( 0, 1 );
            auto& generator = get_global_rng();
            for ( size_t i = 0; i < 1 << log_num_products; i++ ) {
                random_signs[0].push_back( bernoulli( generator ) ? 1.0 : -1.0 );
                random_signs[1].push_back( bernoulli( generator ) ? 1.0 : -1.0 );
                random_signs[2].push_back( bernoulli( generator ) ? 1.0 : -1.0 );
            }
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( num_products, log_num_products, random_signs );
        }

        friend bool operator==( const RandomDotProducts& a,
                                const RandomDotProducts& b ) {
            return a.num_products == b.num_products && a.log_num_products == b.log_num_products &&
                   a.random_signs[0] == b.random_signs[0] &&
                   a.random_signs[1] == b.random_signs[1] && a.random_signs[2] == b.random_signs[2];
        }

        std::vector<float> allocate_scratch() const {
            std::vector<float> scratch(1 << log_num_products, 0.0);
            return scratch;
        }

        void compute( std::vector<float>& in_out, float additional_scaling = 1.0 ) const {
            expect( in_out.size() == num_products );
            // float norm_factor = std::sqrt(num_products) / (num_products * std::sqrt(static_cast<double>(num_products)));
            float norm_factor = additional_scaling / static_cast<float>(num_products);
            for ( uint8_t diagonal = 0; diagonal < 3; diagonal++ ) {
                // Multiply by a diagonal +-1 matrix.
                for ( size_t i = 0; i < ( 1 << log_num_products ); i++ ) {
                    // OPTIMIZE use simd, this takes half as much time as the fht transform below
                    in_out.at(i) *= random_signs[diagonal].at(i);
                }
                // Apply the fast hadamard transform
                fht( in_out.data(), log_num_products );
            }
            for ( size_t i = 0; i < num_products; i++ ) {
                in_out.at( i ) *= norm_factor;
            }
        }
    };

} // namespace panna

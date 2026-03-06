#pragma once

#include <catch2/catch_test_macros.hpp>

#include "panna/data.hpp"
#include "panna/linalg.hpp"
#include "panna/rand.hpp"

namespace panna {

    TEST_CASE( "dot_product_i16 versions equal" ) {
        size_t reps = 100;
        for ( size_t dims : { 50, 100, 128, 200, 256 } ) {

            for ( unsigned i = 0; i < reps; i++ ) {
                UnitNormPoints dataset( dims );
                dataset.push_back_random();
                dataset.push_back_random();
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

    TEST_CASE( "euclidean distance" ) {
        size_t reps = 100;
        for ( size_t dims : { 50, 100, 128, 200, 256, 784 } ) {
            for ( unsigned i = 0; i < reps; i++ ) {
                std::vector<float> a = sample_random_normal_vector(dims);
                std::vector<float> b = sample_random_normal_vector(dims);

                float simple = euclidean_naive( a.data(), b.data(), dims );
#ifdef __AVX2__
                float avx2 = euclidean_avx2( a.data(), b.data(), dims );
                REQUIRE( std::abs( simple - avx2 ) <= 1e-4 );
#endif
            }
        }
    }

    TEST_CASE( "pseudorandom rotations" ) {

        int d = 100;     // Input vector dimension
        int L = 1 << 14; // Output dimension (must be power of 2): 2^14 = 16384

        std::cout << "Input dimension:  " << d << std::endl;
        std::cout << "Output dimension: " << L << std::endl << std::endl;

        std::cout << "Initializing rotator..." << std::endl;
        RandomDotProducts rdps( L );
        std::cout << "Done.\n\n";

        // Create a random input vector (d-dimensional)
        std::vector<float> x( d );
        for ( int i = 0; i < d; ++i ) {
            x[i] = std::sin( i * 0.1 ); // Simple deterministic vector
        }

        float input_norm = std::sqrt(dot_product( x, x ));
        std::cout << "Input vector ||x|| = " << input_norm << std::endl;

        std::cout << "\nApplying random rotation: y = H*D3*H*D2*H*D1*[x; 0, ..., 0]" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> y(L, 0);
        std::copy(x.begin(), x.end(), y.begin());
        rdps.compute( y );
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us =
            std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

        float output_norm = std::sqrt(dot_product( y, y ));
        std::cout << "Output vector ||y|| = " << output_norm << std::endl;
        std::cout << "Time: " << duration_us / 1000.0 << " ms" << std::endl << std::endl;

        // Verify norm preservation (approximately)
        std::cout << "Norm preservation check:" << std::endl;
        std::cout << "  Input norm:  " << input_norm << std::endl;
        std::cout << "  Output norm: " << output_norm << std::endl;
        std::cout << "  Ratio (output/input): " << output_norm / input_norm << std::endl;

        // Second vector for demonstrating angle preservation
        std::vector<float> x2( d );
        for ( int i = 0; i < d; ++i ) {
            x2[i] = std::cos( i * 0.1 );
        }

        std::vector<float> y2(L, 0.0);
        std::copy(x2.begin(), x2.end(), y2.begin());
        rdps.compute( y2 );

        float dot_before = dot_product( x, x2 );
        float dot_after = dot_product( y, y2 );

        float norm_x2 = std::sqrt(dot_product( x2, x2 ));
        float norm_y2 = std::sqrt(dot_product( y2, y2 ));

        float angle_before = std::acos( dot_before / ( input_norm * norm_x2 ) );
        float angle_after = std::acos( dot_after / ( output_norm * norm_y2 ) );

        std::cout << "Angle preservation (for two vectors):" << std::endl;
        std::cout << "  Angle before: " << angle_before * 180.0 / M_PI << " degrees" << std::endl;
        std::cout << "  Angle after:  " << angle_after * 180.0 / M_PI << " degrees" << std::endl;

        // Show first few components
        std::cout << "First 10 components of rotated vector:" << std::endl;
        for ( int i = 0; i < 10; ++i ) {
            std::cout << "  y[" << i << "] = " << std::setw( 10 ) << y[i] << std::endl;
        }
        REQUIRE(std::abs(input_norm - output_norm) <= 1e-5);
    }

    TEST_CASE( "dot product distribution" ) {
        using Vec = std::vector<float>;
        size_t d = 128;
        size_t L = 1 << 10;
        Vec x = sample_random_normal_vector(d);
        // normalize(x);

        std::vector<float> empirical_distribution;
        for(size_t i=0; i<L; i++) {
            Vec dir = sample_random_normal_vector(d);
            // normalize(dir);
            float dp = dot_product(x, dir);
            empirical_distribution.push_back(dp);
        }

        RandomDotProducts rdps(L);
        Vec hadamard = rdps.allocate_scratch();
        std::copy(x.begin(), x.end(), hadamard.begin());
        rdps.compute(hadamard);

        std::ranges::sort(empirical_distribution);
        std::ranges::sort(hadamard);
        for (size_t i=0; i<L; i++) {
            std::cout << empirical_distribution[i] << " " << hadamard[i] << std::endl;
        }
    }

} // namespace panna

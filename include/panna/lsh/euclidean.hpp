#pragma once
#include <random>
#include <vector>

#include "panna/linalg.hpp"
#include "panna/lsh/values.hpp"
#include "panna/rand.hpp"

namespace panna {

    // from https://en.cppreference.com/w/cpp/numeric/math/erfc
    static double normal_cdf( double x ) {
        return std::erfc( -x / std::sqrt( 2 ) ) / 2;
    }

    template <uint8_t K, typename Dataset>
    class E2LSH {
    public:
        //! The datatype of the output
        using Value = BytewiseLshValue<K>;

    private:
        float quantization_width;
        size_t repetitions;
        Dataset random_vectors;
        std::vector<float> offsets;

    public:
        E2LSH( float quantization_width,
               size_t dimensions,
               size_t repetitions ):
            quantization_width( quantization_width ),
            repetitions( repetitions ),
            random_vectors( dimensions ) {
            auto& rng = get_global_rng();
            std::uniform_real_distribution<float> uniform( 0.0,
                                                           quantization_width );
            for ( size_t vec_idx = 0; vec_idx < repetitions * K; vec_idx++ ) {
                random_vectors.push_back_random();
                offsets.push_back( uniform( rng ) );
            }
        }

        static constexpr size_t get_concatenations() { return K; }

        size_t get_repetitions() const { return repetitions; }

        void hash( typename Dataset::PointHandle point,
                   std::vector<Value>& output ) const {
            output.clear();
            BytewiseLshValue<K> cur;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( size_t concat = 0; concat < K; concat++ ) {
                    typename Dataset::PointHandle rand_vec =
                        random_vectors[K * rep + concat];
                    float dotp = dot_product( point, rand_vec );
                    int8_t code =
                        std::floor( ( dotp + offsets[K * rep + concat] ) /
                                    quantization_width );
                    cur.set( concat, code );
                }
                output.push_back( cur );
                cur = BytewiseLshValue<K>();
            }
        }

        float collision_probability( float distance ) const {
            float r = quantization_width;
            return 1.0 - 2.0 * normal_cdf( -r / distance ) -
                   ( 2.0 / ( std::sqrt( M_PI * 2.0 ) * ( r / distance ) ) ) *
                       ( 1.0 -
                         std::exp( -r * r / ( 2.0 * distance * distance ) ) );
        }
    };

    template <uint8_t K, typename Dataset>
    class E2LSHBuilder {
        float quantization_width = 1.0;
        size_t dimensions = 0;

    public:
        using Output = E2LSH<K, Dataset>;

        E2LSHBuilder( float quantization_width, size_t dimensions ):
            quantization_width( quantization_width ),
            dimensions( dimensions ) {}

        Output build( size_t repetitions ) const {
            return E2LSH<K, Dataset>(
                quantization_width, dimensions, repetitions );
        }
    };

} // namespace panna

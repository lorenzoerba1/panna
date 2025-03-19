#pragma once

#include <vector>

#include "panna/linalg.hpp"
#include "panna/lsh/values.hpp"

namespace panna {

    template <uint8_t K, typename Dataset>
    class Simhash {
    public:
        //! The datatype of the output
        using Value = BitwiseLshValue<K>;

    private:
        size_t repetitions;
        Dataset random_vectors;

    public:
        Simhash( size_t dimensions, size_t repetitions ):
            repetitions( repetitions ), random_vectors( dimensions ) {
            for ( size_t vec_idx = 0; vec_idx < repetitions * K; vec_idx++ ) {
                random_vectors.push_back_random();
            }
        }

        static constexpr size_t get_concatenations() { return K; }

        size_t get_repetitions() const { return repetitions; }

        void hash( typename Dataset::PointHandle point,
                   std::vector<Value>& output ) const {
            output.clear();
            uint32_t cur = 0;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( size_t concat = 0; concat < K; concat++ ) {
                    typename Dataset::PointHandle rand_vec =
                        random_vectors[K * rep + concat];
                    float dotp = dot_product( point, rand_vec );
                    cur = ( cur << 1 ) | ( dotp > 0 );
                }
                output.push_back( Value::make( cur ) );
                cur = 0;
            }
        }

        float collision_probability( float distance ) const {
            float dotp = 1 - distance;
            return 1.0 - std::acos( dotp ) / M_PI;
        }
    };

    template <uint8_t K, typename Dataset>
    class SimhashBuilder {
        size_t dimensions = 0;

    public:
        using Output = Simhash<K, Dataset>;

        SimhashBuilder( size_t dimensions ): dimensions( dimensions ) {}

        Output build( size_t repetitions ) const {
            return Simhash<K, Dataset>( dimensions, repetitions );
        }
    };
} // namespace panna

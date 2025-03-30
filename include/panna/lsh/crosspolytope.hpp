#pragma once
#include <algorithm>
#include <random>
#include <vector>

#include "ffht/fht_header_only.h"
#include "panna/lsh/values.hpp"
#include "panna/rand.hpp"
#include "panna/linalg.hpp"

namespace panna {


    struct CrossPolytopeCollisionEstimates {
        std::vector<float> probabilities;
        float eps;

        CrossPolytopeCollisionEstimates() {
        }

        CrossPolytopeCollisionEstimates( unsigned int dimensions,
                                         unsigned int num_repetitions,
                                         float eps ):
            eps( eps ) {
            // adapted from
            // https://github.com/puffinn/puffinn/blob/master/include/puffinn/hash/crosspolytope.hpp

            printf("estimating collision probabilities\n");
            auto log_dimensions = ceil_log( dimensions );
            probabilities = std::vector<float>();
            size_t num_probabilities = 2 / eps;
            probabilities.resize( num_probabilities );

            // foreach [alpha, alpha+eps) segment
#pragma omp parallel for
            for ( size_t i = 0; i < num_probabilities; i++ ) {
                float alpha = -1 + eps * i;
                std::normal_distribution<double> standard_normal( 0, 1 );
                std::mt19937_64 rng;
                rng.seed( 1234 + i );

                size_t collisions = 0;

                for ( size_t i = 0; i < num_repetitions; i++ ) {
                    // length = dimensions
                    // x = (1, 0, ..., 0)
                    // y = (alpha, (1-alpha^2)^(1/2), ..., 0)
                    // The dot product between x and y is alpha, and both are
                    // unit norm.

                    // The hash value so far.
                    uint32_t hash_x = 0;
                    uint32_t hash_y = 0;
                    // Absolute value of highest value seen.
                    double v_x = 0;
                    double v_y = 0;

                    // Compute a random rotation of x and y using the matrix z
                    // [ [ z_1_0, z_2_0 ],
                    //   [ z_1_1, z_2_1 ],
                    //   [ z_1_j, z_2_j ] ]
                    for ( uint32_t j = 0; j < dimensions; j++ ) {
                        double z_1 = standard_normal( rng );
                        double z_2 = standard_normal( rng );
                        // calculate z*x[j] and find the index with the highest
                        // value
                        if ( abs( z_1 ) > v_x ) {
                            v_x = abs( z_1 );
                            hash_x = j;
                            if ( z_1 < 0 ) {
                                hash_x |= ( 1 << log_dimensions );
                            }
                        }
                        // do the same for z*y[j]
                        double h_y = alpha * z_1 + pow( 1 - pow( alpha, 2 ), 0.5 ) * z_2;
                        if ( abs( h_y ) > v_y ) {
                            v_y = abs( h_y );
                            hash_y = j;
                            if ( h_y < 0 ) {
                                hash_y |= ( 1 << log_dimensions );
                            }
                        }
                    }
                    collisions += hash_x == hash_y;
                }
                float prob;
                if ( num_repetitions != 0 ) {
                    prob = static_cast<float>( collisions ) / num_repetitions;
                } else {
                    prob = 1.0;
                }
                assert( prob <= 1.0 );
                probabilities[i] = prob;
            }
            printf("collision probabilities estimated!\n");
        }

        float get_collision_probability( float dotp ) const {
            size_t idx = std::floor( ( 1.0 + dotp ) / eps );
            return probabilities[idx];
        }
    };

    template <uint8_t K, typename Dataset, typename Distance, uint8_t ROTATIONS = 3>
    class CrossPolytope {
    public:
        //! The datatype of the output
        using Value = ShortLshValue<K>;

    private:
        size_t repetitions;
        size_t dimensions;
        size_t log_dimensions;
        std::vector<int8_t> random_signs;
        // scratch space
        std::vector<float> rotated_vector;
        CrossPolytopeCollisionEstimates estimates;

        int16_t encode_closest_axis( std::vector<float>& vec ) const {
            int res = 0;
            float max_sim = 0;
            for ( int i = 0; i < ( 1 << log_dimensions ); i++ ) {
                if ( vec[i] > max_sim ) {
                    res = i;
                    max_sim = vec[i];
                } else if ( -vec[i] > max_sim ) {
                    res = i + ( 1 << log_dimensions );
                    max_sim = -vec[i];
                }
            }
            return res;
        }

        // Hash a single repetition of the given vector
        int16_t hash_single( std::vector<float>& vec, size_t concatenation, size_t repetition ) {
            const size_t rotation_len = ( 1 << log_dimensions );
            for ( unsigned int rotation = 0; rotation < ROTATIONS; rotation++ ) {
                // Multiply by a diagonal +-1 matrix.
                size_t base_idx = ( concatenation * repetitions * ROTATIONS * rotation_len ) +
                                  ( repetition * ROTATIONS * rotation_len ) +
                                  ( rotation * rotation_len );
                for ( size_t i = 0; i < rotation_len; i++ ) {
                    vec[i] *= random_signs[base_idx + i];
                }
                // Apply the fast hadamard transform
                fht( vec.data(), log_dimensions );
            }

            return encode_closest_axis( vec );
        }

    public:
        static constexpr size_t get_concatenations() {
            return K;
        }

        CrossPolytope( size_t dimensions,
                       size_t repetitions,
                       size_t estimation_repetitions = 2 * 4096,
                       float estimation_eps = 5e-3 ):
            repetitions( repetitions ),
            dimensions( dimensions ),
            estimates( ( 1 << ceil_log( dimensions ) ), estimation_repetitions, estimation_eps ) {
            log_dimensions = ceil_log( dimensions );

            int random_signs_len = K * repetitions * ROTATIONS * ( 1 << log_dimensions );
            random_signs.reserve( random_signs_len );

            rotated_vector.resize( 1 << log_dimensions );

            std::uniform_int_distribution<int_fast32_t> sign_distribution( 0, 1 );
            auto& generator = get_global_rng();
            for ( int i = 0; i < random_signs_len; i++ ) {
                random_signs.push_back( sign_distribution( generator ) * 2 - 1 );
            }
        }

        //! Requires Dataset::PointHandle to support the operation
        //! `PointHandle::into_vec` that copies the contents of the
        //! point into the given vector, without changing its length.
        void hash( typename Dataset::PointHandle point, std::vector<Value>& output ) {
            output.clear();
            Value cur;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( size_t concat = 0; concat < K; concat++ ) {
                    point.into_vec( rotated_vector );
                    std::fill( rotated_vector.begin() + dimensions, rotated_vector.end(), 0.0 );
                    int16_t code = hash_single( rotated_vector, concat, rep );
                    cur.set( concat, code );
                }
                output.push_back( cur );
                cur = Value();
            }
        }

        float collision_probability( float distance ) const {
            float dotp = Distance::to_dot_product( distance );
            return estimates.get_collision_probability( dotp );
        }
    };

    template <uint8_t K, typename Dataset, typename Distance>
    class CrossPolytopeBuilder {
        size_t dimensions = 0;
        size_t estimation_repetitions = 1024;
        float estimation_eps = 5e-3;

    public:
        using Output = CrossPolytope<K, Dataset, Distance>;

        CrossPolytopeBuilder( size_t dimensions,
                              size_t estimation_repetitions = 1024,
                              float estimation_eps = 5e-3 ):
            dimensions( dimensions ),
            estimation_repetitions( estimation_repetitions ),
            estimation_eps( estimation_eps ) {
        }

        Output build( size_t repetitions ) const {
            return CrossPolytope<K, Dataset, Distance>(
                dimensions, repetitions, estimation_repetitions, estimation_eps );
        }
    };

} // namespace panna

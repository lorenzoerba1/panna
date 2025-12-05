#pragma once

#include <cmath>
#include <omp.h>
#include <random>
#include <vector>

#include "panna/distance.hpp"
#include "panna/expect.hpp"
#include "panna/linalg.hpp"
#include "panna/logging.hpp"
#include "panna/lsh/values.hpp"
#include "panna/lsh/lattice_probabilities.hpp"
#include "panna/rand.hpp"
namespace panna {

    template <size_t LATTICE_DIMENSIONS>
    static std::array<float, LATTICE_DIMENSIONS>
    decode_d8( std::array<float, LATTICE_DIMENSIONS>& x, float offset ) {
        std::array<float, LATTICE_DIMENSIONS> y;
        size_t worst_dimension = 0;
        float max_diff = 0.0;
        long rounded_sum = 0;
        // compute f(x)
        for ( size_t dim = 0; dim < LATTICE_DIMENSIONS; dim++ ) {
            float xval = x[dim] + offset;
            const long r = lroundf( xval );
            const float diff = abs( xval - r );
            rounded_sum += r;
            y[dim] = r;
            if ( diff > max_diff ) {
                worst_dimension = dim;
                max_diff = diff;
            }
        }

        if ( rounded_sum % 2 != 0 ) {
            // compute g(x), taking into account that out[worst_dimension]
            // already contains the correct rounding of x[worst_dimension]
            float frac = x[worst_dimension] + offset - y[worst_dimension];
            if ( frac > 0.5 || (-0.5 <= frac && frac < 0) ) {
                // round down instead of up
                y[worst_dimension]--;
            } else {
                // round up instead of down
                y[worst_dimension]++;
            }
        }

        for ( size_t dim = 0; dim < LATTICE_DIMENSIONS; dim++ ) {
            y[dim] -= offset;
        }

        return y;
    }

    template <size_t LATTICE_DIMENSIONS>
    static std::array<float, LATTICE_DIMENSIONS>
    decode_e8( std::array<float, LATTICE_DIMENSIONS>& x ) {
        // based on http://neilsloane.com/doc/Me83.pdf section 6
        auto snap1 = decode_d8( x, 0.0 );
        auto snap2 = decode_d8( x, -0.5 );
        if ( euclidean( x, snap1 ) < euclidean( x, snap2 ) ) {
            return snap1;
        } else {
            return snap2;
        }
    }

    template <size_t LATTICE_DIMENSIONS>
    static std::array<long, LATTICE_DIMENSIONS>
    to_integer_coords( std::array<float, LATTICE_DIMENSIONS>& y ) {
        std::array<long, LATTICE_DIMENSIONS> out;
        for ( size_t dim = 0; dim < LATTICE_DIMENSIONS; dim++ ) {
            out[dim] = static_cast<long>( y[dim] * 2 );
        }
        return out;
    }

    /// packs the given array into a 32 bit integer, taking the 4 least significant bits
    /// of each element of the array
    static int32_t to_int32( std::array<long, 8> arr ) {
        int32_t out = 0;
        for ( size_t i = 0; i < arr.size(); i++ ) {
            const int32_t bits = arr[i] & 0xF;
            out = ( out << 4 ) | bits;
        }
        return out;
    }

    static int64_t to_int64( std::array<long, 8> arr ) {
        int64_t out = 0;
        for ( size_t i = 0; i < arr.size(); i++ ) {
            const int64_t bits = arr[i] & 0xFF;
            out = ( out << 8 ) | bits;
        }
        return out;
    }

    template <uint8_t K, typename Dataset, typename Distance>
    class LatticeLSHBuilder;

    template <uint8_t K, typename Dataset, typename Distance>
    class LatticeLSH {
    public:
        //! The datatype of the output
        using Value = LongLshValue<K>;
        using Builder = LatticeLSHBuilder<K, Dataset, Distance>;
        static const size_t LATTICE_DIMENSIONS = 8;

    private:

        float offset;
        float scaling_factor;
        size_t repetitions;
        Dataset random_vectors;
        std::vector<float> offsets;
        // the corrections to apply to projections so that
        // they behave like the input vector was first offset and scaled
        std::vector<float> corrections;

        std::array<float, LATTICE_DIMENSIONS> project( typename Dataset::PointHandle point,
                                                       size_t concatenation,
                                                       size_t repetition ) const {
            std::array<float, LATTICE_DIMENSIONS> out;
            const size_t start = repetition * LATTICE_DIMENSIONS * K + concatenation * LATTICE_DIMENSIONS;
            const size_t end = start + LATTICE_DIMENSIONS;
            expect(end <= random_vectors.size());
            for (size_t i=0; i<LATTICE_DIMENSIONS; i++) {
                out[i] = dot_product(point, random_vectors[start+i]) / scaling_factor - corrections[start+i] + offsets[start+i];
            }
            return out;
        }


    public:
        LatticeLSH() {
        }

        LatticeLSH( float offset, float scaling_factor, size_t dimensions, size_t repetitions ):
            LatticeLSH( offset, scaling_factor, dimensions, repetitions, get_global_rng() ) {
        }

        LatticeLSH( float offset, float scaling_factor, size_t dimensions, size_t repetitions, std::mt19937_64 & rng ):
            offset( offset ),
            scaling_factor( scaling_factor ),
            repetitions( repetitions ),
            random_vectors( dimensions ),
            corrections() {
            for ( size_t vec_idx = 0; vec_idx < repetitions * K * LATTICE_DIMENSIONS; vec_idx++ ) {
                std::vector<float> dir = sample_random_normal_vector( dimensions, rng );
                rescale( dir, 1.0 / std::sqrt( LATTICE_DIMENSIONS ) );
                float sum = 0.0;
                for ( size_t i = 0; i < dimensions; i++ ) {
                    sum += dir[i];
                }
                random_vectors.push_back( dir.begin(), dir.end() );
                offsets.push_back( sample_random_01(rng) );
                corrections.push_back( offset / scaling_factor * sum );
            }
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( offset, scaling_factor, repetitions, random_vectors, offsets, corrections );
        }

        static constexpr size_t get_concatenations() {
            return K;
        }

        size_t get_repetitions() const {
            return repetitions;
        }

        void hash( typename Dataset::PointHandle point, std::vector<Value>& output ) const {
            // TODO: for simplicity, offset and rescale here
            output.clear();
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                Value cur;
                for ( size_t concat = 0; concat < K; concat++ ) {
                    auto prj = project( point, concat, rep );
                    auto decoded = decode_e8(prj);
                    int64_t code = to_int64(to_integer_coords(decoded));
                    cur.set( concat, code );
                }
                output.push_back( cur );
                cur = Value();
            }
        }

        float collision_probability( float distance ) const {
            // FIXME: rescale distance
            if (distance > panna::lattice_lsh::MAX_DISTANCE) {
                return 0.0;
            }
            size_t idx = std::floor( distance / panna::lattice_lsh::DISTANCE_STEP );
            if ( idx < panna::lattice_lsh::NUM_ESTIMATES ) {
                return panna::lattice_lsh::PROBABILITIES[idx];
            } else {
                return 0;
            }
        }
    };

    template <uint8_t K, typename Dataset, typename Distance>
    class LatticeLSHBuilder {
        float offset = 0.0;
        float scaling_factor = 0.0;
        size_t dimensions = 0;

    public:
        using Output = LatticeLSH<K, Dataset, Distance>;

        LatticeLSHBuilder() {
        }

        LatticeLSHBuilder( size_t dimensions ):
            offset( 0 ), scaling_factor( 0 ), dimensions( dimensions ) {
        }

        LatticeLSHBuilder( float offset, float scaling_factor, size_t dimensions ):
            offset( offset ), scaling_factor( scaling_factor ), dimensions( dimensions ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( offset, scaling_factor, dimensions );
        }

        void fit( Dataset& points ) {
            if ( scaling_factor != 0.0 ) {
                return;
            }
        }

        Output build( size_t repetitions ) const {
            expect( scaling_factor > 0 );
            return LatticeLSH<K, Dataset, Distance>( offset, scaling_factor, dimensions, repetitions );
        }

        std::string describe() const {
            std::stringstream sstream;
            sstream << "LatticeLSH(scaling=" << scaling_factor << ")";
            return sstream.str();
        }
    };

} // namespace panna

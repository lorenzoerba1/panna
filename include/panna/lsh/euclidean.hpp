#pragma once
#include <limits>
#include <random>
#include <vector>

#include "panna/expect.hpp"
#include "panna/linalg.hpp"
#include "panna/lsh/values.hpp"
#include "panna/rand.hpp"

namespace panna {

    // from https://en.cppreference.com/w/cpp/numeric/math/erfc
    static double normal_cdf( double x ) {
        return std::erfc( -x / std::sqrt( 2 ) ) / 2;
    }

    template <uint8_t K, typename Dataset>
    class E2LSHBuilder;

    template <uint8_t K, typename Dataset>
    class E2LSH {
    public:
        //! The datatype of the output
        using Value = BytewiseLshValue<K>;
        using Builder = E2LSHBuilder<K, Dataset>;

    private:
        float quantization_width;
        size_t repetitions;
        Dataset random_vectors;
        std::vector<float> offsets;

    public:
        E2LSH() {
        }

        E2LSH( float quantization_width, size_t dimensions, size_t repetitions ):
            quantization_width( quantization_width ),
            repetitions( repetitions ),
            random_vectors( dimensions ) {
            auto& rng = get_global_rng();
            std::uniform_real_distribution<float> uniform( 0.0, quantization_width );
            for ( size_t vec_idx = 0; vec_idx < repetitions * K; vec_idx++ ) {
                std::vector<float> dir = sample_random_normal_vector( dimensions );
                random_vectors.push_back( dir.begin(), dir.end() );
                offsets.push_back( uniform( rng ) );
            }
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( quantization_width, repetitions, random_vectors, offsets );
        }

        static constexpr size_t get_concatenations() {
            return K;
        }

        size_t get_repetitions() const {
            return repetitions;
        }

        void hash( typename Dataset::PointHandle point, std::vector<Value>& output ) const {
            output.clear();
            BytewiseLshValue<K> cur;
            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                for ( size_t concat = 0; concat < K; concat++ ) {
                    typename Dataset::PointHandle rand_vec = random_vectors[K * rep + concat];
                    float dotp = dot_product( point, rand_vec );
                    float quantized =
                        std::floor( ( dotp + offsets[K * rep + concat] ) / quantization_width );
                    int8_t code = static_cast<int8_t>( quantized );
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
                       ( 1.0 - std::exp( -r * r / ( 2.0 * distance * distance ) ) );
        }
    };

    template <uint8_t K, typename Dataset>
    class E2LSHBuilder {
        float quantization_width = 0.0;
        size_t dimensions = 0;

    public:
        using Output = E2LSH<K, Dataset>;

        E2LSHBuilder() {
        }

        E2LSHBuilder( size_t dimensions ): quantization_width( 0 ), dimensions( dimensions ) {
        }

        E2LSHBuilder( float quantization_width, size_t dimensions ):
            quantization_width( quantization_width ), dimensions( dimensions ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( quantization_width, dimensions );
        }

        void fit( Dataset& points ) {
            expect( quantization_width == 0.0 );
            using Vec      = std::vector<float>;
            Dataset random( dimensions );
            for ( size_t i = 0; i < 1000; i++ ) {
                std::vector<float> dir = sample_random_normal_vector( dimensions );
                random.push_back( dir.begin(), dir.end() );
            }

            float min = std::numeric_limits<float>::infinity();
            float max = -std::numeric_limits<float>::infinity();
            for ( size_t i = 0; i < random.size(); i++ ) {
                for ( size_t j = 0; j < points.size(); j++ ) {
                    float dotp = dot_product( random[i], points[j] );
                    if ( dotp < min ) {
                        min = dotp;
                    }
                    if ( dotp > max ) {
                        max = dotp;
                    }
                }
            }

            // This uses fewer than 8 bits per hash, but it makes the hashes go faster
            // TODO: we migh consider using only 4 bits per hash.
            quantization_width = ( max - min ) / 16;

            // Build 4 repetitions of K hashes and count the mean number of collisions in the dataset
            size_t collisions = 0;
            size_t rep = 4;
            size_t high_thresh = sqrt(points.size()) * 1.4;
            size_t low_thresh = sqrt(points.size()) * 0.8;
            size_t max_iters = 10;
            float incr_mul= 1.5f, decr_mul = 1.5f, r_low = 0, r_high = 0;

            for ( std::size_t iter = 0; iter < max_iters; ++iter ) {
            // Generate random vectors for the projections
            Dataset proj ( dimensions );
            for (std::size_t i = 0; i < rep * K; ++i) {
                Vec a = sample_random_normal_vector(dimensions);
                proj.push_back(a.begin(), a.end());
            }
            //Hash the points
            double total_collisions = 0.0;
            for (size_t rep_c = 0; rep_c < rep; ++rep_c)
            {
                std::unordered_map<std::string, size_t> buckets;
                buckets.reserve(points.size());

                for (size_t idx_point = 0; idx_point < points.size(); ++idx_point)        
                {
                    std::ostringstream key;
                    key.precision(0);
                    key.setf(std::ios::fixed);

                    for (std::size_t k = 0; k < K; ++k)
                    {
                        const auto& a = proj[rep_c * K + k];
                        float h = std::floor( (dot_product(a, points[idx_point]) /
                                                   quantization_width) );
                        key << h << '|';  // cheap concatenation
                    }
                    ++buckets[key.str()];
                }

                // Count the collisions
                for (const auto& [_, cnt] : buckets)
                    total_collisions += (cnt > 1) ? (cnt * (cnt - 1) / 2.0) : 0.0;
            }

            double mean_collisions = total_collisions / rep;
            std::cout << "E2LSH: mean collisions = " << mean_collisions << std::endl;

            if ( r_low != 0 && r_high != 0 && (mean_collisions > high_thresh || mean_collisions < low_thresh) ) {
                if ( mean_collisions > high_thresh ) {
                    r_high = quantization_width;
                }
                else {
                    r_low = quantization_width;
                }
                // Do binary search between the two values to find the optimal one
                quantization_width = (r_low + r_high) / 2.0;
            }
            // Adjust the quantization width based on the mean number of collisions
            else if (mean_collisions > high_thresh) {
                r_low = quantization_width; // save the last good value
                quantization_width /= decr_mul;      // too many collisions
                decr_mul = sqrt(decr_mul);               // decrease the step size
            }
            else if (mean_collisions < low_thresh) {
                r_high = quantization_width; // save the last good value
                quantization_width *= incr_mul;      // too few – buckets too small
                incr_mul = sqrt(incr_mul);               // increase the step size
            }
            else
                break;                           
        }                

            std::cout << "E2LSH: quantization width = " << quantization_width << std::endl;
        }

        Output build( size_t repetitions ) const {
            expect( quantization_width > 0 );
            return E2LSH<K, Dataset>( quantization_width, dimensions, repetitions );
        }

        std::string describe() const {
            std::stringstream sstream;
            sstream << "E2LSH(quantization=" << quantization_width << ")";
            return sstream.str();
        }
    };

} // namespace panna

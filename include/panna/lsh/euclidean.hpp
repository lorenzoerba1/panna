#pragma once
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include "panna/expect.hpp"
#include "panna/linalg.hpp"
#include "panna/lsh/values.hpp"
#include "panna/prefixmap.hpp"
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

            // Now we can adjust the quantization width so that there are
            // enough collisions at the lowest level, but not too many
            const size_t sample_repetitions = 4;
            std::optional<float> qw_lower = std::nullopt, qw_upper = std::nullopt;
            size_t high_thresh = sqrt( points.size() ) * 1.3;
            size_t low_thresh = sqrt( points.size() ) * 0.7;

            // Do a binary search to find the fight value
            while ( true ) {
                if ( qw_lower && qw_upper ) {
                    quantization_width = ( *qw_lower + *qw_upper ) / 2.0;
                }

                // instantiate some repetitions, and count the collisions
                std::vector<PrefixMap<typename Output::Value>> pmaps;
                pmaps.resize( sample_repetitions );
                Output hasher( quantization_width, dimensions, sample_repetitions );
                PrefixMap<typename Output::Value>::populate_from( pmaps, points, hasher );

                // Count the average pairwise collisions
                size_t collisions = 0;
                for ( PrefixMap<typename Output::Value>& pmap : pmaps ) {
                    PrefixMapCursor<typename Output::Value> cursor = pmap.create_cursor();
                    do {
                        auto ranges = cursor.get_ranges();
                        // when using the cursor like this we expect the second of the
                        // two returned ranges to be empty at all times
                        assert( ranges[1].first == ranges[1].second );
                        size_t bucket_size = ranges[0].second - ranges[0].first;
                        collisions += bucket_size * ( bucket_size - 1 ) / 2;
                    } while ( cursor.next_hash() );
                }
                float avg_collisions = ( (float)collisions ) / pmaps.size();

                if ( avg_collisions < low_thresh ) {
                    // the quantization width is too small
                    qw_lower = quantization_width;
                    quantization_width *= 2.0;
                } else if ( ( avg_collisions >
                              high_thresh ) && // too many collisions
                                               // the difference between bounds is not too small
                            ( qw_upper.value_or( 0.0 ) - qw_lower.value_or( 0.0 ) > 1e-7 ) ) {
                    qw_upper = quantization_width;
                    quantization_width /= 2.0;
                } else {
                    break;
                }
            }
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

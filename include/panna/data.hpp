#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "cereal/types/array.hpp"
#include "cereal/types/vector.hpp"
#include "panna/rand.hpp"

namespace panna {
    //! A chunk of a longer vector of 16bit integers.
    //! On C++17 this class is guaranteed to be aligned to the 256-bit boundary
    //! even when stored in a std::vector.
    struct alignas( 32 ) Int16Chunk {
        static constexpr size_t CHUNK_SIZE = 16;
        // int16_t chunk[CHUNK_SIZE];
        std::array<int16_t, CHUNK_SIZE> chunk;

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( chunk );
        }

        friend bool operator==( const Int16Chunk& a, const Int16Chunk& b ) {
            return a.chunk == b.chunk;
        }
    };
    static_assert( sizeof( Int16Chunk ) == 256 / 8, "int16 chunk should fill 256 bits" );
    static_assert( alignof( Int16Chunk ) == 256 / 8, "int16 chunk should align to 256 bits" );

    static constexpr int16_t to_16bit_fixed_point( float val ) {
        assert( val >= -1.0 && val <= 1.0 );

        val = std::min( val * ( 1 << 15 ), static_cast<float>( INT16_MAX ) );
        return static_cast<int16_t>( val );
    }

    static constexpr float from_16bit_fixed_point( int16_t val ) {
        return static_cast<float>( val ) / ( 1 << 15 );
    }

    struct UnitNormPointHandle {
        Int16Chunk const* chunks;
        size_t num_chunks;
        size_t dimensions;

        void into_vec( std::vector<float>& vec ) const {
            size_t i = 0;
            for ( size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++ ) {
                Int16Chunk chunk = chunks[chunk_idx];
                for ( size_t j = 0; j < Int16Chunk::CHUNK_SIZE; j++ ) {
                    vec[i] = from_16bit_fixed_point( chunk.chunk[j] );
                    i++;
                }
            }
        }
    };

    class UnitNormPoints {

        size_t dimensions = 0;
        size_t padding = 0;
        size_t chunks_per_point = 0;

        // Since C++17 the allocation of std::vector respects the alignment of
        // the template argument.
        std::vector<Int16Chunk> chunks;

    public:
        using PointHandle = UnitNormPointHandle;

        UnitNormPoints() {
        }

        UnitNormPoints( size_t dimensions ):
            dimensions( dimensions ),
            padding( ( Int16Chunk::CHUNK_SIZE - ( dimensions % Int16Chunk::CHUNK_SIZE ) ) %
                     Int16Chunk::CHUNK_SIZE ),
            chunks_per_point( std::ceil( ( (float)dimensions ) / Int16Chunk::CHUNK_SIZE ) ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( dimensions, padding, chunks_per_point, chunks );
        }

        friend bool operator==( const UnitNormPoints& a, const UnitNormPoints& b ) {
            return a.dimensions == b.dimensions && a.padding == b.padding && a.chunks_per_point &&
                   b.chunks_per_point && a.chunks == b.chunks;
        }

        void clear() {
            chunks.clear();
        }

        size_t get_padding() const {
            return padding;
        }

        size_t get_chunks_per_point() const {
            return chunks_per_point;
        }

        PointHandle operator[]( size_t i ) const {
            assert( i * chunks_per_point < chunks.size() );
            UnitNormPointHandle handle;
            handle.chunks = &chunks[i * chunks_per_point];
            handle.num_chunks = chunks_per_point;
            handle.dimensions = dimensions;
            return handle;
        }

        template <typename FloatIter>
        void push_back( FloatIter begin, FloatIter end ) {
            float sq_norm = 0.0;
            for ( FloatIter it = begin; it != end; it++ ) {
                float v = *it;
                sq_norm += v * v;
            }

            // prepare the space
            size_t base_chunk_idx = chunks.size();
            for ( size_t i = 0; i < chunks_per_point; i++ ) {
                chunks.emplace_back();
            }

            auto norm = std::sqrt( sq_norm );
            size_t i = 0;
            for ( FloatIter it = begin; it != end; it++ ) {
                float normalized = ( norm == 0.0 ) ? *it : *it / norm;
                chunks[base_chunk_idx + i / Int16Chunk::CHUNK_SIZE]
                    .chunk[i % Int16Chunk::CHUNK_SIZE] = to_16bit_fixed_point( normalized );
                i++;
            }
            for ( size_t i = dimensions; i < dimensions + padding; i++ ) {
                chunks[base_chunk_idx + i / Int16Chunk::CHUNK_SIZE]
                    .chunk[i % Int16Chunk::CHUNK_SIZE] = to_16bit_fixed_point( 0.0 );
            }
        }

        PointHandle push_back_random() {
            std::vector<float> values;
            for ( unsigned int i = 0; i < dimensions; i++ ) {
                values.push_back( sample_random_normal() );
            }

            push_back( values.begin(), values.end() );
            return operator[]( size() - 1 );
        }

        size_t size() const {
            return chunks.size() / chunks_per_point;
        }
    };

    struct NormedPointHandle {
        UnitNormPointHandle inner;
        float sq_norm;

        float squared_norm() const {
            return sq_norm;
        }
    };

    class NormedPoints {
        size_t dimensions = 0;
        UnitNormPoints normalized_points;
        std::vector<float> squared_norms;

    public:
        using PointHandle = NormedPointHandle;

        NormedPoints() {
        }

        NormedPoints( size_t dimensions ):
            dimensions( dimensions ), normalized_points( dimensions ) {
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( dimensions, normalized_points, squared_norms );
        }

        friend bool operator==( const NormedPoints& a, const NormedPoints& b ) {
            return a.dimensions == b.dimensions && a.normalized_points == b.normalized_points &&
                   a.squared_norms == b.squared_norms;
        }

        void clear() {
            normalized_points.clear();
            squared_norms.clear();
        }

        PointHandle operator[]( size_t i ) const {
            PointHandle handle;
            handle.inner = normalized_points[i];
            handle.sq_norm = squared_norms[i];
            return handle;
        }

        template <typename FloatIter>
        void push_back( FloatIter begin, FloatIter end ) {
            float sq_norm = 0.0;
            for ( FloatIter it = begin; it != end; it++ ) {
                float v = *it;
                sq_norm += v * v;
            }
            normalized_points.push_back( begin, end );
            squared_norms.push_back( sq_norm );
        }

        PointHandle push_back_random() {
            std::vector<float> values;
            for ( unsigned int i = 0; i < dimensions; i++ ) {
                values.push_back( sample_random_normal() );
            }

            push_back( values.begin(), values.end() );
            return operator[]( size() - 1 );
        }

        size_t size() const {
            return squared_norms.size();
        }
    };

    class SparseSetHandle {
        size_t set_size;
        const uint32_t* tokens;

        friend class SparseSets;

    public:
        const uint32_t* begin() const {
            return tokens;
        }
        const uint32_t* end() const {
            return tokens + set_size;
        }

        size_t size() const {
            return set_size;
        }

        size_t intersection_size( SparseSetHandle other ) const {
            size_t asize = set_size;
            size_t bsize = other.set_size;

            size_t res = 0;
            size_t aidx = 0;
            size_t bidx = 0;

            while ( aidx < asize && bidx < bsize ) {
                if ( tokens[aidx] == other.tokens[bidx] ) {
                    res++;
                    aidx++;
                    bidx++;
                } else if ( tokens[aidx] < other.tokens[bidx] ) {
                    aidx++;
                } else {
                    bidx++;
                }
            }

            return res;
        }
    };

    class SparseSets {
        size_t dimensions;
        std::vector<uint32_t> set_data;
        std::vector<size_t> starts;

    public:
        using PointHandle = SparseSetHandle;

        SparseSets( size_t dimensions ): dimensions( dimensions ) {
            starts.push_back( set_data.size() );
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( dimensions, set_data, starts );
        }

        void clear() {
            set_data.clear();
            starts.clear();
            starts.push_back( set_data.size() );
        }

        PointHandle operator[]( size_t i ) const {
            PointHandle handle;
            handle.tokens = &set_data[starts[i]];
            handle.set_size = starts[i + 1] - starts[i];
            return handle;
        }

        void push_back( std::vector<uint32_t>& set ) {
            std::sort( set.begin(), set.end() );
            for ( uint32_t token : set ) {
                if ( token > dimensions ) {
                    throw std::invalid_argument( "token outside of the universe" );
                }
                set_data.push_back( token );
            }
            starts.push_back( set_data.size() );
        }

        PointHandle push_back_random() {
            auto& rng = get_global_rng();
            std::bernoulli_distribution coin( 0.3 );
            std::vector<uint32_t> values;

            for ( unsigned int i = 0; i < dimensions; i++ ) {
                if ( coin( rng ) ) {
                    //
                    values.push_back( i );
                }
            }

            push_back( values );
            return operator[]( size() - 1 );
        }

        size_t size() const {
            return starts.size() - 1;
        }
    };
} // namespace panna

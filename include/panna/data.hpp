#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "panna/rand.hpp"

namespace panna {
    //! A dummy storage for points, only for testing purposes.
    class DummyPoints {
        using PointHandle = std::vector<float>;

        size_t dimensions;
        std::vector<std::vector<float>> points;

    public:
        DummyPoints( size_t dimensions ): dimensions( dimensions ) {}

        void push_back_random_normal() {
            auto& rng = panna::get_global_rng();
            std::normal_distribution<float> normal_distribution( 0.0, 1.0 );

            std::vector<float> values;
            for ( unsigned int i = 0; i < dimensions; i++ ) {
                values.push_back( normal_distribution( rng ) );
            }

            points.push_back( values );
        }

        void push_back( std::vector<float> v ) { points.push_back( v ); }

        size_t size() const { return points.size(); }

        PointHandle operator[]( size_t i ) {
            assert( i < points.size() );
            return points[i];
        }
    };

    //! A chunk of a longer vector of 16bit integers.
    //! On C++17 this class is guaranteed to be aligned to the 256-bit boundary
    //! even when stored in a std::vector.
    struct alignas( 32 ) Int16Chunk {
        static constexpr size_t CHUNK_SIZE = 16;
        int16_t chunk[CHUNK_SIZE];
    };
    static_assert( sizeof( Int16Chunk ) == 256 / 8,
                   "int16 chunk should fill 256 bits" );
    static_assert( alignof( Int16Chunk ) == 256 / 8,
                   "int16 chunk should align to 256 bits" );

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
    };

    class UnitNormPoints {

        size_t dimensions;
        size_t padding;
        size_t chunks_per_point;

        // Since C++17 the allocation of std::vector respects the alignment of
        // the template argument.
        std::vector<Int16Chunk> chunks;

    public:
        using PointHandle = UnitNormPointHandle;

        UnitNormPoints( size_t dimensions ):
            dimensions( dimensions ),
            padding( ( Int16Chunk::CHUNK_SIZE -
                       ( dimensions % Int16Chunk::CHUNK_SIZE ) ) %
                     Int16Chunk::CHUNK_SIZE ),
            chunks_per_point(
                std::ceil( ( (float)dimensions ) / Int16Chunk::CHUNK_SIZE ) ) {}

        size_t get_padding() const { return padding; }

        size_t get_chunks_per_point() const { return chunks_per_point; }

        PointHandle operator[]( size_t i ) const {
            assert( i * chunks_per_point < chunks.size() );
            UnitNormPointHandle handle;
            handle.chunks = &chunks[i * chunks_per_point];
            handle.num_chunks = chunks_per_point;
            return handle;
        }

        template <typename FloatVec>
        void push_back( FloatVec& vec ) {
            float sq_norm = 0.0;
            for ( size_t i = 0; i < dimensions; i++ ) {
                float v = vec[i];
                sq_norm += v * v;
            }

            // prepare the space
            size_t base_chunk_idx = chunks.size();
            for ( size_t i = 0; i < chunks_per_point; i++ ) {
                chunks.emplace_back();
            }

            auto norm = std::sqrt( sq_norm );
            for ( size_t i = 0; i < dimensions; i++ ) {
                float normalized = ( norm == 0.0 ) ? vec[i] : vec[i] / norm;
                chunks[base_chunk_idx + i / Int16Chunk::CHUNK_SIZE]
                    .chunk[i % Int16Chunk::CHUNK_SIZE] =
                    to_16bit_fixed_point( normalized );
            }
            for ( size_t i = dimensions; i < dimensions + padding; i++ ) {
                chunks[base_chunk_idx + i / Int16Chunk::CHUNK_SIZE]
                    .chunk[i % Int16Chunk::CHUNK_SIZE] =
                    to_16bit_fixed_point( 0.0 );
            }
        }

        PointHandle push_back_random_normal() {
            auto& rng = panna::get_global_rng();
            std::normal_distribution<float> normal_distribution( 0.0, 1.0 );

            std::vector<float> values;
            for ( unsigned int i = 0; i < dimensions; i++ ) {
                values.push_back( normal_distribution( rng ) );
            }

            push_back( values );
            return operator[]( size() - 1 );
        }

        size_t size() const { return chunks.size() / chunks_per_point; }
    };

    struct NormedPointHandle {
        UnitNormPointHandle inner;
        float sq_norm;

        float squared_norm() const { return sq_norm; }
    };

    class NormedPoints {
        size_t dimensions;
        UnitNormPoints normalized_points;
        std::vector<float> squared_norms;

    public:
        using PointHandle = NormedPointHandle;

        NormedPoints( size_t dimensions ):
            dimensions( dimensions ), normalized_points( dimensions ) {}

        PointHandle operator[]( size_t i ) const {
            PointHandle handle;
            handle.inner = normalized_points[i];
            handle.sq_norm = squared_norms[i];
            return handle;
        }

        template <typename FloatVec>
        void push_back( FloatVec& vec ) {
            float sq_norm = 0.0;
            for ( size_t i = 0; i < dimensions; i++ ) {
                float v = vec[i];
                sq_norm += v * v;
            }
            normalized_points.push_back( vec );
            squared_norms.push_back( sq_norm );
        }

        PointHandle push_back_random_normal() {
            auto& rng = panna::get_global_rng();
            std::normal_distribution<float> normal_distribution( 0.0, 1.0 );

            std::vector<float> values;
            for ( unsigned int i = 0; i < dimensions; i++ ) {
                values.push_back( normal_distribution( rng ) );
            }

            push_back( values );
            return operator[]( size() - 1 );
        }

        size_t size() const { return squared_norms.size(); }
    };
} // namespace panna

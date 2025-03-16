#pragma once

#include <cassert>
#include <cstddef>
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
            auto rng = panna::get_global_rng();
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

    //! A chunk of a longer vector of unsigned 16bit integers.
    //! On C++17 this class is guaranteed to be aligned to the 256-bit boundary
    //! even when stored in a std::vector.
    struct alignas( 32 ) Uint16Chunk {
        uint16_t chunk[16];
    };
    static_assert( sizeof( Uint16Chunk ) == 256 / 8,
                   "Uint16 chunk should fill 256 bits" );
    static_assert( alignof( Uint16Chunk ) == 256 / 8,
                   "Uint16 chunk should align to 256 bits" );

    struct UnitNormPointHandle {
        Uint16Chunk* chunks;
        size_t num_chunks;
    };

    class UnitNormPoints {
        using PointHandle = UnitNormPointHandle;

        size_t dimensions;
        size_t padding;
        // how many chunks are
        size_t chunks_per_point;

        // Since C++17 the allocation of std::vector respects the alignment of
        // the template argument.
        std::vector<Uint16Chunk> chunks;

        PointHandle operator[]( size_t i ) {
            assert( i * chunks_per_point < chunks.size() );
            UnitNormPointHandle handle;
            handle.chunks = &chunks[i * chunks_per_point];
            handle.num_chunks = chunks_per_point;
            return handle;
        }
    };
} // namespace panna

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "omp.h"
#include "panna/expect.hpp"
// Tree import
#include "panna/dsu.hpp"
namespace panna {

    //! Returns the index of the first element strictly larger than the `needle`, starting from
    //! position `from`. "Larger" is defined in terms of the `prefix_less` method.
    template <typename T>
    static inline size_t
    gallop_right( const std::vector<T>& hashes, T needle, size_t from, uint8_t prefix ) {
        // first do an exponential search
        size_t offset = 1;
        size_t lower = from;
        size_t upper = from + offset;
        while ( upper < hashes.size() && !needle.prefix_less( hashes[upper], prefix ) ) {
            offset *= 2;
            lower = upper;
            upper = std::min( from + offset, hashes.size() );
        }
        // now do a binary search
        size_t half = upper - lower;
        while ( half != 0 ) {
            half /= 2;
            size_t mid = lower + half;
            if ( mid >= hashes.size() ) {
                return hashes.size();
            }
            lower = ( !needle.prefix_less( hashes[mid], prefix ) ) ? ( mid + 1 ) : lower;
        }
        return lower;
    }

    //! Returns the smallest index before `to` (excluded) of the hash with the same prefix as the
    //! given needle. If there is no such index, returns `to`.
    template <typename T>
    static inline size_t
    gallop_left( const std::vector<T>& hashes, T needle, size_t to, uint8_t prefix ) {
        if ( to == 0 ) {
            return to;
        }
        // first do an exponential search
        size_t offset = 1;
        size_t upper = to;
        size_t lower = to - offset;
        while ( lower > 0 && hashes[lower].prefix_eq( needle, prefix ) ) {
            upper = lower;
            lower = ( to > offset ) ? to - offset : 0;
            offset *= 2;
        }
        assert( upper <= hashes.size() );
        assert( lower < upper );

        // now do a binary search
        size_t half = upper - lower;
        while ( half != 0 ) {
            half /= 2;
            size_t mid = lower + half;
            if ( mid > upper ) {
                return to;
            }
            assert( mid < hashes.size() );
            lower = ( hashes[mid].prefix_less( needle, prefix ) ) ? ( mid + 1 ) : lower;
        }
        assert( lower < hashes.size() );
        if ( !needle.prefix_eq( hashes[lower], prefix ) ) {
            return to;
        }
        return lower;
    }

    struct CombinationsIndex {
        size_t begin;
        size_t end;
        size_t i;
        size_t j;

        CombinationsIndex(): CombinationsIndex(0, 0) {}

        CombinationsIndex( size_t begin, size_t end ):
            begin( begin ), end( end ), i( begin ), j( begin ) {
        }

        bool next() {
            j += 1;
            if ( j >= end ) {
                i += 1;
                j = i + 1;
                if ( i >= end - 1 ) {
                    return false;
                }
            }
            return true;
        }
    };

    template <typename THashValue>
    class PrefixMapCursor {
    private:
        const std::vector<THashValue>& hashes;
        const std::vector<uint32_t>& indices;
        THashValue hash;
        uint8_t prefix_length;
        size_t prev_range_start;
        size_t prev_range_end;
        size_t range_start;
        size_t range_end;

    public:
        using Iter = const uint32_t*;

        PrefixMapCursor( THashValue hash,
                         const std::vector<THashValue>& hashes,
                         const std::vector<uint32_t>& indices ):
            hashes( hashes ),
            indices( indices ),
            hash( hash ),
            prefix_length( THashValue::get_concatenations() ) {
            assert( hashes.size() > 0 );
            assert( std::is_sorted( hashes.begin(), hashes.end() ) );
            assert(
                std::is_sorted( hashes.begin(), hashes.end(), [&]( const auto& a, const auto& b ) {
                    return a.prefix_less( b, prefix_length );
                } ) );
            update_range_start();
            update_range_end();
            assert( range_start <= range_end );

            // set the previous range to the empty range
            prev_range_start = range_end;
            prev_range_end = range_end;
        }

        //! Builds the cursor, placing it at the smallest hash value in the provided list
        PrefixMapCursor( const std::vector<THashValue>& hashes,
                         const std::vector<uint32_t>& indices ):
            PrefixMapCursor( hashes[0], hashes, indices ) {
        }

        size_t current_prefix() const {
            return prefix_length;
        }

        // Find the first index such that the prefix is >= the given hash.
        // In other words, in the first part the hashes are all < the given hash.
        void update_range_start() {
            range_start = std::distance(
                hashes.begin(),
                std::partition_point( hashes.begin(), hashes.end(), [&]( const auto& h ) {
                    return h.prefix_less( hash, prefix_length );
                } ) );
        }

        // Find the first index such that the prefix is > the given hash (**strictly** larger).
        // In other words, in the first part the hashes are all <= the given hash.
        void update_range_end() {
            range_end = std::distance(
                hashes.begin(),
                std::partition_point( hashes.begin(), hashes.end(), [&]( const auto& h ) {
                    return !hash.prefix_less( h, prefix_length );
                } ) );
        }

        // Shortens the prefix by one, and adjusts ranges accordingly
        void shorten_prefix( size_t new_prefix ) {
            if ( new_prefix > prefix_length ) {
                throw std::invalid_argument( "the new prefix must be shorter than the old one" );
            }
            if ( prefix_length == 0 ) {
                throw std::invalid_argument( "the new prefix must be positive" );
            };
            if ( new_prefix == prefix_length ) {
                // nothing to do
                return;
            }
            prev_range_start = range_start;
            prev_range_end = range_end;
            prefix_length = new_prefix;
            update_range_start();
            update_range_end();

            if ( range_start < range_end && range_start > 0 ) {
                expect( hashes[range_start - 1].prefix_less( hash, prefix_length ) );
            }

            assert( range_start <= range_end );
        }

        //! Moves the reference hash of the cursor to the next one in the `hashes` array.
        //! Returns `false` if we reached the end of the available hashes,
        //! and true otherwise
        bool next_hash() {
            // if ( range_end == hashes.size() ) {
            //     return false;
            // }
            // hash = hashes[range_end];
            // // We don't need to search for the start
            // range_start = range_end;
            // // but we do need to search for the end
            // range_end = first_lt_pos( hash, prefix_length );
            // // The previous start and end are the bounds of the ranges of hashes with a prefix
            // // equality at the old prefix
            // if ( prefix_length == THashValue::get_concatenations() ) {
            //     prev_range_start = range_end;
            //     prev_range_end = range_end;
            // } else {
            //     prev_range_start = first_le_pos( hash, prefix_length + 1 );
            //     prev_range_end = first_lt_pos( hash, prefix_length + 1 );
            // }
            // return true;
            throw "boom";
        }

        std::array<std::pair<size_t, size_t>, 2> get_ranges() const {
            return { std::make_pair( range_start, prev_range_start ),
                     std::make_pair( prev_range_end, range_end ) };
        }

        std::array<std::pair<Iter, Iter>, 2> get_indices() const {
            auto ranges = get_ranges();
            return {
                std::make_pair( &indices[ranges[0].first], &indices[ranges[0].second] ),
                std::make_pair( &indices[ranges[1].first], &indices[ranges[1].second] ),
            };
        }

        // Fill the given output buffer, without changing its capacity.
        void fill_pairs_buffer( std::vector<std::pair<uint32_t, uint32_t>>& buffer ) {
            size_t max_num = buffer.capacity();
            while ( buffer.size() < max_num ) {
            }
        }
    };

    template <typename THashValue>
    class PairPrefixMapCursorNew {
    private:
        const std::vector<THashValue>& hashes;
        const std::vector<uint32_t>& indices;
        THashValue hash;
        size_t range_start;
        size_t range_end;
        uint8_t prefix_length;
        std::optional<uint8_t> prev_prefix_length;
        CombinationsIndex idx;

        // Find the first index such that the prefix is >= the given hash.
        size_t first_ge_pos( THashValue hash, uint8_t prefix ) {
            return std::distance(
                hashes.begin(),
                std::partition_point( hashes.begin(), hashes.end(), [&]( const auto& h ) {
                    return h.prefix_less( hash, prefix );
                } ) );
        }

        // Find the first index such that the prefix is > the given hash (**strictly** larger).
        size_t first_gt_pos( THashValue hash, uint8_t prefix ) {
            return std::distance(
                hashes.begin(),
                std::partition_point( hashes.begin(), hashes.end(), [&]( const auto& h ) {
                    return !hash.prefix_less( h, prefix );
                } ) );
        }

        //! Moves the reference hash of the cursor to the next one in the `hashes` array.
        //! Returns `false` if we reached the end of the available hashes,
        //! and true otherwise
        bool next_hash() {
            if ( range_end == hashes.size() ) {
                return false;
            }
            hash = hashes[range_end];
            // We don't need to search for the start
            range_start = range_end;
            // but we do need to search for the end
            range_end = first_gt_pos( hash, prefix_length );
            idx = CombinationsIndex( range_start, range_end );
            return true;
        }

    public:
        PairPrefixMapCursorNew( const std::vector<THashValue>& hashes,
                                const std::vector<uint32_t>& indices,
                                uint8_t prefix ):
            PairPrefixMapCursorNew( hashes, indices, prefix, std::nullopt ) {
        }

        PairPrefixMapCursorNew( const std::vector<THashValue>& hashes,
                                const std::vector<uint32_t>& indices,
                                uint8_t prefix,
                                std::optional<uint8_t> prev_prefix ):
            hashes( hashes ),
            indices( indices ),
            prefix_length( prefix ),
            prev_prefix_length( prev_prefix ) {

            assert( hashes.size() > 0 );
            assert( std::is_sorted( hashes.begin(), hashes.end() ) );

            hash = hashes[0];
            range_start = 0;
            range_end = first_gt_pos( hash, prefix );
            idx = CombinationsIndex( range_start, range_end );
        }

        //! fill the given buffer of pairs, without changing its capacity
        void fill_pairs_buffer( std::vector<std::tuple<uint32_t, uint32_t, float>>& buffer ) {
            size_t max_num = buffer.capacity();
            buffer.clear();
            while ( buffer.size() < max_num ) {
                if ( !idx.next() ) {
                    // we exhausted the bucket, move to the next one
                    if ( !next_hash() ) {
                        // there are no more buckets
                        break;
                    }
                }
                expect( idx.i < indices.size() );
                expect( idx.j < indices.size() );
                // check if we had a collision at the previous prefix
                if ( prev_prefix_length &&
                     hashes[idx.i].prefix_eq( hashes[idx.j], *prev_prefix_length ) ) {
                    // skip the pair, in this case
                    continue;
                }
                buffer.emplace_back(
                    indices[idx.i], indices[idx.j], std::numeric_limits<float>::infinity() );
            }
        }

        //! The total number of collisions, _including_ the ones on longer prefixes
        size_t total_collisions() {
            size_t cnt = 0;
            while ( true ) {
                size_t bucket_size = range_end - range_start;
                cnt += (bucket_size - 1) * bucket_size / 2;
                if ( !next_hash() ) {
                    // there are no more buckets
                    break;
                }
            }
            return cnt;
        }
    };

    template <typename THashValue>
    class PairPrefixMapCursor {
    private:
        const std::vector<THashValue>& hashes;
        const std::vector<uint32_t>& indices;
        THashValue current_hash;
        size_t current_index;
        size_t current_comparison;
        size_t range_start;
        size_t range_end;
        uint8_t prefix_length;

    public:
        using Iter = const uint32_t*;

        PairPrefixMapCursor( const std::vector<THashValue>& hashes,
                             const std::vector<uint32_t>& indices ):
            hashes( hashes ),
            indices( indices ),
            prefix_length( THashValue::get_concatenations() ) {
            assert( hashes.size() > 0 );
            assert( std::is_sorted( hashes.begin(), hashes.end() ) );

            range_start = range_end = 0;
            current_index = current_comparison = 0;
            current_hash = hashes[0];
        }

        // Shortens the prefix by one
        void shorten_prefix( size_t new_prefix ) {
            if ( new_prefix > prefix_length ) {
                throw std::invalid_argument( "the new prefix must be shorter than the old one" );
            }
            if ( prefix_length == 0 ) {
                throw std::invalid_argument( "the new prefix must be positive" );
            };
            if ( new_prefix == prefix_length ) {
                // nothing to do
                return;
            }
            prefix_length = new_prefix;
            current_index = current_comparison = 0;
            current_hash = hashes[0];
        }

        // Find the first index such that the prefix is >= the given hash.
        // In other words, in the first part the hashes are all < the given hash.
        void update_range_start() {
            range_start = std::distance(
                hashes.begin(),
                std::partition_point( hashes.begin(), hashes.end(), [&]( const auto& h ) {
                    return h.prefix_less( current_hash, prefix_length );
                } ) );
        }

        // Find the first index such that the prefix is > the given hash (**strictly** larger).
        // In other words, in the first part the hashes are all <= the given hash.
        void update_range_end() {
            range_end = std::distance(
                hashes.begin(),
                std::partition_point( hashes.begin(), hashes.end(), [&]( const auto& h ) {
                    return !current_hash.prefix_less( h, prefix_length );
                } ) );
        }

        std::pair<size_t, bool> next( std::vector<std::pair<Iter, Iter>>& scratch_space ) {
            // Setup
            size_t collisions = 0;
            bool continue_cycle = false;

            while ( range_end < hashes.size() ) {
                // Update the range
                update_range_start();
                update_range_end();
                for ( size_t current = range_start; current < range_end; current++ ) {
                    for ( size_t next = current + 1; next < range_end; next++ ) {

                        scratch_space.emplace_back( &indices[current], &indices[next] );
                        collisions++;
                    }
                }
                // Switch to the next hash
                current_hash = hashes[range_end];
            }

            return std::make_pair( collisions, continue_cycle );
        }

        std::tuple<Iter, Iter, std::vector<std::pair<Iter, Iter>>, bool> next_filter() {
            update_range_start();
            update_range_end();
            std::vector<std::pair<Iter, Iter>> split_ranges;

            current_hash = hashes[range_end];
            bool continue_cycle = true;
            if ( range_end >= hashes.size() ) {
                continue_cycle = false;
            }
            // If we are at the beginning, we have to compare everything in the range
            if ( prefix_length == THashValue::get_concatenations() ) {
                split_ranges.emplace_back( &indices[range_start], &indices[range_end] );
                return std::make_tuple( &indices[range_start], &indices[range_end], split_ranges, continue_cycle );
            }
            // Find the portions of the indices that are new
            // since we reduced the prefix, there is a part of comparisons that we already did
            Iter current_range_start = &indices[range_start];
            for (size_t i = range_start + 1; i < range_end; ++i) {
                // A split happens when the prefix of the current hash differs from the previous one
                if (hashes[i - 1].prefix_less(hashes[i], prefix_length + 1)) {
                    split_ranges.emplace_back(current_range_start, &indices[i]);
                    current_range_start = &indices[i];
                }
            }
            if (current_range_start != &indices[range_end]) {
                split_ranges.emplace_back(current_range_start, &indices[range_end]);
            }

            return std::make_tuple( &indices[range_start], &indices[range_end], split_ranges, continue_cycle );
        }
    };

    // A PrefixMap stores all inserted values in sorted order by their hash codes.
    //
    // This allows querying all values that share a common prefix. The length of the prefix
    // can be decreased to look at a larger set of values. When the prefix is decreased,
    // previously queried values are not queried again.
    template <typename THashValue>
    class PrefixMap {
        using HashedVecIdx = std::pair<uint32_t, THashValue>;

        // contents
        std::vector<uint32_t> indices;
        std::vector<THashValue> hashes;
        // Scratch space for use when rebuilding. The length and capacity is set to 0 otherwise.
        std::vector<std::vector<HashedVecIdx>> parallel_rebuilding_data;

    public:
        PrefixMap() {
            // Ensure that the map can be queried even if nothing is inserted.
            rebuild();
            auto max_threads = omp_get_max_threads();
            parallel_rebuilding_data.resize( max_threads );
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( indices, hashes );
        }

        size_t memory_usage() const {
            size_t total_size = sizeof( *this );
            total_size += indices.size() * sizeof( uint32_t );
            total_size += hashes.size() * sizeof( THashValue );
            for ( const auto& rd : parallel_rebuilding_data ) {
                total_size += rd.capacity() * sizeof( HashedVecIdx );
            }
            return total_size;
        }

        friend bool operator==( const PrefixMap<THashValue>& a, const PrefixMap<THashValue>& b ) {
            return a.indices == b.indices && a.hashes == b.hashes &&
                   a.parallel_rebuilding_data == b.parallel_rebuilding_data;
        }

        // Add a hash value, and associated index, to be included next time rebuild is called.
        void insert( int tid, uint32_t idx, THashValue hash_value ) {
            parallel_rebuilding_data[tid].push_back( { idx, hash_value } );
        }

        template <typename Dataset, typename Hasher>
        static void populate_from( std::vector<PrefixMap<THashValue>>& prefix_maps,
                                   Dataset& points,
                                   Hasher& hasher ) {
            // this vector is reused across hashing calls
            std::vector<typename Hasher::Value> hashes;

#pragma omp parallel for private( hashes )
            for ( size_t i = 0; i < points.size(); i++ ) {
                auto tid = omp_get_thread_num();
                hasher.hash( points[i], hashes );
                for ( size_t rep = 0; rep < prefix_maps.size(); rep++ ) {
                    prefix_maps[rep].insert( tid, i, hashes[rep] );
                }
            }

#pragma omp parallel for
            for ( size_t rep = 0; rep < prefix_maps.size(); rep++ ) {
                prefix_maps[rep].rebuild();
            }
        }

        // Reserve the correct amount of memory before inserting.
        void reserve( size_t size ) {
            for ( auto& rd : parallel_rebuilding_data ) {
                rd.reserve( size / parallel_rebuilding_data.size() );
            }
        }

        void rebuild() {
            size_t rebuilding_data_size = 0;
            for ( auto& rd : parallel_rebuilding_data ) {
                rebuilding_data_size += rd.size();
            }

            std::vector<std::pair<THashValue, uint32_t>> tmp;

            if ( hashes.size() != 0 ) {
                // Move data to temporary vector for sorting.
                for ( size_t i = 0; i < hashes.size(); i++ ) {
                    tmp.push_back( std::make_pair( hashes[i], indices[i] ) );
                }
            }
            for ( auto& rebuilding_data : parallel_rebuilding_data ) {
                for ( auto pair : rebuilding_data ) {
                    tmp.push_back( std::make_pair( pair.second, pair.first ) );
                }
            }

            // OPTIMIZE: use radix sort?
            std::sort( tmp.begin(), tmp.end() );

            indices.clear();
            indices.reserve( tmp.size() );
            hashes.clear();
            hashes.reserve( tmp.size() );

            for ( size_t i = 0; i < tmp.size(); i++ ) {
                hashes.push_back( tmp[i].first );
                indices.push_back( tmp[i].second );
            }
            assert( std::is_sorted( hashes.begin(), hashes.end() ) );

            for ( auto& rd : parallel_rebuilding_data ) {
                rd.clear();
                rd.shrink_to_fit();
            }
        }

        PrefixMapCursor<THashValue> create_cursor( THashValue hash ) const {
            return PrefixMapCursor<THashValue>( hash, hashes, indices );
        }

        PrefixMapCursor<THashValue> create_cursor() const {
            return PrefixMapCursor<THashValue>( hashes, indices );
        }

        PairPrefixMapCursor<THashValue> create_pair_cursor() const {
            return PairPrefixMapCursor<THashValue>( hashes, indices );
        }

        PairPrefixMapCursorNew<THashValue>
        create_pair_cursor_new( uint8_t prefix, std::optional<uint8_t> prev_prefix ) const {
            return PairPrefixMapCursorNew<THashValue>( hashes, indices, prefix, prev_prefix );
        }
        THashValue hash_for( size_t idx ) const {
            auto pos =
                std::distance( indices.begin(), std::find( indices.begin(), indices.end(), idx ) );
            expect( pos < hashes.size() );
            return hashes[pos];
        }
    };
} // namespace panna

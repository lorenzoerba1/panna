#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>

#include "panna/lsh/predicates.hpp"
#include "panna/prefixmap.hpp"

namespace panna {
    template <typename Dataset, typename Hasher, typename Distance>
    class Index {
        using PointHandle = typename Dataset::PointHandle;
        using THashValue = typename Hasher::Value;

        // The actual data points
        Dataset dataset;
        // Contains either one or zero points to be used
        // as the current query. This is mostly for convenience,
        // since doing this way we have that the query is formatted
        // in the same way as the data.
        Dataset current_query;
        // Hash tables used by LSH.
        std::vector<PrefixMap<THashValue>> lsh_maps;
        // How to hash the points
        Hasher hasher;

        size_t hashed_points = 0;

    public:
        template <typename HasherBuilder>
        Index( size_t dimensions, HasherBuilder builder, size_t repetitions ):
            dataset( dimensions ),
            current_query( dimensions ),
            hasher( builder.build( repetitions ) ),
            hashed_points( 0 ) {
            lsh_maps.resize( repetitions );
        }

        template <typename InputPoint>
        void insert( InputPoint& point ) {
            dataset.push_back( point );
        }

        void rebuild() {
            std::vector<THashValue> hashes;
#pragma omp parallel for private( hashes )
            for ( size_t i = hashed_points; i < dataset.size(); i++ ) {
                auto tid = omp_get_thread_num();
                hasher.hash( dataset[i], hashes );
                for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                    lsh_maps[rep].insert( tid, i, hashes[rep] );
                }
            }

#pragma omp parallel for
            for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                lsh_maps[rep].rebuild();
            }

            hashed_points = dataset.size();
        }

        template <typename InputPoint>
        void search_brute_force( InputPoint& query,
                                 size_t k,
                                 std::vector<std::pair<float, uint32_t>>& output ) {
            auto timer = std::chrono::steady_clock::now();
            current_query.clear();
            current_query.push_back( query );

            std::priority_queue<std::pair<float, uint32_t>> top;

            PointHandle q = current_query[0];

            for ( size_t i = 0; i < dataset.size(); i++ ) {
                float dist = Distance::compute( q, dataset[i] );
                top.emplace( dist, i );
                while ( top.size() > k ) {
                    top.pop();
                }
            }

            output.clear();
            while ( top.size() > 0 ) {
                output.push_back( top.top() );
                top.pop();
            }
            std::sort( output.begin(), output.end() );
            auto elapsed = std::chrono::steady_clock::now() - timer;
            dbg( std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count() );
        }

        template <typename InputPoint>
        void search( InputPoint& query,
                     size_t k,
                     float delta,
                     std::vector<std::pair<float, uint32_t>>& output ) {
            auto timer = std::chrono::steady_clock::now();
            size_t collisions = 0;
            // Setup
            output.clear();
            current_query.clear();
            current_query.push_back( query );
            PointHandle q = current_query[0];

            // FIXME: remove this allocation
            std::vector<typename Hasher::Value> q_hashes;
            hasher.hash( q, q_hashes );

            // FIXME: remove this allocation
            std::vector<PrefixMapCursor<typename Hasher::Value>> cursors;
            for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                cursors.push_back( lsh_maps[rep].create_cursor( q_hashes[rep] ) );
            }

            // Search
            bool stop = false;
            size_t max_concat = hasher.get_concatenations();
            for ( size_t concat = max_concat; concat > 0; concat-- ) {
                if ( stop ) {
                    break;
                }
                for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                    cursors[rep].shorten_prefix( concat );
                    for ( auto range : cursors[rep].get_indices() ) {
                        for ( const uint32_t* it = range.first; it != range.second; it++ ) {
                            PointHandle x = dataset[*it];
                            float dist = Distance::compute( q, x );
                            collisions++;
                            if ( std::find( output.begin(),
                                            output.end(),
                                            std::make_pair( dist, *it ) ) == output.end() ) {
                                output.push_back( std::make_pair( dist, *it ) );
                                std::push_heap( output.begin(), output.end() );
                                while ( output.size() > k ) {
                                    std::pop_heap( output.begin(), output.end() );
                                    output.pop_back();
                                }
                            }
                        }
                    }

                    // check stopping condition
                    if ( output.size() == k ) {
                        float topdist = output.back().first;
                        float fp = failure_probability(
                            hasher, topdist, concat, rep + 1, lsh_maps.size() );
                        if ( fp <= delta ) {
                            dbg( concat, rep, topdist, collisions, fp );
                            stop = true;
                            break;
                        }
                    }
                }
                float topdist = (output.size() > 0)? output.back().first : std::numeric_limits<float>::infinity();
                dbg( concat, collisions, topdist );
            }
            auto elapsed = std::chrono::steady_clock::now() - timer;
            dbg( std::chrono::duration_cast<std::chrono::microseconds>( elapsed ).count() );
        }
    };
} // namespace panna

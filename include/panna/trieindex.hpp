#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <queue>
#include <stdexcept>

#include "dbg.h"

#include "cereal/archives/binary.hpp"
#include "panna/lsh/predicates.hpp"
#include "panna/prefixmap.hpp"

namespace panna {
    static std::atomic<size_t> g_collisions( 0 );

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
        Index() {
        }

        template <typename HasherBuilder>
        Index( size_t dimensions, HasherBuilder builder, size_t repetitions ):
            dataset( dimensions ),
            current_query( dimensions ),
            hasher( builder.build( repetitions ) ),
            hashed_points( 0 ) {
            lsh_maps.resize( repetitions );
        }

        template <typename Archive>
        void serialize( Archive& ar ) {
            ar( dataset, current_query, lsh_maps, hasher, hashed_points );
        }

        friend bool operator==( const Index<Dataset, Hasher, Distance>& a,
                                const Index<Dataset, Hasher, Distance>& b ) {
            return a.dataset == b.dataset && a.current_query == b.current_query &&
                   a.lsh_maps == b.lsh_maps && a.hasher == b.hasher &&
                   a.hashed_points == b.hashed_points;
        }

        void save_to( std::string path ) const {
            if ( std::filesystem::exists( path ) ) {
                throw std::invalid_argument( "path already exists" );
            }

            std::ofstream os( path, std::ios::binary );
            cereal::BinaryOutputArchive ar( os );
            ar( *this );
        }

        static Index<Dataset, Hasher, Distance> load_from( std::string path ) {
            std::ifstream is( path, std::ios::binary );
            cereal::BinaryInputArchive ar( is );

            Index<Dataset, Hasher, Distance> index;
            ar( index );
            return index;
        }

        template <typename HasherBuilder, typename InputPoint>
        static Index<Dataset, Hasher, Distance> build_or_load_from( size_t dimensions,
                                                                    HasherBuilder builder,
                                                                    size_t repetitions,
                                                                    std::vector<InputPoint>& points,
                                                                    std::string path ) {
            if ( std::filesystem::exists( path ) ) {
                std::cerr << "loading from file" << std::endl;
                return load_from( path );
            } else {
                Index<Dataset, Hasher, Distance> index( dimensions, builder, repetitions );
                for ( auto p : points ) {
                    index.insert( p.begin(), p.end() );
                }
                index.rebuild();
                return index;
            }
        }

        template <typename Iter>
        void insert( Iter begin, Iter end ) {
            dataset.push_back( begin, end );
        }

        void rebuild() {
            std::vector<THashValue> hashes;

#pragma omp parallel for private( hashes )
            for ( size_t i = hashed_points; i < dataset.size(); i++ ) {
                auto tid = omp_get_thread_num();
                // auto & hashes = tl_hash_values[tid];
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

        template <typename Iter>
        void search_brute_force( Iter begin, Iter end,
                                 size_t k,
                                 std::vector<std::pair<float, uint32_t>>& output ) {
            current_query.clear();
            current_query.push_back( begin, end );

            std::priority_queue<std::pair<float, uint32_t>> top;

            PointHandle q = current_query[0];
            dbg(q);

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
        }

        // TODO: collect statistics of the execution, including the average distance of the
        // collisions
        template <typename Iter>
        void search( Iter begin,
                     Iter end,
                     size_t k,
                     float delta,
                     std::vector<std::pair<float, uint32_t>>& output ) {
            size_t collisions = 0;
            // Setup
            output.clear();
            current_query.clear();
            current_query.push_back( begin, end );
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
                            stop = true;
                            break;
                        }
                    }
                }
            }
            g_collisions += collisions;
        }
    };
} // namespace panna

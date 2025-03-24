#pragma once

#include <queue>

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
            for ( size_t i = hashed_points; i < dataset.size(); i++ ) {
                auto tid = omp_get_thread_num();
                hasher.hash( dataset[i], hashes );
                for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                    lsh_maps[rep].insert( tid, i, hashes[rep] );
                }
            }

#pragma omp parallel for
            for ( size_t rep = 0; rep < lsh_maps.size(); rep++ ) {
                lsh_maps.rebuild();
            }

            hashed_points = dataset.size();
        }

        template <typename InputPoint>
        void search_brute_force( InputPoint& query,
                                 size_t k,
                                 std::vector<std::pair<float, size_t>>& output ) {
            current_query.clear();
            current_query.push_back( query );

            std::priority_queue<std::pair<float, size_t>> top;

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
        }
    };
} // namespace panna

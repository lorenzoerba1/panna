#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <omp.h>
#include <optional>
#include <unordered_map>
#include <vector>

#include "panna/data.hpp"
#include "panna/dsu.hpp"
#include "panna/logging.hpp"

namespace panna::baselines {

    template <typename Dataset, typename Hasher, typename Distance>
    class NearNbr {
        using PointHandle = typename Dataset::PointHandle;
        using THashValue = typename Hasher::Value;

        size_t dimensions;
        size_t repetitions;
        // A reference to the actual data points
        Dataset& dataset;

        // The buckets, with a map for each repetition
        // OPTIMIZE: switch to unordered_map
        std::vector<std::map<THashValue, std::vector<uint32_t>>> buckets;

        // How to build hash functions
        typename Hasher::Builder builder;

        // How to hash the points. Initialized upon the first call to "rebuild"
        std::optional<Hasher> hasher;

    public:
        NearNbr( Dataset& dataset, typename Hasher::Builder builder, size_t repetitions ):
            dimensions( dataset.get_dimensions() ),
            repetitions( repetitions ),
            dataset( dataset ),
            buckets( repetitions ) {
            if ( !hasher.has_value() ) {
                builder.fit( dataset );
                hasher = builder.build( repetitions );
            }

            std::vector<THashValue> hashes;

            for ( size_t i = 0; i < dataset.size(); i++ ) {
                hasher->hash( dataset[i], hashes );
                for ( size_t rep = 0; rep < buckets.size(); rep++ ) {
                    buckets[rep][hashes[rep]].push_back( i );
                }
            }
        }

        std::optional<std::pair<size_t, float>> near_neighbor( typename Dataset::PointHandle query,
                                                               float range,
                                                               std::vector<bool>& ignored ) {
            size_t collisions = 0;
            const size_t collisions_limit = 3 * repetitions;

            std::vector<THashValue> hashes;
            hasher->hash( query, hashes );

            for ( size_t rep = 0; rep < buckets.size(); rep++ ) {
                if ( collisions >= collisions_limit ) {
                    return std::nullopt;
                }

                for ( auto id : buckets[rep][hashes[rep]] ) {
                    if ( !ignored[id] ) {
                        float d = Distance::compute( query, dataset[id] );
                        if ( d <= range ) {
                            return std::optional( std::make_pair( id, d ) );
                        }
                    }
                    collisions++;
                }
            }

            return std::nullopt;
        }
    }; // namespace panna::baselines

    template <typename Dataset, typename HasherBuilder, typename Distance>
    static std::vector<std::tuple<float, uint32_t, uint32_t>> approximate_cc(
        Dataset& points, float range, float failure_probability, HasherBuilder builder ) {

        typename HasherBuilder::Output dummy = builder.build( 0 );
        float p1 = dummy.collision_probability( range );
        size_t repetitions = std::ceil( std::log( points.size() / failure_probability ) /
                                        std::pow( p1, dummy.get_concatenations() ) );

        LOG_INFO( "repetitions", repetitions, "p1", p1 );

        NearNbr<Dataset, typename HasherBuilder::Output, Distance> nearnbr(
            points, builder, repetitions );
        LOG_INFO( "msg", "repetitions instantiated" );

        size_t remaining_points = points.size();
        std::vector<bool> removed( points.size() );
        std::vector<std::tuple<float, uint32_t, uint32_t>> edges;

        while ( remaining_points > 0 ) {
            size_t p = 0;
            for ( ; p < points.size(); p++ ) {
                if ( !removed[p] ) {
                    break;
                }
            }
            removed[p] = true;
            remaining_points--;
            std::vector<size_t> S;
            S.push_back( p );

            while ( !S.empty() ) {
                size_t q = S.back();
                S.pop_back();
                while ( true ) {
                    std::optional<std::pair<uint32_t, float>> pprime_pair =
                        nearnbr.near_neighbor( points[q], range, removed );
                    if ( !pprime_pair ) {
                        break;
                    } else {
                        float pprime_dist = pprime_pair->second;
                        uint32_t pprime = pprime_pair->first;
                        edges.emplace_back( pprime_dist, q, pprime );
                        removed[pprime] = true;
                        remaining_points--;
                        S.push_back( pprime );
                    }
                }
            }
        }

        return edges;
    }

    template <typename Dataset, typename HasherBuilder, typename Distance>
    static std::pair<float, std::vector<std::tuple<float, uint32_t, uint32_t>>>
    emst_theory_of_computing( Dataset& points,
                              float gamma,
                              float failure_probability,
                              HasherBuilder builder ) {

        size_t n = points.size();
        builder.fit( points );

        DSU dsu( n );
        std::vector<std::tuple<float, uint32_t, uint32_t>> tree;
        float r = approximate_diameter<Distance>( points ) / 2.0;
        size_t M = std::ceil( std::log( n / gamma ) / std::log( 1 + gamma ) );
        LOG_INFO( "r", r, "M", M, "n", n );

        for ( size_t i = 0; i <= M; i++ ) {
            if ( tree.size() == n - 1 ) {
                break;
            }
            float ri = r / std::pow( 1 + gamma, M - i );
            auto eprime = approximate_cc<Dataset, HasherBuilder, Distance>(
                points, ri, failure_probability, builder );
            LOG_INFO( "i", i, "discovered-edges", eprime.size(), "ri", ri );
            for ( auto e : eprime ) {
                if ( !dsu.is_connected( std::get<1>( e ), std::get<2>( e ) ) ) {
                    dsu.union_sets( std::get<1>( e ), std::get<2>( e ) );
                    tree.push_back( e );
                }
            }
        }

        expect( tree.size() == n - 1 );
        std::sort( tree.begin(), tree.end() );
        float cost = 0.0;
        for ( auto& e : tree ) {
            cost += std::get<0>( e );
        }
        return std::make_pair( cost, tree );
    }
} // namespace panna::baselines

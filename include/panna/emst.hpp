#pragma once
#include <algorithm>
#include <iostream>
#include <random>
#include <unistd.h>
#include <vector>

#include "panna/dsu.hpp"
#include "panna/logging.hpp"
#include "panna/trieindex.hpp"
using EdgeTuple = std::tuple<float, std::pair<uint32_t, uint32_t>>;

namespace panna {

    // @brief Computes the binomial coefficient,
    //        adapted from
    //        https://stackoverflow.com/questions/55421835/c-binomial-coefficient-is-too-slow
    // @param n number of elements
    // @param k number of elements to choose
    // @return the binomial coefficient C(n, k)
    static double binomial_coefficient( const double n, const double k ) {
        double aSolutions = 0, oldSolutions = n - k + 1;

        for ( double i = 1; i < k; ++i ) {
            aSolutions = oldSolutions * ( n - k + 1 + i ) / ( i + 1 );
            oldSolutions = aSolutions;
        }
        return aSolutions;
    }

    template <typename Dataset, typename Hasher, typename Distance>
    class EMST {
        // Object variables
        uint32_t dimensionality;
        size_t MAX_REPETITIONS;
        uint32_t MAX_HASHBITS;
        Index<Dataset, Hasher, Distance> table;
        std::vector<std::vector<float>> data{};
        uint32_t num_data{ 0 };
        double delta{ 0.01 };
        const float epsilon{ 0.2 };
        DSU dsu_true;
        DSU filter;
        // Sets for the confimed and the unconfirmed edges
        std::vector<EdgeTuple> top;
        std::vector<std::vector<EdgeTuple>> local_confirmed;
        std::vector<std::vector<EdgeTuple>> local_edges;
        float max_weight;
        size_t distances_computed = 0;

    public:
        /**
         * @brief Class to construct an approximate Euclidean Mininmum Spanning Tree from data
         * points
         *
         * @param dimensions Dimension of the hash index
         * @param repetitions Number of repetitions for the LSH index
         * @param builder Builder for the hash function
         * @param data_in Input data points
         * @param data_dimensionality Dimensionality of the input data
         * @param delta Probability of failure parameter (default: 0.01)
         * @param epsilon Approximation factor parameter (default: 0.2)
         *
         * @details This constructor initializes an EMST object by:
         * 1. Set up the LSH index table with the distance metric
         * 2. Insert all input vectors into the index
         * 3. Rebuilding the index structure
         * 4. Construct a Union Find data structure
         * The constructor takes ownership of the input data through a move operation.
         */
        EMST( const size_t dimensions,
              const size_t repetitions,
              const typename Hasher::Builder builder,
              std::vector<std::vector<float>>& data_in,
              const double delta_in = 0.01,
              const float epsilon = 0.2 ):
            dimensionality( dimensions ),
            table( Index<Dataset, Hasher, Distance>( dimensions, builder, repetitions ) ),
            data( data_in ),
            num_data( ( data ).size() ),
            epsilon( epsilon ),
            dsu_true( DSU( num_data ) ),
            filter( DSU( num_data ) ) {
            // Insert the data
            for ( auto& point : data_in ) {
                table.insert( point.begin(), point.end() );
            }
            table.rebuild();

            // Get info on the index
            MAX_HASHBITS = table.num_concatenations();
            MAX_REPETITIONS = table.num_repetitions();
            delta = delta_in;// / num_data;
            max_weight = std::numeric_limits<float>::infinity();

            local_edges.resize( MAX_REPETITIONS );
            local_confirmed.resize( MAX_REPETITIONS );
            LOG_INFO("msg", "Index constructed",
                     "L", MAX_REPETITIONS,
                     "K", MAX_HASHBITS,
                     "num_data", num_data,
                     "delta", delta);
        };

        /// @brief Destructor
        ~EMST() = default;

        /// @brief Computes the exact MST with Kruskal's algorithm in a naive way
        /// @return weight of the exact MST
        float exact_tree() {
            // Clear from any previous runs
            clear();
            // Compute all the distances
            //  We can pre-allocate all the memory, and avoid the critical region
            std::vector<EdgeTuple> all_edges( binomial_coefficient( num_data, 2 ) );
#pragma omp parallel for collapse (2)
            for ( size_t i = 0; i < num_data; i++ ) {
                for ( size_t j = i + 1; j < num_data; j++ ) {
                    float dist = table.get_distance( i, j );
                    all_edges[ i * (num_data -1) - ( i * ( i + 1 ) / 2 ) + j - 1 ] =
                        std::make_tuple(  dist, std::make_pair( i, j ) );
                }
            }
            // Sort the edges
            std::sort( all_edges.begin(), all_edges.end() );
            // Create the DSU
            DSU dsu( num_data );
            float tree_weight = 0;
            std::cout << "Creating the MST" << std::endl;
            std::vector<EdgeTuple> tree;
            kruskal( dsu, all_edges, tree );
            size_t position_last_edge = static_cast<size_t>(std::find( all_edges.begin(), all_edges.end(), tree.back() ) - all_edges.begin());
            LOG_INFO("msg", "MST created",
                      "heaviest_edge", std::get<0>( tree.back() ),
                      "Weight_edge_2n", std::get<0>( all_edges[std::min(2 * (position_last_edge), (all_edges.size() - 1))] )
            );
            for ( const auto& edge : tree ) {
                tree_weight += std::get<0>( edge );
            }
            return tree_weight;
        }

        /// @brief Find the Minimum Spanning Tree using only confirmed edges
        float find_tree() {
            clear();
            float tree_weight = 0;
            std::vector<std::pair<unsigned int, unsigned int>> tree;

            bool found = false;
            std::vector<EdgeTuple> edges;
            for ( size_t i_rev = 0; i_rev <= MAX_HASHBITS; i_rev++ ) {
                size_t i = MAX_HASHBITS - i_rev;
                if ( found ) {
                    break;
                }
                size_t completed_repetitions = 0;

                #pragma omp parallel for schedule(static, 1)
                for ( size_t j = 0; j < MAX_REPETITIONS; j++ ) {
                    if ( found ) {
                        continue;
                    }
                    DSU local_dsu( num_data );
                    std::vector<EdgeTuple> local_top, local_Tu;
                    enumerate_edges( i, j, local_Tu );
                    
                    std::sort(local_Tu.begin(), local_Tu.end());
                    kruskal( local_dsu, local_Tu, local_top );

                    local_confirmed[j].insert( local_confirmed[j].end(),
                                               std::make_move_iterator( local_top.begin() ),
                                               std::make_move_iterator( local_top.end() ) );

                    if ( j%50 == 0 )
                     {
                    #pragma omp critical
                    {
                        completed_repetitions+= 50;
                        edges.insert( edges.end(),
                                      std::make_move_iterator( top.begin() ),
                                      std::make_move_iterator( top.end() ) );

                        for ( size_t local_index = 0; local_index < MAX_REPETITIONS; local_index++ ) {
                            auto& local = local_confirmed[local_index];
                            edges.insert( edges.end(),
                                std::make_move_iterator(local.begin()),
                                std::make_move_iterator(local.end()));
                            local.clear();
                        }

                        top.clear();

                        if ( edges.size() >= num_data - 1 ) {
                            dsu_true = DSU( num_data );
                            std::sort( edges.begin(), edges.end() );
                            kruskal( dsu_true, edges, top );

                            LOG_INFO( "prefix", i, "repetition", completed_repetitions, "tree_size", top.size() );
                            if ( top.size() == num_data - 1 ) {
                                float new_tree_weight = 0;
                                max_weight = std::pow( std::get<float>( top.back() ), 2 );
                                for ( const auto& edge : top ) {
                                    new_tree_weight += std::get<0>( edge );
                                }
                                tree_weight = new_tree_weight;
                                // Fill the DSU filter with just the confirmed edgesù
                                auto partition_point = std::partition_point(
                                    top.begin(), top.end(), [&]( const auto& e ) {
                                        return table.fail_probability(
                                                   std::get<float>( e ), i, j ) < delta;
                                    } );
                                filter = DSU( num_data );
                                for ( auto it = top.begin(); it != top.end(); ++it ) {
                                    filter.union_sets( std::get<1>( *it ).first, std::get<1>( *it ).second );
                                }

                                float fp = failure_probability( i, completed_repetitions );
                                LOG_INFO( "prefix",
                                          i,
                                          "repetition",
                                          completed_repetitions,
                                          "tree_weight",
                                          tree_weight,
                                          "failure_probability",
                                          fp,
                                          "max_edge_weight",
                                          std::get<float>( top.back() ),
                                          "mean_edge_weight",
                                          tree_weight / ( num_data - 1 ) );
                                if ( fp < delta ) {
                                    found = true;
                                    // Fill the tree
                                    for ( const auto& edge : top ) {
                                        tree.push_back( std::get<1>( edge ) );
                                    }
                                }
                            }
                            // Lose the unused edges, MST is composable wrt to edge partitioning
                            edges.clear();
                        }
                    }
                }
            }
                LOG_INFO( "msg", "finished prefix", "prefix", i );
            }
            // This is just a sanity check to see if dsu works as intended
            is_connected( tree );
            LOG_INFO("msg", "EMST finished",
                     "distances_computed", distances_computed);
            return tree_weight;
        }

        /// @brief Find the ɛ-EMST using both confirmed and unconfirmed edges
        float find_epsilon_tree() {
            clear();
            // dirty_start(local_confirmed[0]);
            float tree_weight = 0;
            float old_weight = std::numeric_limits<float>::infinity();
            std::vector<std::pair<unsigned int, unsigned int>> tree;

            bool found = false;
            std::vector<EdgeTuple> edges;
            for ( size_t i_rev = 0; i_rev <= MAX_HASHBITS; i_rev++ ) {
                size_t i = MAX_HASHBITS - i_rev;
                size_t completed_repetitions = 0;
                if ( found )
                    break;
#pragma omp parallel for schedule(static, 1)
                for ( size_t j = 0; j < MAX_REPETITIONS; j++ ) {
                    if ( found )
                        continue;
                    DSU local_dsu( num_data );
                    std::vector<EdgeTuple> local_top, local_Tu;
                    enumerate_edges( i, j, local_Tu );

                    kruskal( local_dsu, local_Tu, local_top );

                    local_confirmed[j].insert( local_confirmed[j].end(),
                                               std::make_move_iterator( local_top.begin() ),
                                               std::make_move_iterator( local_top.end() ) );

                    // Every x iterations we have a batch, construct the MST from these edges
                    if ( j%50 == 0 )
                     {
#pragma omp critical
                        {
                            edges.insert( edges.end(),
                                          std::make_move_iterator( top.begin() ),
                                          std::make_move_iterator( top.end() ) );
                            top.clear();
                            dsu_true = DSU( num_data );
                            for ( size_t local_index = 0; local_index < j + 1; local_index++ ) {
                                auto& local = local_confirmed[local_index];
                                edges.insert( edges.end(), 
                                    std::make_move_iterator(local.begin()),
                                    std::make_move_iterator(local.end()) );
                                local.clear();
                            }
                            std::sort( edges.begin(), edges.end() );
                            // edges.erase( std::unique( edges.begin(), edges.end() ), edges.end()
                            // );
                            if ( edges.size() > num_data - 1 ) {
                               kruskal( dsu_true, edges, top );
                                LOG_INFO("prefix", i,
                                         "repetition", j,
                                         "tree_size", top.size());
                                if ( top.size() == num_data - 1 ) {
                                    float new_tree_weight = 0;
                                    max_weight = std::pow( std::get<float>( top.back() ), 2 );
                                    for ( const auto& edge : top ) {
                                        new_tree_weight += std::get<0>( edge );
                                    }

                                    // Find the edges that satisfy the failure probability
                                    float delta_local = delta;
                                    // auto partition_point = std::find_if(
                                    //     top.begin(), top.end(), [&]( const auto& e ) {
                                    //         delta_local -= table.fail_probability(
                                    //                    std::get<float>( e ), i, j );
                                    //         return delta_local <= 0.0f;
                                    //     } );
                                    auto partition_point = std::partition_point(
                                        top.begin(), top.end(), [&]( const auto& e ) {
                                            return table.fail_probability(
                                                       std::get<float>( e ), i, j ) < delta;
                                        } );

                                    // Fill the DSU filter with just the confirmed edges
                                    filter = DSU( num_data );
                                    for ( auto it = top.begin(); it != partition_point; ++it ) {
                                        filter.union_sets( std::get<1>( *it ).first, std::get<1>( *it ).second );
                                    }
                                    // Find the bound on the weight of the tree
                                    tree_weight = new_tree_weight;
                                    float bound_weight =
                                        ( 1 + epsilon ) *
                                            std::accumulate( top.begin(),
                                                             partition_point,
                                                             0.0f,
                                                             []( float acc, const EdgeTuple& e ) {
                                                                 return acc + std::get<float>( e );
                                                             } );
                                    if ( partition_point != top.begin() ) {
                                        bound_weight += ( 1 + epsilon ) * ( std::get<float>( *( partition_point - 1 ) ) *
                                                          std::distance( partition_point, top.end() ) );
                                    }
                                    LOG_INFO("prefix", i,
                                             "repetition", completed_repetitions,
                                             "tree_weight", tree_weight,
                                             "bound_weight", bound_weight,
                                             "max_edge_weight", std::get<float>(top.back()),
                                             "mean_edge_weight", tree_weight / (num_data - 1)
                                    );
                                    if ( tree_weight <= bound_weight ) {
                                        found = true;
                                        // Fill the tree
                                        for ( const auto& edge : top ) {
                                            tree.push_back( std::get<1>( edge ) );
                                        }
                                    }
                                    // If the weight hasn't changed epsilon*old_weight
                                    // else if ( i <= 5 && std::abs( old_weight - tree_weight ) / old_weight < epsilon ) {
                                    //     LOG_INFO("msg", "Tree weight converged",
                                    //              "old_weight", old_weight,
                                    //              "tree_weight", tree_weight,
                                    //              "relative_change", std::abs( old_weight - tree_weight ) / old_weight);
                                    //     found = true;
                                    //     // Fill the tree
                                    //     for ( const auto& edge : top ) {
                                    //         tree.push_back( std::get<1>( edge ) );
                                    //     }
                                    // }
                                    old_weight = tree_weight;
                                }
                                // Lose the unused edges, MST is composable wrt to edge partitioning
                                edges.clear();
                            }
                            completed_repetitions+= 50;
                        }
                    }
                }
                LOG_INFO("msg", "finished prefix", "prefix", i);}
            is_connected( tree );
            LOG_INFO("msg", "EMST finished",
                     "distances_computed", distances_computed);
            return tree_weight;
        }

        std::pair<
        std::vector<std::tuple<float, float, float>>,
        std::vector< std::vector< std::pair<float, unsigned int>>> > find_tree_dbscan( size_t k=5 ) {
            clear();
            // dirty_start(local_confirmed[0]);
            float tree_weight = 0, old_weight = std::numeric_limits<float>::infinity();
            std::vector<std::tuple<float, float, float>> tree;
            std::vector< std::vector< std::pair<float, unsigned int>>> neighbors (num_data, std::vector< std::pair<float, unsigned int> >() );
            std::vector<std::vector< std::vector< std::pair<float, unsigned int>>>> local_neighbors_list( 16, std::vector< std::vector< std::pair<float, unsigned int> > >(num_data, std::vector< std::pair<float, unsigned int> >(1, std::make_pair( 0.0f, -1) ) ) );

            bool found = false;
            std::vector<EdgeTuple> edges;
            // Find also the top-k nearest neighbors for each node
            for ( size_t i_rev = 0; i_rev <= MAX_HASHBITS; i_rev++ ) {
                size_t i = MAX_HASHBITS - i_rev;
                size_t completed_repetitions = 0;
                if ( found )
                    break;
#pragma omp parallel for schedule(static, 1)
                for ( size_t j = 0; j < MAX_REPETITIONS; j++ ) {
                    if ( found )
                        continue;
                    DSU local_dsu( num_data );
                    auto& local_neighbors = local_neighbors_list[ omp_get_thread_num() ];
                    std::vector<EdgeTuple> local_top, local_Tu;
                    enumerate_edges( i, j, local_Tu );

                    // Fill the neighbors
                    for ( const auto& edge : local_Tu ) {
                        local_neighbors[ std::get<1>( edge ).first ].emplace_back( std::get<0>( edge ), std::get<1>( edge ).second );
                        local_neighbors[ std::get<1>( edge ).second ].emplace_back( std::get<0>( edge ), std::get<1>( edge ).first );
                    }
                    for ( size_t index = 0; index < local_neighbors.size(); index++ ) {
                        // Use partial sort to keep just the best k
                        if ( local_neighbors[index].size() > k) {
                            std::partial_sort( local_neighbors[index].begin(),
                                                  local_neighbors[index].begin() + k,
                                                    local_neighbors[index].end() );
                            local_neighbors[index].resize( k );
                            continue;
                        }

                        std::sort( local_neighbors[index].begin(), local_neighbors[index].end() );
                    }
                    for ( auto& edge : local_Tu ) {
                        // Use the reachability
                        // Check if we have the elements in local neighbors before accessing the back
                        if ( !local_neighbors[ std::get<1>( edge ).first ].empty() && !local_neighbors[ std::get<1>( edge ).second ].empty() ) {
                            std::get<float>( edge ) = std::max( 
                                std::get<float>( edge ),
                                std::max( local_neighbors[ std::get<1>( edge ).first ].back().first,
                                           local_neighbors[ std::get<1>( edge ).second ].back().first ) );
                        }
                        if ( local_top.size() == num_data - 1 ) {
                            break;
                        }
                        add_edge( edge, local_dsu, local_top );
                    }
                    local_confirmed[j].insert( local_confirmed[j].end(),
                                               std::make_move_iterator( local_top.begin() ),
                                               std::make_move_iterator( local_top.end() ) );

                    // Every x iterations we have a batch, construct the MST from these edges
                    if ( j%50 == 0 )
                     {
#pragma omp critical
                        {   
                            // Merge the local neighbors
                            for ( size_t thread_index = 0; thread_index < local_neighbors_list.size();
                                    thread_index++ ) {
                                    auto& thread_neighbors = local_neighbors_list[ thread_index ];
                                    for ( size_t index = 0; index < thread_neighbors.size(); index++ ) {
                                        auto& local = thread_neighbors[index];
                                        neighbors[index].insert( neighbors[index].end(),
                                                                std::make_move_iterator( local.begin() ),
                                                                std::make_move_iterator( local.end() ) );
                                        local.clear();
                                    }
                                }
                            for ( size_t index = 0; index < neighbors.size(); index++ ) {
                                if ( neighbors[index].size() > k) {
                                    // Use partial sort to keep just the best k
                                    std::partial_sort( neighbors[index].begin(),
                                                       neighbors[index].begin() + k,
                                                       neighbors[index].end() );
                                    neighbors[index].resize( k );
                                    continue;
                                }

                                std::sort( neighbors[index].begin(), neighbors[index].end() );
                            }

                            // Merge the local tops
                            edges.insert( edges.end(),
                                          std::make_move_iterator( top.begin() ),
                                          std::make_move_iterator( top.end() ) );
                            top.clear();
                            dsu_true = DSU( num_data );
                            for ( size_t local_index = 0; local_index < j + 1; local_index++ ) {
                                auto& local = local_confirmed[local_index];
                                edges.insert( edges.end(),
                                                std::make_move_iterator(local.begin()),
                                                std::make_move_iterator(local.end()) );
                                local.clear();
                            }
                            // Change the edge weights based on their reachability
                            for (auto& edge: edges){
                                std::get<float>( edge ) = std::max( 
                                    std::get<float>( edge ),
                                    std::max( neighbors[ std::get<1>( edge ).first ].back().first,
                                              neighbors[ std::get<1>( edge ).second ].back().first ) );
                            }
                            std::sort( edges.begin(), edges.end() );

                            if ( edges.size() > num_data - 1 ) {
                                for ( const auto& edge : edges ) {
                                    add_edge( edge, dsu_true, top );
                                    if ( top.size() == num_data - 1 ) {
                                        break;
                                    }
                                }

                                if ( top.size() == num_data - 1 ) {
                                    float new_tree_weight = 0;
                                    max_weight = std::get<float>( top.back() );
                                    for ( const auto& edge : top ) {
                                        new_tree_weight += std::get<0>( edge );
                                    }
                                    // Find the edges that satisfy the failure probability
                                    float delta_local = delta;
                                    auto partition_point = std::find_if(
                                        top.begin(), top.end(), [&]( const auto& e ) {
                                            delta_local -= table.fail_probability(
                                                       std::get<float>( e ), i, j );
                                            return delta_local <= 0.0f;
                                        } );
                                        
                                    tree_weight = new_tree_weight;
                                    float bound_weight =
                                        ( 1 + epsilon ) *
                                            std::accumulate( top.begin(),
                                                             partition_point,
                                                             0.0f,
                                                             []( float acc, const EdgeTuple& e ) {
                                                                 return acc + std::get<float>( e );
                                                             } );
                                    if ( partition_point != top.begin() ){
                                        bound_weight += ( std::get<float>( *( partition_point - 1 ) ) *
                                                          std::distance( partition_point, top.end() ) );
                                    }
                                    // Fill the pruning DSU
                                    filter = DSU( num_data );
                                    for ( auto it = top.begin(); it != partition_point; ++it ) {
                                        filter.union_sets( std::get<1>( *it ).first, std::get<1>( *it ).second );
                                    }
                                    LOG_INFO("prefix", i,
                                             "repetition", j,
                                             "tree_weight", tree_weight,
                                             "bound_weight", bound_weight,
                                             "max_edge_weight", std::get<float>(top.back()),
                                             "mean_edge_weight", tree_weight / (num_data - 1));
                                    if ( tree_weight <= bound_weight ) {
                                        tree.clear();
                                        found = true;
                                        // Fill the tree
                                        for ( const auto& edge : top ) {
                                            tree.emplace_back( static_cast<float>( std::get<1>( edge ).first ),
                                                               static_cast<float>( std::get<1>( edge ).second ),
                                                                std::get<0>( edge ) ); 
                                        }
                                    }
                                    // If the weight hasn't changed epsilon*old_weight
                                    else if ( i <= 5 && std::abs( old_weight - tree_weight ) / old_weight < epsilon ) {
                                        tree.clear();
                                        LOG_INFO("msg", "Tree weight converged",
                                                 "old_weight", old_weight,
                                                 "tree_weight", tree_weight,
                                                 "relative_change", std::abs( old_weight - tree_weight ) / old_weight);
                                        found = true;
                                        // Fill the tree
                                        for ( const auto& edge : top ) {
                                            tree.emplace_back( static_cast<float>( std::get<1>( edge ).first ),
                                                               static_cast<float>( std::get<1>( edge ).second ),
                                                                std::get<0>( edge ) );
                                        }
                                    }
                                    old_weight = tree_weight;
                                }
                                // Lose the unused edges, MST is composable wrt to edge partitioning
                                edges.clear();
                            }
                        }
                        completed_repetitions+= 50;
                    }
                }
                // Clear the local neighbors
                for ( auto& local : local_neighbors_list ) {
                    for ( auto& vec : local ) {
                        vec.clear();
                    }
                }
    
            }
            
            return { tree, neighbors };
        }


        float mean_weight() {
            size_t edges_to_pick = std::min<size_t>( binomial_coefficient( num_data, 2 ), 10000 );
            // Pick edges_to_pick random couples of nodes and compute the mean weight
            float mean = 0;
            for ( size_t i = 0; i < edges_to_pick; i++ ) {
                // Get a random edge
                std::random_device rd;
                std::mt19937 gen( rd() );
                std::uniform_int_distribution<size_t> dist_1( 0, num_data - 1 );
                size_t random_index_1 = dist_1( gen );
                size_t random_index_2 = dist_1( gen );
                // Get the distance of the edge
                float dist = table.get_distance( random_index_1, random_index_2 );
                // Add it to the mean
                mean += dist;
            }
            mean /= edges_to_pick;
            return mean;
        }

        //*** Private methods */
    private:
        /// @brief Obtain the couples of nodes that share the same prefix from the hash table
        /// @param i Current concatenation in the hash index
        /// @param j current repetition in the hash index
        /// @param Tu_local vector that stores the edges
        void enumerate_edges( size_t i, size_t j, std::vector<EdgeTuple>& Tu_local ) {
            // Discover edges that share the same prefix at iteration i, j
            std::vector<EdgeTuple> couples;
            size_t computed = table.search_pairs_filter( j, i, couples, max_weight, filter );
            Tu_local.insert( Tu_local.end(), 
                std::make_move_iterator(couples.begin()),
                std::make_move_iterator(couples.end()) );
// #pragma omp atomic
//             distances_computed += computed;
            return;
        };

        /// @brief Checks wheter a tree is connected
        /// @param tree the tree that we want to check
        /// @return true if all edge are connected, false otherwise.
        bool is_connected( std::vector<std::pair<unsigned int, unsigned int>>& tree ) {
            // Check if the tree is connected
            std::vector<std::pair<unsigned int, unsigned int>> edges = tree;
            std::vector<bool> visited( num_data, false );
            std::vector<std::vector<unsigned int>> adj_list( num_data );
            for ( const auto& edge : edges ) {
                adj_list[edge.first].push_back( edge.second );
                adj_list[edge.second].push_back( edge.first );
            }
            std::vector<unsigned int> stack;
            stack.push_back( 0 );
            visited[0] = true;
            while ( !stack.empty() ) {
                unsigned int node = stack.back();
                stack.pop_back();
                for ( const auto& neighbor : adj_list[node] ) {
                    if ( !visited[neighbor] ) {
                        visited[neighbor] = true;
                        stack.push_back( neighbor );
                    }
                }
            }
            // for (const auto& edge : tree) {
            //     std::cout << edge.first << " " << edge.second << std::endl;
            // }

            if ( !std::accumulate(
                     visited.begin(), visited.end(), true, std::logical_and<bool>() ) ) {
                LOG_INFO("msg", "Not connected");
                return false;
            }
            LOG_INFO("msg", "Connected");
            return true;
        };

        /// @brief Add the edge to the tree if it does not create a cycle using the DSU data
        /// structure
        /// @param new_edge_input the edge that we have to add
        /// @param dsu the data structure that keeps track of the connected components
        /// @param edge_list the current edges in the tree
        /// @return true if an edge has been added to the edge_list and the DSU data structure,
        /// false otherwise
        inline bool add_edge( const EdgeTuple& new_edge_input, DSU& dsu, std::vector<EdgeTuple>& edge_list ) {
            // Extract new edge and its weight.
            std::pair<uint32_t, uint32_t> new_edge = std::get<1>( new_edge_input );

            // Try to add new edge normally.
            if ( dsu.union_sets( new_edge.first, new_edge.second ) ) {
                edge_list.push_back( new_edge_input );
                return true;
            }
            return false;
        }

        /// @brief Run Kruskal's algorithm to find the minimum spanning tree
        /// @param dsu the data structure that keeps track of the connected components
        /// @param edge_list the current edges in the tree
        /// @param output the output vector that will contain the edges in the minimum spanning tree
        inline void kruskal( DSU& dsu, std::vector<EdgeTuple>& edge_list, std::vector<EdgeTuple>& output ) {
            for ( const auto& edge : edge_list ) {
                if ( output.size() == num_data - 1 ) {
                    break;
                }
                add_edge( edge, dsu, output );
            }
        }

        /// @brief Generate a random spanning tree to have an initial solution
        void dirty_start( std::vector<EdgeTuple>& clean ) {
            std::vector<unsigned int> vertices( num_data );
            std::iota( vertices.begin(), vertices.end(), 0 );
            std::shuffle(
                vertices.begin(), vertices.end(), std::mt19937{ std::random_device{}() } );
            for ( size_t i = 1; i < vertices.size(); i++ ) {
                clean.emplace_back( table.get_distance( vertices[i - 1], vertices[i] ),
                                    std::make_pair( vertices[i - 1], vertices[i] ) );
            }
        }

        /// @brief Compute an upper bound to the failure probability of the stored spanning tree
        /// @param i current concatenation in the hash index
        /// @param j current repetition in the hash index
        /// @return failure probability of the minimum spanning tree
        float failure_probability( size_t i, size_t j ) {
            float loose_upper_bound =
                ( num_data - 1 ) * table.fail_probability( std::get<0>( top.back() ), i, j );
            float prob = 0.0f;
            for ( auto& edge : top ) {
                prob += table.fail_probability( std::get<float>( edge ), i, j );
            }
            expect( prob <= loose_upper_bound );
            LOG_DEBUG( "msg", "failure-probability",
                      "loose-upper-bound", loose_upper_bound,
                      "union-bound", prob );
            return prob;
        }

        /// @brief Clear the data structures from previous runs
        void clear() {
            top.clear();
            for ( auto& local : local_edges ) {
                local.clear();
            }
            for ( auto& local : local_confirmed ) {
                local.clear();
            }
            local_confirmed.resize( MAX_REPETITIONS );
            local_edges.resize( MAX_REPETITIONS );
            dsu_true = DSU( num_data );
            max_weight = std::numeric_limits<float>::infinity();
            distances_computed = 0;
            filter = DSU( num_data );
        }
    }; // closes class
} // namespace panna

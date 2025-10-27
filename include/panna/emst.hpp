#pragma once
#include <algorithm>
#include <iostream>
#include <random>
#include <unistd.h>
#include <vector>

#include "panna/dsu.hpp"
#include "panna/logging.hpp"
#include "panna/trieindex.hpp"

namespace panna {

    struct StoppingConditionInfo {
        const float total_weight;
        const float confirmed_weight;
        const float heaviest_confirmed_edge;
        const size_t edges_to_confirm;
        const size_t confirmed_edges;
    };

    template <typename Dataset, typename Hasher, typename Distance>
    class EMST {
        // Object variables
        uint32_t dimensionality;
        size_t MAX_REPETITIONS; // TODO: make lowercase, as upper case usually denotes constants
        uint32_t MAX_HASHBITS;
        Index<Dataset, Hasher, Distance> table;
        uint32_t num_data{ 0 };
        double delta{ 0.01 };
        const float epsilon{ 0.2 };
        DSU dsu_true;
        DSU filter;
        // Sets for the confimed and the unconfirmed edges
        std::vector<Edge> top;
        std::vector<std::vector<Edge>> local_confirmed;
        std::vector<std::vector<Edge>> local_edges;
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
            num_data( data_in.size() ),
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
            std::vector<Edge> all_edges( ( num_data - 1 ) * num_data / 2 );
#pragma omp parallel for collapse (2)
            for ( size_t i = 0; i < num_data; i++ ) {
                for ( size_t j = i + 1; j < num_data; j++ ) {
                    float dist = table.get_distance( i, j );
                    all_edges[i * ( num_data - 1 ) - ( i * ( i + 1 ) / 2 ) + j - 1] =
                        Edge{ .weight = dist, .a = (uint32_t)i, .b = (uint32_t)j };
                }
            }
            // Sort the edges
            std::sort( all_edges.begin(), all_edges.end() );
            // Create the DSU
            DSU dsu( num_data );
            float tree_weight = 0;
            std::cout << "Creating the MST" << std::endl;
            std::vector<Edge> tree;
            kruskal( dsu, all_edges, tree );
            size_t position_last_edge = static_cast<size_t>(std::find( all_edges.begin(), all_edges.end(), tree.back() ) - all_edges.begin());
            LOG_INFO("msg", "MST created",
                      "heaviest_edge",  tree.back().weight ,
                      "Weight_edge_2n",  all_edges[std::min(2 * (position_last_edge), (all_edges.size() - 1))].weight 
            );
            for ( const auto& edge : tree ) {
                tree_weight += edge.weight ;
            }
            return tree_weight;
        }

        /// @brief Find the Minimum Spanning Tree using only confirmed edges
        std::pair<float, std::vector<Edge>> find_tree() {
            clear();
            float tree_weight = 0;
            std::vector<Edge> tree;

            bool found = false;
            std::vector<Edge> edges;
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
                    std::vector<Edge> local_top, local_Tu;
                    enumerate_edges( i, j, local_Tu );

                    std::sort( local_Tu.begin(), local_Tu.end() );
                    kruskal( local_dsu, local_Tu, local_top );

                    local_confirmed[j].insert( local_confirmed[j].end(),
                                               std::make_move_iterator( local_top.begin() ),
                                               std::make_move_iterator( local_top.end() ) );

                    // FIXME: make the batch size a parameter
#pragma omp critical
                    {
                        completed_repetitions++;
                        if ( completed_repetitions % 50 == 0 ||
                             completed_repetitions >= MAX_REPETITIONS ) {
                            edges.insert( edges.end(),
                                          std::make_move_iterator( top.begin() ),
                                          std::make_move_iterator( top.end() ) );

                            for ( size_t local_index = 0; local_index < MAX_REPETITIONS;
                                  local_index++ ) {
                                auto& local = local_confirmed[local_index];
                                edges.insert( edges.end(),
                                              std::make_move_iterator( local.begin() ),
                                              std::make_move_iterator( local.end() ) );
                                local.clear();
                            }

                            top.clear();

                            if ( edges.size() >= num_data - 1 ) {
                                dsu_true = DSU( num_data );
                                std::sort( edges.begin(), edges.end() );
                                kruskal( dsu_true, edges, top );
                                // clang-format off
                                LOG_INFO( "prefix", i,
                                          "repetition", completed_repetitions,
                                          "tree_size", top.size() );
                                // clang-format on
                                if ( top.size() == num_data - 1 ) {
                                    StoppingConditionInfo stop =
                                        stopping_condition( i, completed_repetitions );
                                    float weight_lower_bound =
                                        stop.confirmed_weight +
                                        stop.edges_to_confirm * stop.heaviest_confirmed_edge;
                                    bool should_stop =
                                        stop.total_weight <= ( 1 + epsilon ) * weight_lower_bound;
                                    // clang-format off
                                    LOG_INFO( "stop.total_weight", stop.total_weight,
                                              "stop.confirmed_weight", stop.confirmed_weight,
                                              "stop.heaviest_confirmed_edge", stop.heaviest_confirmed_edge,
                                              "stop.edges_to_confirm", stop.edges_to_confirm,
                                              "weight_lower_bound", weight_lower_bound,
                                              "should_stop", should_stop );
                                    // clang-format on

                                    // stop if we are done
                                    if ( should_stop ) {
                                        found = true;
                                        tree_weight = stop.total_weight;
                                        // Fill the tree
                                        tree.clear();
                                        for ( const auto& edge : top ) {
                                            tree.push_back(  edge  );
                                        }
                                    }
                                    // Fill the DSU filter with just the confirmed edges
                                    filter = DSU( num_data );
                                    for ( size_t idx = 0; idx < stop.confirmed_edges; idx++ ) {
                                        auto edge = top[idx];
                                        filter.union_sets( edge.a, edge.b );
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
            LOG_INFO( "msg", "EMST finished", "distances_computed", distances_computed );
            return { tree_weight, tree };
        }

        std::vector<std::pair<float, unsigned int>>
        find_tree_hist( std::vector<std::pair<unsigned int, unsigned int>> tree ) {
            clear();
            float tree_weight = 0;
            std::vector<std::pair<float, unsigned int>> hist;
            hist.push_back( std::make_pair( 0.0f, 0 ) );
            for ( const auto& edge : tree ) {
                float weight = table.get_distance( edge.first, edge.second );
                //top.push_back( std::make_tuple( weight, edge ) );
                hist.push_back( std::make_pair( weight, 0 ) );
            }
            hist.push_back( std::make_pair( std::numeric_limits<float>::max(), 0 ) );

            bool found = false;
            std::vector<Edge> edges;
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
                    std::vector<Edge> local_top, local_Tu;
                    enumerate_edges( i, j, local_Tu );
                    
                    std::sort(local_Tu.begin(), local_Tu.end());

                    local_confirmed[j].insert( local_confirmed[j].end(),
                                               std::make_move_iterator( local_Tu.begin() ),
                                               std::make_move_iterator( local_Tu.end() ) );

                    if ( j%50 == 0 )
                    {
                    #pragma omp critical
                    {
                        completed_repetitions+= 50;

                        for ( size_t local_index = 0; local_index < MAX_REPETITIONS; local_index++ ) {
                            auto& local = local_confirmed[local_index];

                            // For each edge in local, find where it is placed between the distances of the histogram and increment its count
                            for ( const auto& edge : local ) {
                                float weight =  edge.weight ;
                                for ( size_t h = 0; h < hist.size() - 1; h++ ) {
                                    if ( weight >= hist[h].first && weight < hist[h + 1].first ) {
                                        hist[h].second += 1;
                                        break;
                                    }
                                }

                            }
                            edges.insert( edges.end(),
                                std::make_move_iterator(local.begin()),
                                std::make_move_iterator(local.end()));
                            local.clear();
                        }
                        std::sort( edges.begin(), edges.end() );
                        edges.erase( std::unique( edges.begin(), edges.end() ), edges.end() );
                        top.clear();
                        dsu_true = DSU( num_data );
                        kruskal( dsu_true, edges, top );

                        LOG_INFO( "prefix", i, "repetition", completed_repetitions, "tree_size", top.size() );
                        if ( top.size() == num_data - 1 ) {
                            float new_tree_weight = 0;
                            max_weight = std::pow(  top.back().weight , 2 );
                            for ( const auto& edge : top ) {
                                new_tree_weight += edge.weight ;
                            }
                            tree_weight = new_tree_weight;
                            // Fill the DSU filter with just the confirmed edgesù
                            auto partition_point =
                                std::partition_point( top.begin(), top.end(), [&]( const auto& e ) {
                                    return table.fail_probability( e.weight, i, j ) < delta;
                                } );

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
                                        top.back().weight ,
                                        "mean_edge_weight",
                                        tree_weight / ( num_data - 1 ) );
                            if ( fp < delta ) {
                                found = true;
                            }   
                        }
                        
                    }
                }
            }
                LOG_INFO( "msg", "finished prefix", "prefix", i );
            }
            LOG_INFO("msg", "EMST finished",
                     "distances_computed", distances_computed);
            // Write the histogram to a file
            std::ofstream hist_file("histogram.csv");
            hist_file << "weight,count\n";
            for (const auto& [weight, count] : hist) {
                hist_file << weight << "," << count << "\n";
            }
            return hist;
        }


        // TODO: maybe this should also be factored in the find_tree code?
        std::pair<
        std::vector<Edge>,
        std::vector< std::vector< std::pair<float, unsigned int>>> > find_tree_dbscan( size_t k=5 ) {
            clear();
            // dirty_start(local_confirmed[0]);
            float tree_weight = 0, old_weight = std::numeric_limits<float>::infinity();
            std::vector<Edge> tree;
            std::vector< std::vector< std::pair<float, unsigned int>>> neighbors (num_data, std::vector< std::pair<float, unsigned int> >() );
            std::vector<std::vector< std::vector< std::pair<float, unsigned int>>>> local_neighbors_list( 16, std::vector< std::vector< std::pair<float, unsigned int> > >(num_data, std::vector< std::pair<float, unsigned int> >(1, std::make_pair( 0.0f, -1) ) ) );

            bool found = false;
            std::vector<Edge> edges;
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
                    std::vector<Edge> local_top, local_Tu;
                    enumerate_edges( i, j, local_Tu );

                    // Fill the neighbors
                    for ( const auto& edge : local_Tu ) {
                        local_neighbors[edge.a].emplace_back( edge.weight, edge.b );
                        local_neighbors[edge.b].emplace_back( edge.weight, edge.a );
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
                        if ( !local_neighbors[edge.a].empty() &&
                             !local_neighbors[edge.b].empty() ) {
                           edge.weight = std::max(
                                edge.weight ,
                                std::max(
                                    local_neighbors[ edge.a ].back().first,
                                    local_neighbors[ edge.b ].back().first ) );
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
                                edge.weight  = std::max( 
                                    edge.weight,
                                    std::max( neighbors[  edge .a ].back().first,
                                              neighbors[  edge .b].back().first ) );
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
                                    max_weight =  top.back().weight ;
                                    for ( const auto& edge : top ) {
                                        new_tree_weight +=  edge.weight ;
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
                                                             []( float acc, const Edge& e ) {
                                                                 return acc +  e.weight ;
                                                             } );
                                    if ( partition_point != top.begin() ){
                                        bound_weight += ( ( *( partition_point - 1 ) ).weight *
                                                          std::distance( partition_point, top.end() ) );
                                    }
                                    // Fill the pruning DSU
                                    filter = DSU( num_data );
                                    for ( auto it = top.begin(); it != partition_point; ++it ) {
                                        filter.union_sets( ( *it ).a, ( *it ).b );
                                    }
                                    LOG_INFO("prefix", i,
                                             "repetition", j,
                                             "tree_weight", tree_weight,
                                             "bound_weight", bound_weight,
                                             "max_edge_weight", top.back().weight,
                                             "mean_edge_weight", tree_weight / (num_data - 1));
                                    if ( tree_weight <= bound_weight ) {
                                        tree.clear();
                                        found = true;
                                        // Fill the tree
                                        for ( const auto& edge : top ) {
                                            tree.emplace_back( static_cast<float>( ( edge ).a ),
                                                               static_cast<float>( ( edge ).b ),
                                                                 edge.weight  ); 
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
                                            tree.emplace_back( static_cast<float>(  edge .a ),
                                                               static_cast<float>(  edge .b),
                                                               edge.weight  );
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
            size_t edges_to_pick = std::min<size_t>( ( num_data - 1 ) * num_data / 2, 10000 );
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
        void enumerate_edges( size_t i, size_t j, std::vector<Edge>& Tu_local ) {
            // Discover edges that share the same prefix at iteration i, j
            std::vector<Edge> couples;
            table.search_pairs_filter( j, i, couples, max_weight, filter );
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
        bool is_connected( std::vector<Edge>& tree ) {
            // Check if the tree is connected
            std::vector<Edge> edges = tree;
            std::vector<bool> visited( num_data, false );
            std::vector<std::vector<unsigned int>> adj_list( num_data );
            for ( const auto& edge : edges ) {
                adj_list[edge.a].push_back( edge.b);
                adj_list[edge.b].push_back( edge.a );
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
        inline bool add_edge( const Edge& new_edge, DSU& dsu, std::vector<Edge>& edge_list ) {
            // Try to add new edge normally.
            if ( dsu.union_sets( new_edge.a, new_edge.b ) ) {
                edge_list.push_back( new_edge );
                return true;
            }
            return false;
        }

        /// @brief Run Kruskal's algorithm to find the minimum spanning tree
        /// @param dsu the data structure that keeps track of the connected components
        /// @param edge_list the current edges in the tree
        /// @param output the output vector that will contain the edges in the minimum spanning tree
        inline void kruskal( DSU& dsu, std::vector<Edge>& edge_list, std::vector<Edge>& output ) {
            for ( const auto& edge : edge_list ) {
                if ( output.size() == num_data - 1 ) {
                    break;
                }
                add_edge( edge, dsu, output );
            }
        }

        /// @brief Generate a random spanning tree to have an initial solution
        void dirty_start( std::vector<Edge>& clean ) {
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
        /// @return the failure probability of the minimum spanning tree,
        float failure_probability( size_t i, size_t j ) {
            float loose_upper_bound =
                ( num_data - 1 ) * table.fail_probability(  top.back().weight , i, j );
            float prob = 0.0f;
            for ( auto& edge : top ) {
                prob += table.fail_probability( edge.weight , i, j );
            }
            expect( prob <= loose_upper_bound );
            LOG_DEBUG( "msg", "failure-probability",
                      "loose-upper-bound", loose_upper_bound,
                      "union-bound", prob );
            return prob;
        }

        /// @brief Compute an upper bound to the failure probability of the stored spanning tree
        ///
        /// Returns the total weight of the confirmed edges as well, along with the index of the last
        /// confirmed edges
        /// 
        /// @param i current concatenation in the hash index
        /// @param j current repetition in the hash index
        /// @return a tuple containing the failure probability of the minimum spanning tree,
        /// the total weight of the confirmed edges, and the index of the last confirmed edge
        StoppingConditionInfo stopping_condition( size_t i, size_t j ) {
            float prob = 0.0f;
            float weight = 0.0f;
            size_t idx = 0;
            while (prob < delta && idx < top.size()) {
                float w =  top[idx].weight ;
                prob += table.fail_probability( w, i, j );
                weight += w;
                idx += 1;
            }
            size_t edges_to_confirm = top.size() - idx;

            float total_weight = weight;
            for (size_t jj=idx; jj<top.size(); jj++) {
                float w =  top[jj].weight ;
                total_weight += w;
            }

            return StoppingConditionInfo{ .total_weight = total_weight,
                                          .confirmed_weight = weight,
                                          .heaviest_confirmed_edge =
                                              ( idx > 0 ) ?  top[idx - 1].weight  : 0.0f,
                                          .edges_to_confirm = edges_to_confirm,
                                          .confirmed_edges = idx };
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

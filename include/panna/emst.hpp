#pragma once
#include <algorithm>
#include <iostream>
#include <random>
#include <unistd.h>
#include <vector>

#include "panna/dsu.hpp"
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
        const float epsilon{ 0.01 };
        DSU dsu_true;
        DSU filter;
        // Sets for the confimed and the unconfirmed edges
        std::vector<EdgeTuple> top;
        std::vector<std::vector<EdgeTuple>> local_confirmed;
        std::vector<std::vector<EdgeTuple>> local_edges;
        std::vector<EdgeTuple> confirmed;
        std::vector<EdgeTuple> unconfirmed;
        float max_weight;

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
         * @param epsilon Approximation factor parameter (default: 0.01)
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
              const double delta_in = 0.2,
              const float epsilon = 0.2 ):
            dimensionality( dimensions ),
            table( Index<Dataset, Hasher, Distance>( dimensions, builder, repetitions ) ),
            data( data_in ),
            num_data( ( data ).size() ),
            epsilon( epsilon ),
            dsu_true( DSU( num_data ) ),
            filter( DSU( num_data ) ) {
            // Find the mean for each dimension and center the data
#pragma omp parallel for
            for ( size_t dim = 0; dim < dimensions; dim++ ) {
                float mean = 0;
                for ( size_t i = 0; i < num_data; i++ ) {
                    mean += data_in[i][dim];
                }
                mean /= num_data;
                for ( size_t i = 0; i < num_data; i++ ) {
                    data_in[i][dim] -= mean;
                }
            }

            // Insert the data
            for ( auto& point : data_in ) {
                table.insert( point.begin(), point.end() );
            }
            table.rebuild();

            // Get info on the index
            MAX_HASHBITS = table.num_concatenations();
            MAX_REPETITIONS = table.num_repetitions();
            delta = delta_in / num_data;
            max_weight = std::numeric_limits<float>::infinity();

            local_edges.resize( MAX_REPETITIONS );
            local_confirmed.resize( MAX_REPETITIONS );
            std::cout << "Index constructed, L: " << MAX_REPETITIONS << " K: " << MAX_HASHBITS
                      << " num data: " << num_data << std::endl;
            std::cout << "Delta: " << delta << std::endl;
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
                        std::make_tuple( dist, std::make_pair( i, j ) );
                }
            }
            // Sort the edges
            std::sort( all_edges.begin(), all_edges.end() );
            // Create the DSU
            DSU dsu( num_data );
            float tree_weight = 0;
            std::cout << "Creating the MST" << std::endl;
            std::vector<EdgeTuple> tree;
            for ( const auto& edge : all_edges ) {
                add_edge( edge, dsu, tree );
                if ( tree.size() == num_data - 1 ) {
                    break;
                }
            }
            for ( const auto& edge : tree ) {
                tree_weight += std::get<0>( edge );
            }
            return tree_weight;
        }

        /// @brief Find the Minimum Spanning Tree using only confirmed edges
        float find_tree() {
            clear();
            // dirty_start(local_confirmed[0]);
            float tree_weight = 0;
            std::vector<std::pair<unsigned int, unsigned int>> tree;

            bool found = false;
            std::vector<EdgeTuple> edges;
            for ( size_t i_rev = 0; i_rev <= MAX_HASHBITS; i_rev++ ) {
                size_t i = MAX_HASHBITS - i_rev;
                if ( found )
                    break;
                // QUESTION: if we do all the repetitions in parallel,
                // how can we evaluate the stopping condition? We need to know that
                // the previous iterations have been carried out
                //  If we stop as some iteration j, we will still carry out all previous ones
                //  as they are already dispatched. Now it may happen that one of the j'<j iterations
                //  finds a smaller edge for the mst that would confirm the tree at iteration j
                //  in this case there's just a delay in the confirmation.
                // If we remove the nowait, instead, we are guaranteed that
                // all the previous iterations have been carried out before checking the stopping condition
                //  https://ppc.cs.aalto.fi/ch3/nowait/
#pragma omp parallel
#pragma omp for nowait
                for ( size_t j = 0; j < MAX_REPETITIONS; j++ ) {
                    if ( found )
                        continue;
                    DSU local_dsu( num_data );
                    std::vector<EdgeTuple> local_top, local_Tu;
                    enumerate_edges( i, j, local_Tu );

                    for ( auto& edge : local_Tu ) {
                        if ( local_top.size() == num_data - 1 ) {
                            break;
                        }
                        add_edge( edge, local_dsu, local_top );
                    }

                    local_confirmed[j].insert( local_confirmed[j].end(),
                                               std::make_move_iterator( local_top.begin() ),
                                               std::make_move_iterator( local_top.end() ) );

                    // Every x iterations we have a batch, construct the MST from these edges
                    if ( ( i > 3 && ( ( j + 1 ) == MAX_REPETITIONS ||
                                      ( j + 1 ) == (size_t)MAX_REPETITIONS / 2 ) ) ||
                         ( i <= 3 && j % 5 == 0 ) ) {
#pragma omp critical
                        {
                            edges.insert( edges.end(),
                                          std::make_move_iterator( top.begin() ),
                                          std::make_move_iterator( top.end() ) );
                            top.clear();
                            dsu_true = DSU( num_data );
                            for ( size_t local_index = 0; local_index < j + 1; local_index++ ) {
                                auto& local = local_confirmed[local_index];
                                edges.insert( edges.end(), local.begin(), local.end() );
                                // std::make_move_iterator(local.begin()),
                                // std::make_move_iterator(local.end()) );
                                local.clear();
                            }
                            std::sort( edges.begin(), edges.end() );
                            // edges.erase( std::unique( edges.begin(), edges.end() ), edges.end()
                            // );
                            if ( edges.size() > num_data - 1 ) {
                                for ( const auto& edge : edges ) {
                                    add_edge( edge, dsu_true, top );
                                    if ( top.size() == num_data - 1 ) {
                                        break;
                                    }
                                }
                                std::cout << "Tree size: " << top.size() << std::endl;

                                // QUESTION: shouldn't we check that
                                // the tree is connected?
                                // If we add n-1 edges using DSU 
                                // isn't it guaranteed that the tree is connected?
                                if ( top.size() == num_data - 1 ) {
                                    float new_tree_weight = 0;
                                    max_weight = std::get<float>( top.back() );
                                    for ( const auto& edge : top ) {
                                        new_tree_weight += std::get<0>( edge );
                                    }
                                    tree_weight = new_tree_weight;
                                    std::cout << "Tree weight: " << tree_weight << std::endl;
                                    std::cout << "Probability: "
                                              << table.fail_probability(
                                                     std::get<float>( top.back() ), i, j )
                                              << std::endl;
                                    std::cout
                                        << "Max edge weight: " << std::get<float>( top.back() )
                                        << " Mean edge weight: " << tree_weight / ( num_data - 1 )
                                        << std::endl;
                                    if ( table.fail_probability(
                                             std::get<float>( top.back() ), i, j ) < delta ) {
                                        found = true;
                                        // Fill the tree
                                        for ( const auto& edge : top ) {
                                            tree.push_back( std::get<1>( edge ) );
                                        }
                                    }
                                }
                                // Lose the unused edges, MST is composable wrt to edge partitioning
                                // QUESTION: I keep getting not so sure that this works.
                                // We are not de-duplicating the edges, are we? Then it might be that
                                // some edges that we are clearing end up re-appearing later on,
                                // thus breaking the edge partition on which the composability relies.
                                // From what I'm getting, an edge can re-appear but its one of those things
                                // -edge in MST, Kruskal will ignore all copies of an edge already in the MST
                                // -edge out MST, in this case Kruskal will keep discarding the edge
                                // because it induces a cycle
                                // So our partitioning does not satisfy the composability condition inherently,
                                // but Kruskal makes up for it. (?)
                                edges.clear();
                            }
                        }
                    }
                }
                std::cout << "Finished prefix " << i << std::endl;
            }
            // QUESTION: why don't we use the union-find
            // data structure to check if it is connected? We might have a DSU::is_connected method
            // This is just a sanity check to see if dsu works as intended
            is_connected( tree );
            return tree_weight;
        }

        /// @brief Find the ɛ-EMST using both confirmed and unconfirmed edges
        float find_epsilon_tree() {
            clear();
            // dirty_start(local_confirmed[0]);
            float tree_weight = 0;
            std::vector<std::pair<unsigned int, unsigned int>> tree;

            bool found = false;
            std::vector<EdgeTuple> edges;
            for ( size_t i_rev = 0; i_rev <= MAX_HASHBITS; i_rev++ ) {
                size_t i = MAX_HASHBITS - i_rev;
                if ( found )
                    break;
#pragma omp parallel
#pragma omp for nowait
                for ( size_t j = 0; j < MAX_REPETITIONS; j++ ) {
                    if ( found )
                        continue;
                    DSU local_dsu( num_data );
                    std::vector<EdgeTuple> local_top, local_Tu;
                    enumerate_edges( i, j, local_Tu );

                    for ( auto& edge : local_Tu ) {
                        if ( local_top.size() == num_data - 1 ) {
                            break;
                        }
                        add_edge( edge, local_dsu, local_top );
                    }

                    local_confirmed[j].insert( local_confirmed[j].end(),
                                               std::make_move_iterator( local_top.begin() ),
                                               std::make_move_iterator( local_top.end() ) );

                    // Every x iterations we have a batch, construct the MST from these edges
                    if ( ( i > 3 && ( ( j + 1 ) == MAX_REPETITIONS ||
                                      ( j + 1 ) == (size_t)MAX_REPETITIONS / 2 ) ) ||
                         ( i <= 4 && j % 5 == 0 ) ) {
#pragma omp critical
                        {
                            edges.insert( edges.end(),
                                          std::make_move_iterator( top.begin() ),
                                          std::make_move_iterator( top.end() ) );
                            top.clear();
                            dsu_true = DSU( num_data );
                            for ( size_t local_index = 0; local_index < j + 1; local_index++ ) {
                                auto& local = local_confirmed[local_index];
                                edges.insert( edges.end(), local.begin(), local.end() );
                                // std::make_move_iterator(local.begin()),
                                // std::make_move_iterator(local.end()) );
                                local.clear();
                            }
                            std::sort( edges.begin(), edges.end() );
                            // edges.erase( std::unique( edges.begin(), edges.end() ), edges.end()
                            // );
                            if ( edges.size() > num_data - 1 ) {
                                for ( const auto& edge : edges ) {
                                    add_edge( edge, dsu_true, top );
                                    if ( top.size() == num_data - 1 ) {
                                        break;
                                    }
                                }
                                std::cout << "Tree size: " << top.size() << std::endl;

                                if ( top.size() == num_data - 1 ) {
                                    float new_tree_weight = 0;
                                    max_weight = std::get<float>( top.back() );
                                    for ( const auto& edge : top ) {
                                        new_tree_weight += std::get<0>( edge );
                                    }
                                    auto partition_point = std::partition_point(
                                        top.begin(), top.end(), [&]( const auto& e ) {
                                            return table.fail_probability(
                                                       std::get<float>( e ), i, j ) < delta;
                                        } );
                                    tree_weight = new_tree_weight;
                                    std::cout << "Tree weight: " << tree_weight << std::endl;
                                    float bound_weight =
                                        ( 1 + epsilon ) *
                                            std::accumulate( top.begin(),
                                                             partition_point,
                                                             0.0f,
                                                             []( float acc, const EdgeTuple& e ) {
                                                                 return acc + std::get<float>( e );
                                                             } ) +
                                        ( std::get<float>( *( partition_point - 1 ) ) *
                                          std::distance( partition_point, top.end() ) );
                                    std::cout << "Bound weight: " << bound_weight << std::endl;
                                    if ( tree_weight <= bound_weight ) {
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
                std::cout << "Finished prefix " << i << std::endl;
            }
            is_connected( tree );
            return tree_weight;
        }

        std::pair<
        std::vector<std::tuple<float, float, float>>,
        std::vector< std::vector< std::pair<float, unsigned int>>> > find_tree_dbscan( size_t k=5 ) {
            clear();
            // dirty_start(local_confirmed[0]);
            float tree_weight = 0;
            std::vector<std::tuple<float, float, float>> tree;
            std::vector< std::vector< std::pair<float, unsigned int>>> neighbors (num_data, std::vector< std::pair<float, unsigned int> >() );

            bool found = false;
            std::vector<EdgeTuple> edges;
            // Find the top-k nearest neighbors for each node
            for ( size_t index = 0; index < data.size(); index++ ) {
                table.search( data[index].begin(), data[index].end(), k+1, 0.1, neighbors[index] );
                // Remove the self loop
            }

            for ( size_t i_rev = 0; i_rev <= MAX_HASHBITS; i_rev++ ) {
                size_t i = MAX_HASHBITS - i_rev;
                if ( found )
                    break;

#pragma omp parallel
#pragma omp for nowait
                for ( size_t j = 0; j < MAX_REPETITIONS; j++ ) {
                    if ( found )
                        continue;
                    DSU local_dsu( num_data );
                    std::vector<EdgeTuple> local_top, local_Tu;
                    enumerate_edges( i, j, local_Tu );

                    for ( auto& edge : local_Tu ) {
                        // Use the reachability
                        std::get<float>( edge ) = std::max( 
                            std::get<float>( edge ),
                            std::max( neighbors[ std::get<1>( edge ).first ].front().first, //Remember it's a heap
                                      neighbors[ std::get<1>( edge ).second ].front().first ) );
                        if ( local_top.size() == num_data - 1 ) {
                            break;
                        }
                        add_edge( edge, local_dsu, local_top );
                    }

                    local_confirmed[j].insert( local_confirmed[j].end(),
                                               std::make_move_iterator( local_top.begin() ),
                                               std::make_move_iterator( local_top.end() ) );

                    // Every x iterations we have a batch, construct the MST from these edges
                    if ( ( i > 3 && ( ( j + 1 ) == MAX_REPETITIONS ||
                                      ( j + 1 ) == (size_t)MAX_REPETITIONS / 2 ) ) ||
                         ( i <= 4 && j % 5 == 0 ) ) {
#pragma omp critical
                        {   

                            edges.insert( edges.end(),
                                          std::make_move_iterator( top.begin() ),
                                          std::make_move_iterator( top.end() ) );
                            top.clear();
                            dsu_true = DSU( num_data );
                            for ( size_t local_index = 0; local_index < j + 1; local_index++ ) {
                                auto& local = local_confirmed[local_index];
                                edges.insert( edges.end(), local.begin(), local.end() );
                                // std::make_move_iterator(local.begin()),
                                // std::make_move_iterator(local.end()) );
                                local.clear();
                            }
                            std::sort( edges.begin(), edges.end() );
                            // edges.erase( std::unique( edges.begin(), edges.end() ), edges.end()
                            // );
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
                                    auto partition_point = std::partition_point(
                                        top.begin(), top.end(), [&]( const auto& e ) {
                                            return table.fail_probability(
                                                       std::get<float>( e ), i, j ) < delta;
                                        } );
                                    tree_weight = new_tree_weight;
                                    float bound_weight =
                                        ( 1 + epsilon ) *
                                            std::accumulate( top.begin(),
                                                             partition_point,
                                                             0.0f,
                                                             []( float acc, const EdgeTuple& e ) {
                                                                 return acc + std::get<float>( e );
                                                             } ) +
                                        ( std::get<float>( *( partition_point - 1 ) ) *
                                          std::distance( partition_point, top.end() ) );
                                          std::cout << "Bound weight: " << bound_weight << ", Tree weight: " << tree_weight << std::endl;
                                    if ( tree_weight <= bound_weight ) {
                                        found = true;
                                        // Fill the tree
                                        for ( const auto& edge : top ) {
                                            tree.emplace_back( (float) std::get<1>( edge ).first,
                                                               (float) std::get<1>( edge ).second,
                                                               (float) std::get<0>( edge ) );
                                        }
                                    }
                                }
                                // Lose the unused edges, MST is composable wrt to edge partitioning
                                edges.clear();
                            }
                        }
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
            table.search_pairs_filter( j, i, couples, max_weight );
            Tu_local.insert( Tu_local.end(), couples.begin(), couples.end() );
            return;
        };

        /// @brief Return the bound weight (1+ɛ)(sum over Tc + |Tu|*max(Tu) )
        /// @param top_copy a vector that contains the edges in the spanning tree
        /// @return the weight
        float bound_weight( const std::vector<EdgeTuple>& top_copy, size_t i, size_t j ) {
            float weight = 0;
            float max_confirmed = 0;
            int unconfirmed = 0;
            auto split_point =
                std::partition_point( top_copy.begin(), top_copy.end(), [&]( const auto& e ) {
                    return table.fail_probability( std::get<float>( e ), i, j ) < delta;
                } );
            max_confirmed = std::get<float>( *( split_point - 1 ) );
            unconfirmed = std::distance( split_point, top_copy.end() );
            weight += std::accumulate(
                top_copy.begin(), split_point, 0.0f, []( float acc, const EdgeTuple& e ) {
                    return acc + std::get<float>( e );
                } );

            weight += max_confirmed * unconfirmed;
            return weight;
        }

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
                std::cout << "Not connected" << std::endl;
                return false;
            }
            std::cout << "Connected" << std::endl;
            return true;
        };

        /// @brief Add the edge to the tree if it does not create a cycle using the DSU data
        /// structure
        /// @param new_edge_input the edge that we have to add
        /// @param dsu the data structure that keeps track of the connected components
        /// @param edge_list the current edges in the tree
        /// @return true if an edge has been added to the edge_list and the DSU data structure,
        /// false otherwise
        bool
        add_edge( const EdgeTuple& new_edge_input, DSU& dsu, std::vector<EdgeTuple>& edge_list ) {
            // Extract new edge and its weight.
            std::pair<uint32_t, uint32_t> new_edge = std::get<1>( new_edge_input );

            // Try to add new edge normally.
            if ( dsu.union_sets( new_edge.first, new_edge.second ) ) {
                edge_list.push_back( new_edge_input );
                return true;
            }
            return false;
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
        }
    }; // closes class
} // namespace panna

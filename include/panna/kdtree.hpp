#pragma once
#include <cmath>
#include <vector>

#include "panna/data.hpp"
// Adapted from https://github.com/cdalitz/kdtree-cpp
// This is now a random projection tree

namespace panna {

    template <typename Dataset, typename Distance>
    class KDTree {
        using PointHandle = typename Dataset::PointHandle;

    public:
        /// Build tree from [begin,end) indices, given the full dataset & radius
        KDTree( const uint32_t* begin, const uint32_t* end, const Dataset& dataset, float radius ):
            dataset_( dataset ),
            radius2_( radius * radius ),
            radius_( radius ),
            dim_( dataset_[0].inner.dimensions ) {
            indices_.assign( begin, end );
            nodes_.reserve( indices_.size() * 2 );
            root_ = build_( 0, indices_.size(), 0 );
        }

        /// Collect all pairs (dist, {idxA, idxB}) that pass the filter
        void
        range_pairs( std::vector<std::tuple<float, std::pair<uint32_t, uint32_t>>>& out ) const {
            if ( root_ != NO_NODE ) { // Check if the tree is not empty
                enumerate_( root_, root_, /*same=*/true, out );
            }
        }

    private:
        static constexpr size_t NO_NODE = std::numeric_limits<size_t>::max();
        // Node struct
        struct Node {
            size_t l, r;                 // half-open range inside indices_
            int axis;                    // split axis (-1 for leaf)
            float split;                 // split coordinate
            size_t left = NO_NODE;       // index of left child in nodes_ vector (NO_NODE for null)
            size_t right = NO_NODE;      // index of right child in nodes_ vector (NO_NODE for null)
            std::vector<float> min, max; // bounding box

            explicit Node( size_t l_, size_t r_, int dims ):
                l( l_ ),
                r( r_ ),
                axis( -1 ),
                split( 0.f ),
                min( dims, std::numeric_limits<float>::infinity() ),
                max( dims, -std::numeric_limits<float>::infinity() ) {
            }

            bool leaf() const {
                return axis == -1;
            }
            size_t size() const {
                return r - l;
            }
        };

        // Tree
        static constexpr size_t LEAF_SIZE = 16;

        size_t build_( size_t l, size_t r, int depth ) {
            size_t node_idx = nodes_.size();
            nodes_.emplace_back( l, r, dim_ );

            // Bounding box
            for ( size_t i = l; i < r; i++ ) {
                for ( int d = 0; d < dim_; d++ ) {
                    float v = coord_( indices_[i], d );
                    nodes_[node_idx].min[d] = std::min( nodes_[node_idx].min[d], v );
                    nodes_[node_idx].max[d] = std::max( nodes_[node_idx].max[d], v );
                }
            }

            if ( r - l <= LEAF_SIZE ) {
                return node_idx;
            }

            int axis = std::rand() % dim_;
            float widest = nodes_[node_idx].max[axis] - nodes_[node_idx].min[axis];
            nodes_[node_idx].axis = axis;

            size_t m = ( l + r ) >> 1; // Median
            std::nth_element(
                indices_.begin() + l,
                indices_.begin() + m,
                indices_.begin() + r,
                [&]( uint32_t a, uint32_t b ) { return coord_( a, axis ) < coord_( b, axis ); } );

            nodes_[node_idx].split = coord_( indices_[m], axis, widest );

            nodes_[node_idx].left = build_( l, m, depth + 1 );
            nodes_[node_idx].right = build_( m, r, depth + 1 );

            return node_idx;
        }

        // Coordinates
        float coord_( uint32_t idx, int d, float delta = 0.0f ) const {
            return dataset_[idx]
                           .inner.chunks[d / Int16Chunk::CHUNK_SIZE]
                           .chunk[d % Int16Chunk::CHUNK_SIZE] /
                       32768.0f +
                   delta;
        }

        /// min squared distance between two bounding boxes
        float min_dist_sq_( size_t a_idx, size_t b_idx ) const {
            return Distance::compute( dataset_[a_idx], dataset_[b_idx] );

        }

        // Enumerate pairs of points within the radius
        void
        enumerate_( size_t a_idx,
                    size_t b_idx,
                    bool same,
                    std::vector<std::tuple<float, std::pair<uint32_t, uint32_t>>>& out ) const {
            if ( a_idx == NO_NODE || b_idx == NO_NODE )
                return;
            if ( min_dist_sq_( a_idx, b_idx ) > radius_ )
                return;

            const auto& a_node = nodes_[a_idx];
            const auto& b_node = nodes_[b_idx];

            if ( same && a_node.leaf() ) { // ① Leaf vs itself
                for ( size_t i = a_node.l; i < a_node.r; i++ )
                    for ( size_t j = i + 1; j < a_node.r; j++ )
                        maybe_emit_( indices_[i], indices_[j], out );
                return;
            }
            if ( !same && a_node.leaf() && b_node.leaf() ) { // ② Two different leaves
                for ( size_t i = a_node.l; i < a_node.r; i++ )
                    for ( size_t j = b_node.l; j < b_node.r; j++ )
                        maybe_emit_( indices_[i], indices_[j], out );
                return;
            }

            // Split the largest subtree
            if ( same ) {
                enumerate_( a_node.left, a_node.left, true, out );
                enumerate_( a_node.left, a_node.right, false, out );
                enumerate_( a_node.right, a_node.right, true, out );
            } else {
                // Always split the larger side to guarantee O(n log n)
                if ( a_node.leaf() || ( !b_node.leaf() && b_node.size() > a_node.size() ) ) {
                    enumerate_( a_idx, b_node.left, false, out );
                    enumerate_( a_idx, b_node.right, false, out );
                } else {
                    enumerate_( a_node.left, b_idx, false, out );
                    enumerate_( a_node.right, b_idx, false, out );
                }
            }
        }

        void
        maybe_emit_( uint32_t i,
                     uint32_t j,
                     std::vector<std::tuple<float, std::pair<uint32_t, uint32_t>>>& out ) const {
            float d2 = Distance::compute( dataset_[i], dataset_[j] );
            if ( d2 <= radius_ )
                out.emplace_back( d2, std::make_pair( i, j ) );
        }

        // Data
        std::vector<uint32_t> indices_;
        const Dataset& dataset_;
        float radius2_;
        float radius_;
        int dim_;
        std::vector<Node> nodes_; // Node storage
        size_t root_ = NO_NODE;   // Index of the root node
    };

} // namespace panna

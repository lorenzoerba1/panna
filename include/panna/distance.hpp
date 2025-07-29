#pragma once

#include <cmath>
#include <type_traits>
#include <iostream>

#include "panna/linalg.hpp"

namespace panna {

    struct CosineDistance {
        // The cosine distance is not a metric!
        static constexpr bool is_metric() {
            return false;
        };

        //! Works on any `Point` type for which `panna::dot_product` is
        //! implemented. Assumes (but does not check) that the points have
        //! unit-norm.
        template <typename Point>
        static float compute( Point a, Point b ) {
            float dot = panna::dot_product( a, b );
            assert( dot <= 1.0 );
            return 1 - dot;
        }

        static constexpr float to_angle( float distance ) {
            float dot = 1 - distance;
            return std::acos( dot );
        }

        static constexpr float to_dot_product( float distance ) {
            float dot = 1 - distance;
            return dot;
        }
    };

    struct AngularDistance {
        static constexpr bool is_metric() {
            return true;
        };

        //! Works on any `Point` type for which `panna::dot_product` is
        //! implemented. Assumes (but does not check) that the points have
        //! unit-norm.
        template <typename Point>
        static float compute( Point a, Point b ) {
            float dot = panna::dot_product( a, b );
            assert( dot <= 1.0 );
            return std::acos( dot );
        }

        static constexpr float to_angle( float distance ) {
            return distance;
        }
        static constexpr float to_dot_product( float distance ) {
            return std::cos( distance );
        }
    };

    struct EuclideanDistance {
        static constexpr bool is_metric() {
            return true;
        };

        //! Works on any `Point` type for which `panna::dot_product` is
        //! implemented and that has a `squared_norm` method.
        template <typename Point>
        static float compute( Point a, Point b ) {
            float dot = panna::dot_product( a, b );
            return std::sqrt( a.squared_norm() + b.squared_norm() - 2 * dot );
        }

        template <typename Point>
        static float compute_nosq( Point a, Point b ) {
            float dot = panna::dot_product( a, b );
            return a.squared_norm() + b.squared_norm() - 2 * dot;
        }
    };

    //! Defined as 1 - (jaccard similarity)
    struct JaccardDistance {
        static constexpr bool is_metric() {
            return true;
        };

        //! Works on any `Set` type that has both `Set::intersection_size` and
        //! `Set::size`.
        template <typename Set>
        static float compute( Set a, Set b ) {
            float intersection = a.intersection_size( b );
            return 1.0 - intersection / ( a.size() + b.size() - intersection );
        }
    };
} // namespace panna

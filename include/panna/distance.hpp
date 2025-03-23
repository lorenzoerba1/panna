#pragma once

#include <cmath>
#include <type_traits>

#include "panna/linalg.hpp"

namespace panna {

    //! Use to check if a function in this module is a metric (i.e. the
    //! triangle inequality holds and the usual stuff). By default this is false for any type,
    //! then we can specialize it for distances for which this is true
    template <typename Distance>
    struct is_metric : std::false_type {};

    struct CosineDistance {
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
    // The cosine distance is not a metric!
    template <>
    struct is_metric<CosineDistance> : std::false_type {};

    struct AngularDistance {
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
    template <>
    struct is_metric<AngularDistance> : std::true_type {};

    struct EuclideanDistance {
        //! Works on any `Point` type for which `panna::dot_product` is
        //! implemented and that has a `squared_norm` method.
        template <typename Point>
        static float compute( Point a, Point b ) {
            float dot = panna::dot_product( a, b );
            return std::sqrt( a.squared_norm() + b.squared_norm() - 2 * dot );
        }
    };
    template <>
    struct is_metric<EuclideanDistance> : std::true_type {};

    //! Defined as 1 - (jaccard similarity)
    struct JaccardDistance {
        //! Works on any `Set` type that has both `Set::intersection_size` and
        //! `Set::size`.
        template <typename Set>
        static float compute( Set a, Set b ) {
            float intersection = a.intersection_size( b );
            return 1.0 - intersection / ( a.size() + b.size() - intersection );
        }
    };
    template <>
    struct is_metric<JaccardDistance> : std::true_type {};
} // namespace panna

#pragma once

#include "panna/linalg.hpp"

namespace panna {
    struct AngularDistance {
        //! Works on any `Point` type for which `panna::dot_product` is
        //! implemented. Assumes (but does not check) that the points have
        //! unit-norm.
        template <typename Point>
        static float compute( Point a, Point b ) {
            float dot = panna::dot_product( a, b );
            assert( dot <= 1.0 );
            return 1 - dot; // Ensure the distance is positive
        }
    };

    struct EuclideanDistance {
        //! Works on any `Point` type for which `panna::dot_product` is
        //! implemented and that has a `squared_norm` method.
        template <typename Point>
        static float compute( Point a, Point b ) {
            float dot = panna::dot_product( a, b );
            return std::sqrt( a.squared_norm() + b.squared_norm() - 2 * dot );
        }
    };

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
} // namespace panna

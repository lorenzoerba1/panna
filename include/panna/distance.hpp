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
            assert(dot <= 1.0);
            return 1 - dot; // Ensure the distance is positive
        }
    };
} // namespace panna

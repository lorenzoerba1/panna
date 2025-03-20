#pragma once

#include <cmath>
#include <cstddef>

#include "panna/lsh/tensoring.hpp"
#include "dbg.h"

namespace panna {

    //! Returns the probability for a pair at `distance` of not colliding, under
    //! the given hasher, in any repetition out of `max_repetitions`, where
    //! `repetitions` are done with `concatenations`, and the rest are done at
    //! `concatenations+1`.
    template <typename Hasher>
    static float failure_probability( Hasher& hasher,
                                      float distance,
                                      size_t concatenations,
                                      size_t repetitions,
                                      size_t max_repetitions ) {
        float collision_probability = hasher.collision_probability( distance );
        float p_cur =
            std::pow( 1 - std::pow( collision_probability, concatenations ),
                      repetitions );
        float p_prev =
            std::pow( 1 - std::pow( collision_probability, concatenations + 1 ),
                      max_repetitions - repetitions );
        return p_cur * p_prev;
    }

    template <typename InnerHasher, typename Dataset>
    static float failure_probability( Tensoring<InnerHasher, Dataset>& hasher,
                                      float distance,
                                      size_t concatenations,
                                      size_t repetitions,
                                      size_t max_repetitions ) {
        auto cur_left_concatenations = ( concatenations + 1 ) / 2;
        auto cur_right_concatenations = concatenations - cur_left_concatenations;

        auto last_left_concatenations = ( concatenations + 2 ) / 2;
        auto last_right_concatenations = concatenations + 1 - last_left_concatenations;

        auto cur_repetitions = std::floor( std::sqrt( repetitions ) );
        auto last_repetitions = std::floor( std::sqrt( max_repetitions ) ) - cur_repetitions;

        auto left_prob = std::pow(hasher.collision_probability(distance), cur_left_concatenations);
        auto left_last_prob = std::pow(hasher.collision_probability(distance), last_left_concatenations);

        auto right_prob = std::pow(hasher.collision_probability(distance), cur_right_concatenations);
        auto right_last_prob = std::pow(hasher.collision_probability(distance), last_right_concatenations);

        auto cur_upper_left_prob =
            1.0 - std::pow( 1.0 - left_prob, cur_repetitions );
        auto last_upper_left_prob =
            1.0 - std::pow( 1.0 - left_last_prob, cur_repetitions );
        auto last_lower_left_prob =
            1.0 - std::pow( 1.0 - left_last_prob, last_repetitions );
        auto cur_upper_right_prob =
            1.0 - std::pow( 1.0 - right_prob, cur_repetitions );
        auto last_upper_right_prob =
            1.0 - std::pow( 1.0 - right_last_prob, cur_repetitions );
        auto last_lower_right_prob =
            1.0 - std::pow( 1.0 - right_last_prob, last_repetitions );
        return ( 1 - cur_upper_left_prob * cur_upper_right_prob ) *
               ( 1 - last_upper_left_prob * last_upper_right_prob ) *
               ( 1 - last_lower_left_prob * last_upper_right_prob ) *
               ( 1 - last_lower_left_prob * last_lower_right_prob );
    }
} // namespace panna

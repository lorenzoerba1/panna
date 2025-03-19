#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace panna {
    // Helper function for getting indices to tensor.
    //
    // Retrieve the pair of indices where both sides are incremented as little
    // as possible. The rhs index is incremented first Eg. (0, 0) (0, 1) (1, 0)
    // (1, 1) (0, 2)
    static std::pair<unsigned int, unsigned int>
    get_minimal_index_pair( int idx ) {
        int sqrt = static_cast<int>( std::sqrt( idx ) );
        if ( idx == sqrt * sqrt + 2 * sqrt ) {
            return { sqrt, sqrt };
        } else if ( idx >= sqrt * sqrt + sqrt ) {
            return { sqrt, idx - ( sqrt * sqrt + sqrt ) };
        } else { // idx >= sqrt*sqrt, always true
            return { idx - sqrt * sqrt, sqrt };
        }
    }

    template <typename Hasher, typename Dataset>
    class Tensoring {
    public:
        using Value = typename Hasher::Value::DoubleWidth;

    private:
        size_t repetitions;
        Hasher inner;

    public:
        template <typename HasherBuilder>
        Tensoring( HasherBuilder builder, size_t repetitions ):
            repetitions( repetitions ), inner( builder.build( repetitions ) ) {}

        static constexpr size_t get_concatenations() { return 2 * Hasher::get_concatenations(); }

        size_t get_repetitions() const { return repetitions; }

        void hash( typename Dataset::PointHandle point,
                   std::vector<Value>& output ) const {
            using HalfValue = typename Hasher::Value;
            // In order to avoid allocating a new vector to hold the tensored
            // data every time we hash something, we make the output vector a
            // little bit larger: enough to store both the output **and** the
            // tensored repetitions. After computing the tensored repetitions
            // directly in the output vector, we move them to the end, and place
            // the actual output hashes in the first part of `output`. Finally,
            // we trim the size so that the caller does not see the scratch
            // work. Note that calling `resize` does not de-allocate the memory,
            // so on the next call we will not make an allocation again.
            inner.hash( point, output );
            size_t tensored_hashers = output.size();
            output.resize( repetitions + tensored_hashers );
            // move to the end, see comment above
            for ( size_t i = 0; i < tensored_hashers; i++ ) {
                output[repetitions + i] = output[i];
            }
            size_t right_start = tensored_hashers / 2;

            for ( size_t rep = 0; rep < repetitions; rep++ ) {
                auto index_pair = get_minimal_index_pair( rep );
                HalfValue h_left = output[repetitions + index_pair.first];
                HalfValue h_right =
                    output[repetitions + right_start + index_pair.second];
                output[rep] = Value::interleave(h_left, h_right);
            }
            // hide the scratch space
            output.resize( repetitions );
        }

        float collision_probability( float distance ) const {
            return inner.collision_probability(distance);
        }
    };
} // namespace panna

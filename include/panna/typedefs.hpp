#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <type_traits>

namespace panna {
    //! Places zeros between the bits of the given value.
    //! Only the 16 less significant bits will appear in the output.
    //! The least significant bit will remain where it is.
    constexpr static uint32_t intersperse_zero(uint32_t val) {
        uint32_t mask = 1;
        uint32_t shift = 0;

        uint32_t res = 0;
        for (unsigned i=0; i < sizeof(uint32_t)*8/2; i++) {
            res |= (val & mask) << shift;
            mask <<= 1;
            shift++;
        }
        return res;
    }
    // An example of a call to the function above
    static_assert(intersperse_zero(0xFF) == 0x5555, "");

    // clang-format off
    static constexpr uint32_t BITWISE_HASH_MASKS[] = {
        0b00000000000000000000000000000000,
        0b10000000000000000000000000000000,
        0b11000000000000000000000000000000,
        0b11100000000000000000000000000000,
        0b11110000000000000000000000000000,
        0b11111000000000000000000000000000,
        0b11111100000000000000000000000000,
        0b11111110000000000000000000000000,
        0b11111111000000000000000000000000,
        0b11111111100000000000000000000000,
        0b11111111110000000000000000000000,
        0b11111111111000000000000000000000,
        0b11111111111100000000000000000000,
        0b11111111111110000000000000000000,
        0b11111111111111000000000000000000,
        0b11111111111111100000000000000000,
        0b11111111111111110000000000000000,
        0b11111111111111111000000000000000,
        0b11111111111111111100000000000000,
        0b11111111111111111110000000000000,
        0b11111111111111111111000000000000,
        0b11111111111111111111100000000000,
        0b11111111111111111111110000000000,
        0b11111111111111111111111000000000,
        0b11111111111111111111111100000000,
        0b11111111111111111111111110000000,
        0b11111111111111111111111111000000,
        0b11111111111111111111111111100000,
        0b11111111111111111111111111110000,
        0b11111111111111111111111111111000,
        0b11111111111111111111111111111100,
        0b11111111111111111111111111111110,
        0b11111111111111111111111111111111,
    };
    // clang-format on

    template <uint8_t K>
    struct BitwiseLshValue {
    private:
        //! The bits of this hash value
        uint32_t bits;

    public:
        //! How many concatenated hash values are stored in this value?
        constexpr static uint8_t num_hashes() { return K; };

        // we use a factory function rather than a constructor to keep this
        // struct Plain Old Data.
        constexpr static BitwiseLshValue<K> make( uint32_t bits ) {
            BitwiseLshValue<K> val;
            val.bits = bits;
            return val;
        }

        //! Check if the given hash value and `this` have the same prefix of
        //! bits, with the understanding that the higher order 32-BITS bits are
        //! not used, hence the prefix starts from that position.
        inline bool prefix_eq( BitwiseLshValue<K> other,
                               uint8_t prefix ) const {
            auto idx = 32 - ( K - prefix );
            assert( idx <= 32 );
            uint32_t mask = BITWISE_HASH_MASKS[idx];
            return ( this->bits & mask ) == ( other.bits & mask );
        }

        //! Check if the prefix of `this` is `<` than the prefix of `other`.
        //! with the understanding that the higher order 32-BITS bits are not
        //! used, hence the prefix starts from that position.
        inline bool prefix_less( BitwiseLshValue<K> other,
                                 uint8_t prefix ) const {
            auto idx = 32 - ( K - prefix );
            assert( idx <= 32 );
            uint32_t mask = BITWISE_HASH_MASKS[idx];
            return ( this->bits & mask ) < ( other.bits & mask );
        }

        //! Hash values are ordered lexicographically, from the most to the
        //! least significant bits.
        constexpr inline bool operator<( BitwiseLshValue<K> other ) const {
            return this->bits < other.bits;
        }

        //! This equality operator considers all the bits.
        constexpr inline bool operator==( BitwiseLshValue<K> other ) const {
            return this->bits == other.bits;
        }

        constexpr inline BitwiseLshValue<2*K>
        interleave( BitwiseLshValue<K> other ) {
            uint32_t abits = intersperse_zero(this->bits);
            uint32_t bbits = intersperse_zero(other.bits);
            uint32_t res = (abits << 1) | bbits;
            printf("%x\n", res);
            return BitwiseLshValue<2*K>::make(res);
        }

    };

    // simple static tests on the hash data type, checked at compile time
    static_assert( sizeof( BitwiseLshValue<24> ) == 4,
                   "BitwiseLshValue must be 4 bytes" );
    static_assert( std::is_pod<BitwiseLshValue<24>>(),
                   "BitwiseLshValue must be Plain Old Data" );
    static_assert( std::is_trivially_copyable<BitwiseLshValue<24>>(),
                   "BitwiseLshValue must be trivially copiable" );
    static_assert( std::is_trivial<BitwiseLshValue<24>>(),
                   "BitwiseLshValue must be trivial" );

    template <uint8_t K>
    struct BytewiseLshValue {
    private:
        //! The hash values
        uint8_t hashes[K];

    public:
        //! How many concatenated hash values are stored in this value?
        constexpr static uint8_t num_hashes() { return K; };

        // we use a factory function rather than a constructor to keep this
        // struct Plain Old Data.
        constexpr static BytewiseLshValue<K> make( uint8_t bytes[K] ) {
            BytewiseLshValue<K> hash;
            hash.hashes = bytes;
            return hash;
        }

        inline bool prefix_eq( BytewiseLshValue<K> other,
                               uint8_t prefix ) const {
            assert( prefix <= K );
            for ( uint8_t i = 0; i < prefix; i++ ) {
                if ( hashes[i] != other.hashes[i] ) { return false; }
            }
            return true;
        }

        inline bool prefix_less( BytewiseLshValue<K> other,
                                 uint8_t prefix ) const {
            // OPTIMIZE: maybe we can do it with SIMD, but probably the compiler
            // is smart enough to figure out on its own.
            assert( prefix <= K );
            for ( uint8_t i = 0; i < prefix; i++ ) {
                if ( !( hashes[i] < other.hashes[i] ) ) { return false; }
            }
            return true;
        }

        constexpr inline bool operator<( BytewiseLshValue<K> other ) const {
            return this->hashes < other.hashes;
        }

        constexpr inline bool operator==( BytewiseLshValue<K> other ) const {
            return this->hashes == other.hashes;
        }

        static constexpr inline BytewiseLshValue<K>
        interleave( BytewiseLshValue<K / 2> a, BytewiseLshValue<K / 2> b ) {
            static_assert( K % 2 == 0, "K should be even" );
            BytewiseLshValue<K> out;
            for ( size_t i = 0; i < K / 2; i++ ) {
                out[2 * i] = a[i];
                out[2 * i + 1] = b[i];
            }
            return out;
        }
    };

    static_assert( sizeof( BytewiseLshValue<8> ) == 8,
                   "BytewiseLshValue must use exactly K bytes" );
    static_assert( std::is_pod<BytewiseLshValue<8>>(),
                   "BytewiseLshValue must be Plain Old Data" );
    static_assert( std::is_trivially_copyable<BytewiseLshValue<8>>(),
                   "BytewiseLshValue must be trivially copiable" );
    static_assert( std::is_trivial<BytewiseLshValue<8>>(),
                   "BytewiseLshValue must be trivial" );
} // namespace panna

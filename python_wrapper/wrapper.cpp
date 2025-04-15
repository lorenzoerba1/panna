#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "dbg.h"
#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/trieindex.hpp"

namespace nb = nanobind;

struct AbstractIndex {
    virtual void rebuild() = 0;
    virtual void insert( const nb::ndarray<float, nb::shape<-1>>& vec ) = 0;
    virtual std::vector<uint32_t>
    search( const nb::ndarray<float, nb::shape<-1>>& vec, unsigned int k, float recall ) = 0;
    virtual ~AbstractIndex() = 0;
};

template <typename Dataset, typename Hasher, typename Distance>
struct ConcreteIndex : AbstractIndex {
    panna::Index<Dataset, Hasher, Distance> inner;

    ConcreteIndex( panna::Index<Dataset, Hasher, Distance> inner ): inner( inner ) {
    }

    void rebuild() {
        inner.rebuild();
    }

    void insert( const nb::ndarray<float, nb::shape<-1>>& vec ) {
        // TODO: handle both one and two dimensional vectors
        inner.insert( vec.data(), vec.data() + vec.shape( 0 ) );
    }

    std::vector<uint32_t>
    search( const nb::ndarray<float, nb::shape<-1>>& vec, unsigned int k, float recall ) {
        std::vector<std::pair<float, uint32_t>> res;
        float delta = 1 - recall;
        inner.search( vec.data(), vec.data() + vec.shape( 0 ), k, delta, res );

        std::vector<uint32_t> out;
        return out;
    }
};

struct TrieIndex {
    std::unique_ptr<AbstractIndex> inner;

    TrieIndex( size_t dimensions, std::string distance, nb::kwargs kwargs ) {
        dbg( dimensions );
        dbg( distance );
        size_t repetitions = 64;
        if ( kwargs.contains( "repetitions" ) ) {
            repetitions = nb::cast<size_t>( kwargs["repetitions"] );
        }
        if ( distance == "cosine" ) {
            std::string hasher = "crosspolytope";
            if ( kwargs.contains( "hasher" ) ) {
                hasher = nb::cast<std::string>( kwargs["hasher"] );
            }
            dbg( hasher );
            if ( hasher == "crosspolytope" ) {
                using Builder =
                    panna::CrossPolytopeBuilder<3, panna::UnitNormPoints, panna::CosineDistance>;
                Builder builder( dimensions );
                panna::Index<panna::UnitNormPoints, Builder::Output, panna::CosineDistance> index(
                    dimensions, builder, repetitions );
                inner = std::make_unique<
                    ConcreteIndex<panna::UnitNormPoints, Builder::Output, panna::CosineDistance>>(
                    index );
            }
        } else if ( distance == "euclidean" ) {

        } else {
            throw nb::value_error( "Unsupported distance metric" );
        }
    }
};

NB_MODULE( panna, m ) {
    nb::class_<TrieIndex>( m, "TrieIndex" ).def( nb::init<size_t, std::string, nb::kwargs>() );
}

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "dbg.h"
#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/trieindex.hpp"

namespace nb = nanobind;

struct AbstractIndex {
    virtual void rebuild() = 0;
    virtual void insert( const nb::ndarray<float, nb::c_contig>& vec ) = 0;
    virtual nb::ndarray<uint32_t, nb::numpy, nb::ndim<1>>
    search( const nb::ndarray<float, nb::shape<-1>>& vec, unsigned int k, float recall ) = 0;
};

template <typename Dataset, typename Hasher, typename Distance>
struct ConcreteIndex final : AbstractIndex {
    panna::Index<Dataset, Hasher, Distance> inner;

    ConcreteIndex( panna::Index<Dataset, Hasher, Distance> inner ): inner( inner ) {
    }

    void rebuild() {
        inner.rebuild();
    }

    void insert( const nb::ndarray<float, nb::c_contig>& vec ) {
        // TODO: handle both one and two dimensional vectors
        if ( vec.ndim() == 1 ) {
            inner.insert( vec.data(), vec.data() + vec.shape( 0 ) );
        } else if ( vec.ndim() == 2 ) {
            size_t nrows = vec.shape( 0 );
            size_t dimensionality = vec.shape( 1 );
            float* data = vec.data();
            for ( size_t row = 0; row < nrows; row++ ) {
                float* begin = data + row * dimensionality;
                float* end = data + ( row + 1 ) * dimensionality;
                expect( end - begin == static_cast<ptrdiff_t>( dimensionality ) );
                inner.insert( begin, end );
            }
        }
    }

    nb::ndarray<uint32_t, nb::numpy, nb::ndim<1>>
    search( const nb::ndarray<float, nb::shape<-1>>& vec, unsigned int k, float recall ) {
        std::vector<std::pair<float, uint32_t>> res;
        res.reserve( k + 1 );
        float delta = 1 - recall;
        size_t dimensions = vec.shape( 0 );
        float* data = vec.data();

        inner.search( data, data + dimensions, k, delta, res );
        expect( res.size() == k );

        std::sort( res.begin(), res.end() );
        uint32_t* out_data = new uint32_t[k];

        for ( size_t i = 0; i < k; i++ ) {
            out_data[i] = res[i].second;
        }

        // Delete 'data' when the 'owner' capsule expires
        nb::capsule owner( out_data, []( void* p ) noexcept { delete[] (float*)p; } );

        return nb::ndarray<uint32_t, nb::numpy, nb::ndim<1>>(
            /* data = */ out_data,
            /* shape = */ { k },
            /* owner = */ owner );
    }
};

struct TrieIndex {
    std::unique_ptr<AbstractIndex> inner;

    TrieIndex( size_t dimensions, std::string distance, nb::kwargs kwargs ) {
        size_t repetitions = 64;
        if ( kwargs.contains( "repetitions" ) ) {
            repetitions = nb::cast<size_t>( kwargs["repetitions"] );
        }
        if ( distance == "cosine" ) {
            std::string hasher = "crosspolytope";
            if ( kwargs.contains( "hasher" ) ) {
                hasher = nb::cast<std::string>( kwargs["hasher"] );
            }
            if ( hasher == "crosspolytope" ) {
                using Builder =
                    panna::CrossPolytopeBuilder<3, panna::UnitNormPoints, panna::CosineDistance>;
                Builder builder( dimensions );
                panna::Index<panna::UnitNormPoints, Builder::Output, panna::CosineDistance> index(
                    dimensions, builder, repetitions );
                inner = std::make_unique<
                    ConcreteIndex<panna::UnitNormPoints, Builder::Output, panna::CosineDistance>>(
                    index );
            } else if ( hasher == "simhash" ) {
                using Builder =
                    panna::SimhashBuilder<24, panna::UnitNormPoints, panna::CosineDistance>;
                Builder builder( dimensions );
                panna::Index<panna::UnitNormPoints, Builder::Output, panna::CosineDistance> index(
                    dimensions, builder, repetitions );
                inner = std::make_unique<
                    ConcreteIndex<panna::UnitNormPoints, Builder::Output, panna::CosineDistance>>(
                    index );
            } else {
                throw nb::value_error( "Unsupported hasher for cosine distance. Use either "
                                       "`crosspolytope` or `simhash`" );
            }
        } else if ( distance == "euclidean" ) {
            using Builder = panna::E2LSHBuilder<8, panna::NormedPoints>;
            Builder builder( 0.0, dimensions );
            panna::Index<panna::NormedPoints, Builder::Output, panna::EuclideanDistance> index(
                dimensions, builder, repetitions );
            inner = std::make_unique<
                ConcreteIndex<panna::NormedPoints, Builder::Output, panna::EuclideanDistance>>(
                index );
        } else {
            throw nb::value_error( "Unsupported distance metric" );
        }
    }

    void rebuild() {
        inner->rebuild();
    }

    void insert( const nb::ndarray<float, nb::c_contig>& vec ) {
        inner->insert( vec );
    }

    nb::ndarray<uint32_t, nb::numpy, nb::ndim<1>>
    search( const nb::ndarray<float, nb::shape<-1>>& vec, unsigned int k, float recall ) {
        return inner->search( vec, k, recall );
    }
};

NB_MODULE( _panna_impl, m ) {
    nb::class_<TrieIndex>( m, "TrieIndex" )
        .def( nb::init<size_t, std::string, nb::kwargs>() )
        .def( "insert", &TrieIndex::insert )
        .def( "rebuild", &TrieIndex::rebuild )
        .def( "search", &TrieIndex::search );
}

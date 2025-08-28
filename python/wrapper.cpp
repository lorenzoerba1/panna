#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <sstream>

#include "dbg.h"
#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/trieindex.hpp"
#include "panna/emst.hpp"

namespace nb = nanobind;

struct AbstractIndex {
    virtual void rebuild() = 0;
    virtual void insert( const nb::ndarray<float, nb::c_contig>& vec ) = 0;
    virtual nb::ndarray<uint32_t, nb::numpy, nb::ndim<1>>
    search( const nb::ndarray<float, nb::shape<-1>>& vec, unsigned int k, float recall ) = 0;
    virtual size_t num_points() const = 0;
    virtual size_t num_repetitions() const = 0;
    virtual std::string describe_family() const = 0;
};

template <typename Dataset, typename Hasher, typename Distance>
struct ConcreteIndex final : AbstractIndex {
    panna::Index<Dataset, Hasher, Distance> inner;

    ConcreteIndex( panna::Index<Dataset, Hasher, Distance> inner ): inner( inner ) {
    }

    size_t num_points() const {
        return inner.num_points();
    }

    size_t num_repetitions() const {
        return inner.num_repetitions();
    }

    std::string describe_family() const {
        return inner.describe_family();
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

struct EMST_exposed {
    using EMST_t = panna::EMST<panna::NormedPoints, panna::E2LSH<12, panna::NormedPoints>, panna::EuclideanDistance>;
    std::unique_ptr<EMST_t> inner;

    // Constructor to be called from Python. It takes a NumPy array and optional keyword arguments.
    EMST_exposed(const nb::ndarray<float, nb::c_contig>& data_in, nb::kwargs kwargs) {

        size_t num_points = data_in.shape(0);
        size_t dimensions = data_in.shape(1);

        // Set default parameters, which can be overridden by kwargs from Python
        size_t repetitions = 200;
        double delta = 0.2;
        float epsilon = 0.2;

        if (kwargs.contains("repetitions")) {
            repetitions = nb::cast<size_t>(kwargs["repetitions"]);
        }
        if (kwargs.contains("delta")) {
            delta = nb::cast<double>(kwargs["delta"]);
        }
        if (kwargs.contains("epsilon")) {
            epsilon = nb::cast<float>(kwargs["epsilon"]);
        }

        // Convert the input NumPy array to the std::vector<std::vector<float>> needed by EMST
        std::vector<std::vector<float>> data_cpp;
        data_cpp.reserve(num_points);
        const float* data_ptr = data_in.data();
        for (size_t i = 0; i < num_points; ++i) {
            const float* row_start = data_ptr + i * dimensions;
            data_cpp.emplace_back(row_start, row_start + dimensions);
        }

        using Hasher = panna::E2LSH<12, panna::NormedPoints>;
        Hasher::Builder builder(0.0, dimensions);

        inner = std::make_unique<EMST_t>(dimensions, repetitions, builder, data_cpp, delta, epsilon);
    }

    nb::tuple find_mst(unsigned int k) {
        // Call the underlying C++ method
        auto result_pair = inner->find_tree_dbscan(k);
        
        // MST edges
        auto& tree_edges_vec = result_pair.first;
        size_t num_edges = tree_edges_vec.size();
        
        // Allocate memory on the heap that nanobind will manage.
        auto* tree_data_ptr = new float[num_edges * 3];

        // Copy the edge data into the new buffer.
        for (size_t i = 0; i < num_edges; ++i) {
            tree_data_ptr[i * 3 + 0] = (float) std::get<0>( tree_edges_vec[i] );
            tree_data_ptr[i * 3 + 1] = (float) std::get<1>( tree_edges_vec[i] );
            tree_data_ptr[i * 3 + 2] = (float) std::get<2>( tree_edges_vec[i] );
        }

        // Create a capsule to give ownership of the pointer to the NumPy array.
        // The lambda function tells nanobind how to free the memory later.
        nb::capsule tree_owner(tree_data_ptr, [](void *p) noexcept { delete[] (float *) p; });
        
        nb::ndarray<float, nb::numpy> tree_array = nb::ndarray<float, nb::numpy>(
            tree_data_ptr,
            {num_edges, 3}, // Shape: (num_edges, 3), HDBSCAN wants 3 floats terminal1, terminal2, distance
            tree_owner
        );

        // Core Distances
        auto& neighbor_results = result_pair.second;
        size_t num_points = neighbor_results.size();

        // Allocate heap memory for the core distances.
        auto* core_data_ptr = new float[num_points];

        for (size_t i = 0; i < num_points; ++i) {
            if (!neighbor_results[i].empty()) {
                // The core distance is the distance to the k-th neighbor (or first in the heap that is returned from the search).
                core_data_ptr[i] = neighbor_results[i].front().first;
            } else {
                core_data_ptr[i] = std::numeric_limits<float>::infinity();
            }
        }

        nb::capsule core_owner(core_data_ptr, [](void *p) noexcept { delete[] (float *) p; });

        nb::ndarray<float, nb::numpy, nb::ndim<1>> core_array(
            core_data_ptr,
            {num_points},
            core_owner
        );

        // Neighbors
        size_t num_neighbors_per_point = k + 1; // Search returns k+1 neighbors (including self)
        auto* neighbors_data_ptr = new uint32_t[num_points * num_neighbors_per_point];

        for (size_t i = 0; i < num_points; ++i) {
            for (size_t j = 0; j < num_neighbors_per_point; ++j) {
                // Ensure we don't read out of bounds if fewer than k neighbors were found
                if (j < neighbor_results[i].size()) {
                    neighbors_data_ptr[i * num_neighbors_per_point + j] = neighbor_results[i][j].second;
                } else {
                    neighbors_data_ptr[i * num_neighbors_per_point + j] = -1;
                }
            }
        }

        nb::capsule neighbors_owner(neighbors_data_ptr, [](void *p) noexcept { delete[] (uint32_t *) p; });

        nb::ndarray<uint32_t, nb::numpy, nb::ndim<2>> neighbors_array(
            neighbors_data_ptr,
            {num_points, num_neighbors_per_point},
            neighbors_owner
        );

        // Return a Python tuple containing the three new NumPy arrays.
        return nb::make_tuple(tree_array, core_array, neighbors_array);
    }
};

NB_MODULE( _panna_impl, m ) {
    m.def( "set_seed",
           &panna::seed_global_rng,
           "Set the seed of the global random number generato used by the panna module." );
    nb::class_<TrieIndex>( m, "TrieIndex" )
        .def( nb::init<size_t, std::string, nb::kwargs>() )
        .def( "insert", &TrieIndex::insert )
        .def( "rebuild", &TrieIndex::rebuild )
        .def( "search", &TrieIndex::search )
        .def_prop_ro( "num_repetitions",
                      []( TrieIndex* self ) { return self->inner->num_repetitions(); } )
        .def_prop_ro( "num_points", []( TrieIndex* self ) { return self->inner->num_points(); } )
        .def( "__str__", []( TrieIndex* self ) {
            std::stringstream sstream;
            sstream << "TrieIndex("
                    << "repetitions=" << self->inner->num_repetitions()
                    << " num_points=" << self->inner->num_points()
                    << " family=" << self->inner->describe_family() << ")";
            return sstream.str();
        } );

    nb::class_<EMST_exposed>( m, "EMST")
        .def(nb::init<const nb::ndarray<float, nb::c_contig>&, nb::kwargs>(),
             "Constructs the EMST index from a NumPy array of data points.")
        // Bind the find_mst method
        .def("find_mst", &EMST_exposed::find_mst, nb::arg("k") = 5,
            "Find the minimum spanning tree (MST) and the k-NNs for each node.");

}

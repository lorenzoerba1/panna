#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <sstream>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/baselines/toc_emst.hpp"
#include "panna/trieindex.hpp"
#include "panna/emst.hpp"
#include "panna/logging.hpp"

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
        size_t repetitions = 500;
        double delta = 0.1;
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

        // Take ownership of the input data
        std::vector<std::vector<float>> data_cpp(num_points, std::vector<float>(dimensions));
        for (size_t i = 0; i < num_points; i++) {
            for (size_t j = 0; j < dimensions; j++) {
                data_cpp[i][j] = data_in.data()[i * dimensions + j];
            }
        }



        using Hasher = panna::E2LSH<12, panna::NormedPoints>;
        Hasher::Builder builder(0.0, dimensions);

        inner = std::make_unique<EMST_t>(dimensions, repetitions, builder, data_cpp, delta, epsilon);
    }

    float find_exact_mst() {
        return inner->find_tree().first;
    }

    float find_epsilon_mst() {
        return inner->find_epsilon_tree();
    }

    // Method to find the MST for the reachability and return results as NumPy arrays
    nb::tuple find_mst_dbscan(unsigned int k) {
        // Call the underlying C++ method
        auto result_pair = inner->find_tree_dbscan(k);

        // 1. MST Edges
        auto& tree_edges_vec = result_pair.first;
        size_t num_edges = tree_edges_vec.size();

        // Create a vector on the heap that will be owned by NumPy
        auto tree_vec_ptr = std::make_unique<std::vector<float>>(num_edges * 3);

        for (size_t i = 0; i < num_edges; ++i) {
            (*tree_vec_ptr)[i * 3 + 0] = std::get<0>(tree_edges_vec[i]);
            (*tree_vec_ptr)[i * 3 + 1] = std::get<1>(tree_edges_vec[i]);
            (*tree_vec_ptr)[i * 3 + 2] = std::get<2>(tree_edges_vec[i]);
        }

        nb::capsule tree_owner(tree_vec_ptr.get(), [](void *p) noexcept {
            delete static_cast<std::vector<float>*>(p);
        });
        
        nb::ndarray<float, nb::numpy> tree_array(
            tree_vec_ptr.release()->data(), 
            {num_edges, 3},
            tree_owner 
        );
        LOG_INFO("msg", "Created tree array", "num_edges", num_edges);

        // 2. Core Distances
        auto& neighbor_results = result_pair.second;
        size_t num_points = neighbor_results.size();

        auto core_vec_ptr = std::make_unique<std::vector<float>>(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            if (!neighbor_results[i].empty()) {
                (*core_vec_ptr)[i] = neighbor_results[i].front().first;
            } else {
                (*core_vec_ptr)[i] = std::numeric_limits<float>::infinity();
            }
        }

        nb::capsule core_owner(core_vec_ptr.get(), [](void *p) noexcept {
            delete static_cast<std::vector<float>*>(p);
        });

        LOG_INFO("msg", "Created core array", "num_points", num_points);

        nb::ndarray<float, nb::numpy, nb::ndim<1>> core_array(
            core_vec_ptr.release()->data(),
            {num_points},
            core_owner
        );


        // 3. Neighbors
        size_t num_neighbors_per_point = k + 1;
        auto neighbors_vec_ptr = std::make_unique<std::vector<uint32_t>>(num_points * num_neighbors_per_point);
        
        for (size_t i = 0; i < num_points; ++i) {
            for (size_t j = 0; j < num_neighbors_per_point; ++j) {
                if (j < neighbor_results[i].size()) {
                    (*neighbors_vec_ptr)[i * num_neighbors_per_point + j] = neighbor_results[i][j].second;
                } else {
                    (*neighbors_vec_ptr)[i * num_neighbors_per_point + j] = i;
                }
            }
        }

        nb::capsule neighbors_owner(neighbors_vec_ptr.get(), [](void *p) noexcept {
            delete static_cast<std::vector<uint32_t>*>(p);
        });

        nb::ndarray<uint32_t, nb::numpy, nb::ndim<2>> neighbors_array(
            neighbors_vec_ptr.release()->data(),
            {num_points, num_neighbors_per_point},
            neighbors_owner
        );
        LOG_INFO("msg", "Created neighbors array", "num_points", num_points, "num_neighbors_per_point", num_neighbors_per_point);
        // Return as a python tuple
        return nb::make_tuple(tree_array, core_array, neighbors_array);
    }
};

nb::tuple emst_theory_of_computing( nb::ndarray<float, nb::c_contig>& data_in, nb::kwargs kwargs ) {
    float delta = 0.1;
    float gamma = 1.0;

    if ( kwargs.contains( "delta" ) ) {
        delta = nb::cast<float>( kwargs["delta"] );
    }
    if ( kwargs.contains( "gamma" ) ) {
        gamma = nb::cast<float>( kwargs["gamma"] );
    }

    size_t nrows = data_in.shape( 0 );
    size_t dimensionality = data_in.shape( 1 );
    panna::NormedPoints dataset( dimensionality );
    float* data = data_in.data();
    for ( size_t row = 0; row < nrows; row++ ) {
        float* begin = data + row * dimensionality;
        float* end = data + ( row + 1 ) * dimensionality;
        expect( end - begin == static_cast<ptrdiff_t>( dimensionality ) );
        dataset.push_back( begin, end );
    }
    LOG_INFO("nrows", nrows, "dimensionality", dimensionality);

    const size_t K = 3; // using a larger value entails using way too many repetitions in the last iterations
    using Distance = panna::EuclideanDistance;
    using Hasher = panna::E2LSH<K, panna::NormedPoints>;
    Hasher::Builder builder( 0.0, dimensionality );

    auto res =
        panna::baselines::emst_theory_of_computing<panna::NormedPoints, Hasher::Builder, Distance>(
            dataset, gamma, delta, builder );
    auto tree = res.second;

    size_t tree_size = tree.size();
    float* weights = new float[tree_size];
    uint32_t* ids = new uint32_t[tree_size * 2];
    for ( size_t i = 0; i < tree_size; i++ ) {
        weights[i] = std::get<0>( tree[i] );
        ids[i * 2] = std::get<1>( tree[i] );
        ids[i * 2 + 1] = std::get<2>( tree[i] );
    }

    nb::capsule weights_owner( weights, []( void* p ) noexcept { delete[] (float*)p; } );
    nb::capsule ids_owner( ids, []( void* p ) noexcept { delete[] (uint32_t*)p; } );

    return nb::make_tuple(
        nb::ndarray<nb::numpy, float, nb::ndim<1>>(weights, {tree_size}, weights_owner),
        nb::ndarray<nb::numpy, uint32_t, nb::ndim<2>>(ids, {tree_size, 2}, ids_owner)
    );
}

NB_MODULE( _panna_impl, m ) {
    m.def( "set_seed",
           &panna::seed_global_rng,
           "Set the seed of the global random number generato used by the panna module." );
    m.def( "emst_theory_of_computing",
           &emst_theory_of_computing ),
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
        .def("find_mst_dbscan", &EMST_exposed::find_mst_dbscan, nb::arg("k") = 5,
            "Find the minimum spanning tree (MST) and the k-NNs for each node.")
            .def("find_exact_mst", &EMST_exposed::find_exact_mst,
                 "Find the exact minimum spanning tree (MST) for the dataset.")
            .def("find_epsilon_mst", &EMST_exposed::find_epsilon_mst,
                 "Find the 1+epsilon approximate minimum spanning tree (MST) for the dataset.");

}

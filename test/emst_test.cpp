#include <highfive/H5Easy.hpp>

#include <chrono>
#include <iostream>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/emst.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/rand.hpp"
#include "panna/logging.hpp"

using namespace panna;

int main() {

    seed_global_rng( std::chrono::high_resolution_clock::now().time_since_epoch().count() );
    // Parameters
    const size_t conc = 12;
    // const size_t dimensions = 20;
    const size_t rep = 500;
    const size_t n = 1000;
    using Point = NormedPoints;         // UnitNormPoints or NormedPoints
    using Distance = EuclideanDistance; // EuclideanDistance or AngularDistance or CosineDistance
    using Hasher = E2LSH<conc, Point>;
    // using Hasher = CrossPolytope<conc, Point, Distance, rotations>;

    // CrossPolytopeBuilder<conc, Point, Distance, rotations> builder( dimensions );

    // std::vector<std::vector<float>> points;
    // for ( size_t i = 0; i < n; i++ ) {
    //     std::vector<float> point = sample_random_normal_vector( 20 );
    //     points.push_back( point );
    // }
    H5Easy::File file( "datasets/fashion-mnist-784-euclidean.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/glove-100-angular.hdf5", H5Easy::File::ReadOnly );
    //   H5Easy::File file( "datasets/nytimes-256-angular.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/simplewiki-openai-3072-normalized.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/gist-960-euclidean.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/deep-image-96-angular.hdf5", H5Easy::File::ReadOnly );

    std::vector<std::vector<float>> points =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );
    points.resize( n );

    size_t dimensions = points[0].size();
    E2LSHBuilder<conc, NormedPoints> builder( dimensions );
    EMST<Point, Hasher, Distance> tree( dimensions, rep, builder, points );

    // Exact computation
    auto start_exact = std::chrono::high_resolution_clock::now();
    float weight_exact = tree.exact_tree();
    auto end_exact = std::chrono::high_resolution_clock::now();
    LOG_INFO("msg", "Computed exact weight",
             "exact_weight", weight_exact,
             "elapsed_s", std::chrono::duration<double>( end_exact - start_exact ).count());

    // Exact with predictions (?)
    auto start = std::chrono::high_resolution_clock::now();
    // for (size_t iter= 0; iter< 3 ; iter++) {
    //     EMST<NormedPoints, Hasher, EuclideanDistance> tree( dimensions, rep, builder, points );

    float weigth = tree.find_tree();
    // }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = ( end - start );
    LOG_INFO("msg", "Computed approximate weight",
             "approximate_weight", weigth,
             "elapsed_s", elapsed.count());

    // std::cout << "----- Epsilon Version -----" << std::endl;
    // // Approximate with predictions
    // (void) tree.find_epsilon_tree();

    return 0;
}

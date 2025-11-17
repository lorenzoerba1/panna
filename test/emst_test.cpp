#include <highfive/H5Easy.hpp>

#include <chrono>

#include "panna/baselines/toc_emst.hpp"
#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/emst.hpp"
#include "panna/logging.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/rand.hpp"

using namespace panna;

int main() {

    seed_global_rng(
        365 ); // std::chrono::high_resolution_clock::now().time_since_epoch().count() );
    // Parameters
    const size_t conc_toc = 3;
    const size_t conc = 12;
    // const size_t dimensions = 20;
    const size_t rep = 500;
    const size_t n = 5000;
    using Point = EuclideanPoints;         // UnitNormPoints or EuclideanPoints
    using Distance = EuclideanDistance; // EuclideanDistance or AngularDistance or CosineDistance
    using Hasher = E2LSH<conc, Point, Distance>;
    // using Hasher = CrossPolytope<conc, Point, Distance, rotations>;

    // CrossPolytopeBuilder<conc, Point, Distance, rotations> builder( dimensions );

    // std::vector<std::vector<float>> points;
    // for ( size_t i = 0; i < n; i++ ) {
    //     std::vector<float> point = sample_random_normal_vector( 400 );
    //     points.push_back( point );
    // }
    H5Easy::File file( "datasets/fashion-mnist-784-euclidean.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/glove-100-angular.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/nytimes-256-angular.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/simplewiki-openai-3072-normalized.hdf5", H5Easy::File::ReadOnly); 
    // H5Easy::File file( "datasets/gist-960-euclidean.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/deep-image-96-angular.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/sift-128-euclidean.hdf5", H5Easy::File::ReadOnly );
    // H5Easy::File file( "datasets/deep-image-96-angular.hdf5", H5Easy::File::ReadOnly );

    std::vector<std::vector<float>> points =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );
    // points.resize( n );

    size_t dimensions = points[0].size();
    E2LSHBuilder<conc, EuclideanPoints, Distance> builder( dimensions );
    EMST<Point, Hasher, Distance> tree( dimensions, rep, builder, points, 0.01, 0.2 );

    // Exact computation
    // auto start_exact = std::chrono::high_resolution_clock::now();
    // float weight_exact = tree.exact_tree();
    // auto end_exact = std::chrono::high_resolution_clock::now();
    // auto elapsed_exact_s = std::chrono::duration<double>( end_exact - start_exact ).count();
    // LOG_INFO( "msg",
    //           "Computed exact weight",
    //           "exact_weight",
    //           weight_exact,
    //           "elapsed_s",
    //           elapsed_exact_s );
    // Exact with predictions
    auto start = std::chrono::high_resolution_clock::now();
    // // for (size_t iter= 0; iter< 3 ; iter++) {
    // //     EMST<EuclideanPoints, Hasher, EuclideanDistance> tree( dimensions, rep, builder, points
    // );

    const auto& [weight, emst_exact] = tree.find_tree();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>( end - start ).count();
    LOG_INFO("msg", "Computed exact with predictions weight",
             "exact_weight", weight,
             //"weight-difference", weight - weight_exact,
             "elapsed_s", elapsed);
    
    tree.find_tree_hist(emst_exact);
    exit(0);

    // // // Approximate with predictions
    // start = std::chrono::high_resolution_clock::now();
    // for (size_t iter= 0; iter< 3 ; iter++) {
    // float approx_weight = tree.find_epsilon_tree();

    // end = std::chrono::high_resolution_clock::now();
    // elapsed = ( end - start );
    // LOG_INFO("msg", "Computed approximate with predictions weight",
    //          "approx_weight", approx_weight,
    //             "elapsed_s", elapsed.count());
    // }

     expect(conc_toc <= 4); // more than 4 concatenations is too much in practice!
     EuclideanPoints dataset( dimensions );
     for ( auto& p : points ) {
         dataset.push_back( p.begin(), p.end() );
     }
     float fp = 0.1;
     float gamma = 0.2;
     E2LSHBuilder<conc_toc, EuclideanPoints, Distance> builder_toc( dimensions );
     auto start_toc = std::chrono::high_resolution_clock::now();
     auto res = panna::baselines::emst_theory_of_computing<EuclideanPoints,
                                                           E2LSHBuilder<conc_toc, EuclideanPoints, Distance>,
                                                           Distance>( dataset, gamma, fp, builder_toc );
     auto end_toc = std::chrono::high_resolution_clock::now();
     auto elapsed_toc_s = std::chrono::duration<double>( end_toc - start_toc ).count();
     float weight_toc = res.first;
     LOG_INFO( "msg",
               "computed toc EMST baseline",
               "toc-emst-weight",
               weight_toc,
               "ratio-toc",
               weight_toc / weight,//_exact,
               "elapsed_s",
               elapsed_toc_s,
               "time_ratio",
               elapsed_toc_s / elapsed );
    return 0;
}

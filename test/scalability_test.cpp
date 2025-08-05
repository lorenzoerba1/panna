#include <iostream>
#include <highfive/H5Easy.hpp>
#include "panna/emst.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/distance.hpp"
#include "panna/data.hpp"
#include "panna/rand.hpp"
#include <chrono>

using namespace panna;

int main () {
    seed_global_rng( std::chrono::high_resolution_clock::now().time_since_epoch().count() );
    const size_t conc = 12;
    const uint8_t rotations = 3;
    const std::vector<size_t> dimensions = {50, 100, 200};//,  400, 800, 1600};
    const size_t rep = 500;
    const size_t n = 10000;
    using Point = NormedPoints; // UnitNormPoints or NormedPoints
    using Distance = EuclideanDistance; // EuclideanDistance or AngularDistance or CosineDistance
    using Hasher = E2LSH<conc, Point>;
    //using Hasher = CrossPolytope<conc, Point, Distance, rotations>;
    std::ofstream outfile("results/weight_results.csv", std::ios_base::app);

    for (const auto& dimension : dimensions){
        E2LSHBuilder<conc, NormedPoints> builder ( dimension );
        //CrossPolytopeBuilder<conc, Point, Distance, rotations> builder( dimensions );

        std::vector<std::vector<float>> points;
        for ( size_t i = 0; i < n; i++ ) {
            std::vector<float> point = sample_random_normal_vector( dimension );
            points.push_back( point );
        }

        // Exact computation
        EMST<Point, Hasher, Distance> tree( dimension, rep, builder, points );
        // auto start = std::chrono::high_resolution_clock::now();
        // float weight = tree.exact_tree();
        // auto end = std::chrono::high_resolution_clock::now();
        // outfile << "Exact, "<< n << "," << dimension << "," << weight << "," << std::chrono::duration<double>(end - start).count() << std::endl;

        for ( size_t repet= 0; repet <3 ; repet++) {
            // // Exact with Kruskal+
            // start = std::chrono::high_resolution_clock::now();
            // weight = tree.find_tree();
            // end = std::chrono::high_resolution_clock::now();
            // outfile << "K+, "<< n << "," << dimension << "," << weight << "," << std::chrono::duration<double>(end - start).count() << std::endl;

            // Approximate with Kruskal+-
            auto start = std::chrono::high_resolution_clock::now();
            float weight = tree.find_epsilon_tree();
            auto end = std::chrono::high_resolution_clock::now();
            outfile << "K± e0.2, "<< n << "," << dimension << "," << weight << "," << std::chrono::duration<double>(end - start).count() << std::endl;
        }
    }

    outfile.close();


    return 0;
}
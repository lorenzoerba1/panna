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
#include <string>

using namespace panna;

int main()  {
    seed_global_rng( 1989 );
    std::string datasets[4] = {
        "datasets/fashion-mnist-784-euclidean.hdf5",
        "datasets/glove-100-angular.hdf5",
        "datasets/nytimes-256-angular.hdf5",
        "datasets/gist-960-euclidean.hdf5"
    };
    int index = 0;
    for (const auto& name: datasets) {
        std::cout << "Processing dataset: " << name << std::endl;

        // Parameters
        const size_t conc = 12;
        const uint8_t rotations = 3;
        const size_t dimensions[4] = { 784, 100, 256, 960 };
        float eps[6] = { 0.2, 0.5, 1.0, 5.0, 10.0, 20.0 };
        float probs[2] = {0.1, 0.2};
        std::vector<float> weigths; 
        using Point = NormedPoints; // UnitNormPoints or NormedPoints
        using Distance = EuclideanDistanceNoSqrt; // EuclideanDistance or AngularDistance or CosineDistance
        using Hasher = E2LSH<conc, Point>;
        // using Hasher = CrossPolytope<conc, Point, Distance, rotations>;
        std::ofstream outfile("weight_results.csv", std::ios_base::app);

        for (const auto& prob: probs) {
            for (const auto& ep: eps) {
                for (size_t i = 0; i < 3; i++) {
                    E2LSHBuilder<conc, NormedPoints> builder(dimensions[index]);
                    // CrossPolytopeBuilder<conc, Point, Distance, rotations> builder( dimensions[index] );

                    H5Easy::File file(name, H5Easy::File::ReadOnly);
                    std::vector<std::vector<float>> points =
                        H5Easy::load<std::vector<std::vector<float>>>(file, "/train");
                    if (points.size() > 10000)
                        points.resize(10000); // Limit to 1000 points for testing

                    EMST<Point, Hasher, Distance> tree(dimensions[index], 500, builder, points, prob, ep);
                    float weight;
                    double duration;

                    auto time = std::chrono::high_resolution_clock::now();
                    weight = tree.find_tree();
                    duration = std::chrono::duration<double>( std::chrono::high_resolution_clock::now() - time ).count();
                    outfile <<"K+" << ", " << points.size() << ", " << name << ", " << weight << ", "<< duration << ", " << prob << std::endl;
                    


                    time = std::chrono::high_resolution_clock::now();
                    weight = tree.find_epsilon_tree();
                    duration = std::chrono::duration<double>( std::chrono::high_resolution_clock::now() - time ).count();
                    outfile <<"K± e" << ep << ", " << points.size() << ", " << name << ", " << weight << ", "<< duration << ", " << prob << std::endl;
                    // Push the writes to file
                    outfile.flush();
                }
            }
        }
        index++;
        outfile.close();
    }
}

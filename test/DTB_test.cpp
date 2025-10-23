#include <iostream>
#include <highfive/H5Easy.hpp>
//#include <mlpack/mlpack/mlpack.hpp>
#include <chrono>
#include <string>
#include <fstream>
#include <vector>

int main() {

    //arma::mat data;
    // Load the dataset as an Armadillo matrix
    H5Easy::File file( "nytimes-256-angular.hdf5", H5Easy::File::ReadOnly );
    std::vector<std::vector<float>> data = H5Easy::load<std::vector<std::vector<float>>>(file, "/train");

    // Write erach point as a line in a csv file
    std::ofstream edgeFile("txts/nytimes_edges.csv");
    if (!edgeFile.is_open()) {
        std::cerr << "Error opening edges.csv for writing." << std::endl;
        return 1;
    }
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            edgeFile << data[i][j];
            if (j < data[i].size() - 1) {
                edgeFile << ",";
            }
        }
        edgeFile << "\n";
    }

    // // Initialize the DualTreeBoruvka object
    // mlpack::DualTreeBoruvka<> dtb;

    // arma::mat mstResults;
    // dtb.ComputeMST(mstResults);

    // // Print the weight of the MST
    // double totalWeight = arma::accu(mstResults.row(2)); // The third row contains the weights
    // std::cout << "Total weight of the MST: " << totalWeight << std::endl;

    return 0;
}
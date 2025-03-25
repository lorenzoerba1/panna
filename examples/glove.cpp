#include <highfive/H5Easy.hpp>

#include "dbg.h"
#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/trieindex.hpp"

float compute_recall(std::vector<std::pair<float, uint32_t>> &ground, std::vector<std::pair<float, uint32_t>> &actual) {
    size_t k = actual.size();
    float thresh = ground[k-1].first;

    float cnt = 0.0;
    for (auto pair : actual) {
        if (pair.first <= thresh) {
            cnt += 1.0;
        }
    }

    return cnt / k;
}

int main( int argc, char* argv[] ) {
    using Distance = panna::CosineDistance;
    using Dataset = panna::UnitNormPoints;
    // using HasherBuilder = panna::SimhashBuilder<24, Dataset, Distance>;
    using HasherBuilder = panna::CrossPolytopeBuilder<4, Dataset, Distance>;
    using Hasher = HasherBuilder::Output;

    H5Easy::File file( "glove-100-angular.hdf5", H5Easy::File::ReadOnly );

    std::vector<std::vector<float>> data =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );

    std::vector<std::vector<float>> queries =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/test" );

    size_t dimensions = data[0].size();
    HasherBuilder hbuilder( dimensions );

    panna::Index<Dataset, Hasher, Distance> index( dimensions, hbuilder, 256 );
    size_t cnt = 0;
    for ( auto v : data ) {
        index.insert( v );
        if (cnt++ > 1000) {
            break;
        }
    }
    index.rebuild();

    size_t k = 1;
    float delta = 0.01;
    size_t q_cnt = 0;
    std::vector<std::pair<float, uint32_t>> res;
    std::vector<std::pair<float, uint32_t>> res_prob;

    auto q = queries[0];
    index.search_brute_force( q, k, res );
    dbg(res.back().first);
    index.search( q, k, delta, res_prob );
    float recall = compute_recall(res, res_prob);
    dbg(recall);
}

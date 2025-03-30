#include <highfive/H5Easy.hpp>
#include <chrono>

#include "dbg.h"
#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/lsh/simhash.hpp"
#include "panna/trieindex.hpp"

float compute_recall( std::vector<std::pair<float, uint32_t>>& ground,
                      std::vector<std::pair<float, uint32_t>>& actual ) {
    size_t k = actual.size();
    float thresh = ground[k - 1].first;

    float cnt = 0.0;
    for ( auto pair : actual ) {
        if ( pair.first <= thresh ) {
            cnt += 1.0;
        }
    }

    return cnt / k;
}

int main( int argc, char* argv[] ) {
    using Distance = panna::CosineDistance;
    using Dataset = panna::UnitNormPoints;
    // using HasherBuilder = panna::SimhashBuilder<24, Dataset, Distance>;
    using HasherBuilder = panna::CrossPolytopeBuilder<3, Dataset, Distance>;
    using Hasher = HasherBuilder::Output;

    H5Easy::File file( "glove-100-angular.hdf5", H5Easy::File::ReadOnly );

    std::vector<std::vector<float>> data =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );

    std::vector<std::vector<float>> queries =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/test" );
    std::cout << "data loaded" << std::endl;

    size_t dimensions = data[0].size();
    HasherBuilder hbuilder( dimensions );
    
    panna::Index<Dataset, Hasher, Distance> index( dimensions, hbuilder, 380);
    size_t cnt = 0;
    for ( auto v : data ) {
        if ( cnt > 10000 ) {
            break;
        }
        index.insert( v );
        cnt++;
    }
    index.rebuild();
    std::cout << "index built" << std::endl;

    size_t k = 1;
    float delta = 0.1;
    size_t q_cnt = 0;
    std::vector<std::pair<float, uint32_t>> res;
    std::vector<std::pair<float, uint32_t>> res_prob;

    auto start = std::chrono::steady_clock::now();
    for ( auto q : queries ) {
        if ( q_cnt > 1000 ) {
            break;
        }
        // index.search_brute_force( q, k, res );
        // dbg( res.back().first );
        index.search( q, k, delta, res_prob );
        // float recall = compute_recall( res, res_prob );
        // dbg( recall );
        q_cnt++;
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);
    double qps = q_cnt / elapsed_s;
    std::cout << "qps: " << qps << std::endl;
}

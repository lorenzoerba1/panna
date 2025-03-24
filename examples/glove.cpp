#include <highfive/H5Easy.hpp>

#include "dbg.h"
#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/crosspolytope.hpp"
#include "panna/trieindex.hpp"

int main( int argc, char* argv[] ) {
    using Distance = panna::CosineDistance;
    using Dataset = panna::UnitNormPoints;
    using Hasher = panna::CrossPolytope<4, Dataset, Distance>;
    using HasherBuilder = panna::CrossPolytopeBuilder<4, Dataset, Distance>;

    H5Easy::File file( "glove-100-angular.hdf5", H5Easy::File::ReadOnly );

    std::vector<std::vector<float>> data =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );

    std::vector<std::vector<float>> queries =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/test" );

    size_t dimensions = data[0].size();
    HasherBuilder hbuilder( dimensions );

    panna::Index<Dataset, Hasher, Distance> index( dimensions, hbuilder, 64 );
    for ( auto v : data ) {
        index.insert( v );
    }

    size_t q_cnt = 0;
    std::vector<std::pair<float, size_t>> res;
    for ( auto q : queries ) {
        index.search_brute_force( q, 10, res );
        dbg( res );
        if ( q_cnt++ > 10 ) {
            break;
        }
    }
}

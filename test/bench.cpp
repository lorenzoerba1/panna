#include "panna/data.hpp"
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#include "panna/rand.hpp"
#include "panna/linalg.hpp"

int main() {
    using namespace panna;
    double dims = 980;
    std::vector<float> a = sample_random_normal_vector(dims);
    std::vector<float> b = sample_random_normal_vector(dims);
    NormedPoints pts(dims);
    pts.push_back(a.begin(), a.end());
    pts.push_back(b.begin(), b.end());

    ankerl::nanobench::Bench().run("euclidean naive", [&] {
        float d = euclidean_naive( a.data(), b.data(), dims );
        ankerl::nanobench::doNotOptimizeAway(d);
    });
    ankerl::nanobench::Bench().run("euclidean avx2", [&] {
        float d = euclidean_avx2( a.data(), b.data(), dims );
        ankerl::nanobench::doNotOptimizeAway(d);
    });
    ankerl::nanobench::Bench().run("euclidean normed points", [&] {
        float d = euclidean( pts[0], pts[1] );
        ankerl::nanobench::doNotOptimizeAway(d);
    });
}

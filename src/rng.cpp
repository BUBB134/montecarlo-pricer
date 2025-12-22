#include "rng.hpp"
#include <cmath>

namespace montecarlo{

    RNG::RNG(uint64_t seed)
    : engine_(seed),
        uniform_dist_(0.0, std::nextafter(1.0, 2.0)),
        normal_dist_(0.0, 1.0)
        {}

    void RNG::seed(uint64_t seed){
        engine_.seed(seed);
    }

    double RNG::uniform(){
        return uniform_dist_(engine_);
    }

    double RNG::normal(){
        return normal_dist_(engine_);
    }

    std::vector<double> RNG::normal_vector(std::size_t n){
        std::vector<double> out;
        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(normal_dist_(engine_));
        }
        return out;
}
}
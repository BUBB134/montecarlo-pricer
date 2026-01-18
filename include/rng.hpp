#ifndef RNG_HPP
#define RNG_HPP

#include <random>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace montecarlo
{
    class RNG
    {
    public:
        explicit RNG(uint64_t seed = std::random_device{}());
        void seed(uint64_t seed);

        double uniform();

        double normal();

        std::vector<double> normal_vector(std::size_t n);

        // Batch generation for SIMD-friendly processing
        void normal_batch(double* out, std::size_t n);

    private:
        std::mt19937_64 engine_;
        std::uniform_real_distribution<double> uniform_dist_;
        std::normal_distribution<double> normal_dist_;
    };
}

#endif // RNG_HPP
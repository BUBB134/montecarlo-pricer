#ifndef MONTECARLO_PRICER_MAIN_HPP
#define MONTECARLO_PRICER_MAIN_HPP

#include <cstddef>
#include <cstdint>

namespace montecarlo
{
    void run_demo(std::size_t n_paths = 100'000, uint64_t seed = 0);
    void run_benchmark(std::size_t n_paths, std::size_t max_threads = 0);
}

#endif // MONTECARLO_PRICER_MAIN_HPP
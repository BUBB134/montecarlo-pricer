#ifndef MONTECARLO_PRICER_TIMER_HPP
#define MONTECARLO_PRICER_TIMER_HPP

#include <chrono>
#include <string>

namespace montecarlo
{
    class Timer
    {
    public:
        Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}

        void reset()
        {
            start_time_ = std::chrono::high_resolution_clock::now();
        }

        double elapsed_seconds() const
        {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double>(end_time - start_time_).count();
        }

        double elapsed_milliseconds() const
        {
            return elapsed_seconds() * 1000.0;
        }

        double elapsed_nanoseconds() const
        {
            return elapsed_seconds() * 1e9;
        }

    private:
        std::chrono::high_resolution_clock::time_point start_time_;
    };

    struct BenchmarkResult
    {
        double time_seconds{0.0};
        double paths_per_second{0.0};
        double nanoseconds_per_path{0.0};
        std::size_t num_paths{0};
        std::size_t num_threads{1};

        void compute_metrics()
        {
            if (time_seconds > 0.0 && num_paths > 0)
            {
                paths_per_second = static_cast<double>(num_paths) / time_seconds;
                nanoseconds_per_path = (time_seconds * 1e9) / static_cast<double>(num_paths);
            }
        }
    };
}

#endif // MONTECARLO_PRICER_TIMER_HPP

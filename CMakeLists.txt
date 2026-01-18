cmake_minimum_required(VERSION 3.15)
project(montecarlo_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable SIMD optimizations and auto-vectorization
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

find_package(Threads REQUIRED)

add_subdirectory(src)
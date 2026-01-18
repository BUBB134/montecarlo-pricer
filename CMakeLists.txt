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

# Check if building Python bindings
if(SKBUILD)
    # Building with scikit-build-core for Python
    find_package(pybind11 CONFIG REQUIRED)
    
    # Create a library with the core functionality
    add_library(montecarlo_core STATIC
        src/rng.cpp
        src/payoff.cpp
        src/monte_carlo.cpp
        src/bs_analytical.cpp
        src/path_simulator.cpp
    )
    
    target_include_directories(montecarlo_core PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )
    
    target_link_libraries(montecarlo_core PUBLIC Threads::Threads)
    
    # Python module
    pybind11_add_module(montecarlo_pricer bindings/bindings.cpp)
    target_link_libraries(montecarlo_pricer PRIVATE montecarlo_core)
    
    # Installation
    install(TARGETS montecarlo_pricer LIBRARY DESTINATION .)
else()
    # Building standalone C++ executable
    add_subdirectory(src)
endif()
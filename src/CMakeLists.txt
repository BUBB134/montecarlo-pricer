add_executable(montecarlo_pricer
    main.cpp
    rng.cpp
    payoff.cpp
    monte_carlo.cpp
    bs_analytical.cpp
    path_simulator.cpp
)

target_include_directories(montecarlo_pricer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
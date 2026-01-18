add_executable(mc_pricer_cli
    main.cpp
    rng.cpp
    payoff.cpp
    monte_carlo.cpp
    bs_analytical.cpp
    path_simulator.cpp
)

target_include_directories(mc_pricer_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(mc_pricer_cli PRIVATE Threads::Threads)
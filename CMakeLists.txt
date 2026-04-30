cmake_minimum_required(VERSION 3.16)
project(tailslayer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_library(tailslayer INTERFACE)
target_include_directories(tailslayer INTERFACE include)

add_executable(tailslayer_example tailslayer_example.cpp)
target_link_libraries(tailslayer_example PRIVATE tailslayer Threads::Threads)

option(TAILSLAYER_BUILD_TRANSFORMER_BENCH "Build the transformer latency experiment" ON)
if(TAILSLAYER_BUILD_TRANSFORMER_BENCH)
    add_subdirectory(experiments/transformer_tailslayer)
endif()

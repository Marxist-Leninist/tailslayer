cmake_minimum_required(VERSION 3.16)
project(tailslayer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tailslayer INTERFACE)
target_include_directories(tailslayer INTERFACE include)

if(NOT WIN32)
    find_package(Threads REQUIRED)
endif()

# Original example
add_executable(tailslayer_example tailslayer_example.cpp)
target_link_libraries(tailslayer_example PRIVATE tailslayer)
if(NOT WIN32)
    target_link_libraries(tailslayer_example PRIVATE Threads::Threads)
endif()

# Transformer benchmark
add_executable(transformer_bench transformer_bench.cpp)
target_link_libraries(transformer_bench PRIVATE tailslayer)
if(NOT WIN32)
    target_link_libraries(transformer_bench PRIVATE Threads::Threads)
endif()

if(MSVC)
    target_compile_options(tailslayer_example PRIVATE /W3 /O2)
    target_compile_options(transformer_bench  PRIVATE /W3 /O2)
else()
    target_compile_options(tailslayer_example PRIVATE -O3 -Wall)
    target_compile_options(transformer_bench  PRIVATE -O3 -Wall)
    if(NOT WIN32)
        target_compile_definitions(tailslayer_example PRIVATE _GNU_SOURCE)
        target_compile_definitions(transformer_bench  PRIVATE _GNU_SOURCE)
    endif()
endif()

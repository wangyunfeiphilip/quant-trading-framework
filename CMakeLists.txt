cmake_minimum_required(VERSION 3.16)
project(quant_trading_framework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 QUIET)

add_library(cpp_engine STATIC
    cpp_engine/order.cpp
    cpp_engine/position.cpp
    cpp_engine/portfolio.cpp
    cpp_engine/backtester.cpp
)

target_include_directories(cpp_engine PUBLIC cpp_engine)

if(Eigen3_FOUND)
    target_link_libraries(cpp_engine PUBLIC Eigen3::Eigen)
    target_compile_definitions(cpp_engine PUBLIC QUANT_HAS_EIGEN=1)
endif()

add_executable(cpp_engine_demo cpp_engine/main.cpp)
target_link_libraries(cpp_engine_demo PRIVATE cpp_engine)

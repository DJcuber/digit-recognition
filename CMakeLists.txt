cmake_minimum_required(VERSION 3.15...4.1)

project(
  DigitRecognition
  VERSION 1.0.0
  DESCRIPTION "A neural network trained to recognise digits"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(TEST_NAME "Test")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE source "src/*.cpp" "include/*.h")
add_executable(${PROJECT_NAME} ${source})
target_compile_options(${PROJECT_NAME} PUBLIC -O2 -Wall -Wextra -std=c++20)
target_include_directories(${PROJECT_NAME} PUBLIC include)

file(GLOB_RECURSE test_source "src/matrix.cpp" "src/network.cpp" "test/*.cpp" "include/*.h")
add_executable(${TEST_NAME} ${test_source})
target_compile_options(${TEST_NAME} PUBLIC -O2 -Wall -Wextra)
target_include_directories(${TEST_NAME} PUBLIC include)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/mnist/mnist_test.csv 
  ${CMAKE_CURRENT_SOURCE_DIR}/mnist/mnist_train.csv
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/mnist
)

file(
  COPY ${CMAKE_CURRENT_SOURCE_DIR}/network.csv
  ${CMAKE_CURRENT_SOURCE_DIR}/test_network.csv
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

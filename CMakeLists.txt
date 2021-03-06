cmake_minimum_required(VERSION 2.8)

project(facedetect)
set(EXECUTABLE_NAME "facedetect")

#cuda
find_package(CUDA)

# Define sources and executable
file(GLOB_RECURSE sources_files src/*)
# file(GLOB CUDA_FILES "src/" *.cu)
list( APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets -std=c++11")
cuda_add_library(kernel_obj src/image/integral-kernel.cu src/trainer/weak-train-kernel.cu)
target_link_libraries(kernel_obj)
#CUDA_COMPILE(CU_O ${CUDA_FILES})


# SETUP FOR CPP FILES
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++14 -pedantic -fPIC")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
#cuda_add_executable(${EXECUTABLE_NAME} ${sources_files} ${CU_O})
#cuda_add_executable(GPU ${INTEGRAL_O} ${EXECUTABLE_NAME})
cuda_add_executable(${EXECUTABLE_NAME} ${sources_files})
target_link_libraries(${EXECUTABLE_NAME} kernel_obj ${CUDA_LIBRARIES})
#target_link_libraries(${EXECUTABLE_NAME} kernel_obj)
#add_executable(${EXECUTABLE_NAME} 

# Detect and add SFML
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules" ${CMAKE_MODULE_PATH})
#Find any version 2.X of SFML
#See the FindSFML.cmake file for additional details and instructions
find_package(SFML 2 REQUIRED system window graphics)

# Adding Boost
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

if(SFML_FOUND)
  include_directories(${SFML_INCLUDE_DIR})
  target_link_libraries(${EXECUTABLE_NAME} ${SFML_LIBRARIES} ${Boost_LIBRARIES})
endif()

# Install target
install(TARGETS ${EXECUTABLE_NAME} DESTINATION bin)
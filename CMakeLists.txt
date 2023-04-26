cmake_minimum_required(VERSION 3.20)
project(rpa_sa)

set(CMAKE_CXX_STANDARD 17)

option(WITH_CUDA "Enable CUDA support" ON)
option(USE_PYTHON "Link to Python when building" ON)

if(WITH_CUDA)
    enable_language(CUDA)
    add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
    add_definitions(-DWITH_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if (USE_PYTHON)
    add_definitions(-DUSE_PYTHON)
    find_package(Python COMPONENTS Development)
    include_directories(${Python_INCLUDE_DIRS})
endif ()

#set(CMAKE_GENERATOR_PLATFORM x64)
set(Torch_DIR D:/code/libs/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

include_directories(rpa_sa/csrc)

file(GLOB_RECURSE ALL_SOURCES ${ALL_SOURCES} rpa_sa/csrc/*.*)

add_library(rpa_sa SHARED ${ALL_SOURCES})

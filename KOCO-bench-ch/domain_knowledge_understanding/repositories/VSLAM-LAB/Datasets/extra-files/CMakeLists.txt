cmake_minimum_required(VERSION 3.18)
project(PhotometricDatasetCalibration)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
set (CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED )
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(libzip CONFIG REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/FOVUndistorter.cpp 
src/PhotometricUndistorter.cpp
)

#target_include_directories(${PROJECT_NAME} PUBLIC
#        ${Pangolin_INCLUDE_DIRS}
#)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBRARIES}
Eigen3::Eigen
libzip::zip
)

# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(playbackDataset
        src/main_playbackDataset.cpp)
target_link_libraries(playbackDataset ${PROJECT_NAME})


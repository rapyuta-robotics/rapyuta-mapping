cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
set(ROS_BUILD_TYPE Debug)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)


if(NOT ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "armv7l")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -msse3 -ftree-vectorizer-verbose=0")
else(NOT ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "armv7l")
	include_directories(/opt/tbb41_20130116oss/include/)
	SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -march=armv7-a -mcpu=cortex-a9 -mfpu=neon -funsafe-math-optimizations -mvectorize-with-neon-quad -ftree-vectorize -ftree-vectorizer-verbose=5" )
endif(NOT ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "armv7l")


find_package(VTK REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})

rosbuild_add_library(${PROJECT_NAME} src/frame.cpp src/keyframe.cpp)
target_link_libraries(${PROJECT_NAME} tbb)

rosbuild_add_executable(localization src/main.cpp)
target_link_libraries(localization ${PROJECT_NAME})

rosbuild_add_executable(test_vo src/test_vo.cpp)
target_link_libraries(test_vo ${PROJECT_NAME} ${VTK_LIBRARIES})



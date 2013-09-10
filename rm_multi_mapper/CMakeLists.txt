cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

include_directories(/usr/include/suitesparse)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)

find_package(VTK REQUIRED)
include_directories(${VTK_INCLUDE_DIRS})


rosbuild_add_library(${PROJECT_NAME} src/color_keyframe.cpp src/keyframe_map.cpp src/reduce_jacobian_rgb.cpp 
src/reduce_jacobian_slam_3d.cpp src/robot_mapper.cpp)
target_link_libraries(${PROJECT_NAME} tbb rm_localization mysqlcppconn)

rosbuild_add_executable(test_keyframe_extraction src/test_keyframe_extraction.cpp)
target_link_libraries(test_keyframe_extraction ${PROJECT_NAME})


#rosbuild_add_library(${PROJECT_NAME} src/robot_mapper.cpp src/icp_map.cpp src/color_keyframe.cpp
# src/reduce_jacobian_icp.cpp src/reduce_jacobian_rgb.cpp src/reduce_jacobian_icp_p2p.cpp src/reduce_jacobian_rgb_3d.cpp
# src/panorama_map.cpp)
#target_link_libraries(${PROJECT_NAME} tbb rm_localization_util)

rosbuild_add_executable(multi_mapper src/main.cpp)
target_link_libraries(multi_mapper ${PROJECT_NAME})

#rosbuild_add_executable(ransac_test src/test.cpp src/robot_mapper.cpp src/keypoint_map.cpp)
#target_link_libraries(ransac_test cholmod vtkCommon vtkFiltering rm_localization_util)

#rosbuild_add_executable(test2 src/test2.cpp)

#rosbuild_add_executable(ba_demo src/ba_demo.cpp)
#target_link_libraries(ba_demo cholmod)

#rosbuild_add_executable(test_lkt src/test_lkt.cpp)

rosbuild_add_executable(test_icp src/test_icp.cpp)
target_link_libraries(test_icp ${PROJECT_NAME})

rosbuild_add_executable(test_merge src/test_merge.cpp)
target_link_libraries(test_merge ${VTK_LIBRARIES} ${PROJECT_NAME})

rosbuild_add_executable(test_show src/test_show.cpp)
target_link_libraries(test_show ${VTK_LIBRARIES} ${PROJECT_NAME})

#rosbuild_add_executable(test_rgbd src/test_rgbd.cpp)

rosbuild_add_executable(test_panorama src/test_panorama.cpp)
target_link_libraries(test_panorama ${PROJECT_NAME})

#rosbuild_add_executable(test_opt src/test_opt.cpp)
#target_link_libraries(test_opt)

rosbuild_add_executable(test_mysql src/test_mysql.cpp)
target_link_libraries(test_mysql mysqlcppconn)

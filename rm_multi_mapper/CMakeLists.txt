cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)
rosbuild_find_ros_package(actionlib_msgs)
include(${actionlib_msgs_PACKAGE_PATH}/cmake/actionbuild.cmake)
genaction()
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

############################## Library ###########################

rosbuild_add_library(${PROJECT_NAME} src/color_keyframe.cpp src/keyframe_map.cpp
src/reduce_jacobian_rgb.cpp src/reduce_jacobian_slam.cpp 
src/reduce_jacobian_slam_3d.cpp src/reduce_jacobian_ros.cpp 
src/reduce_measurement_g2o.cpp 
src/reduce_measurement_g2o_dist.cpp  
src/robot_mapper.cpp)
target_link_libraries(${PROJECT_NAME} tbb rm_localization mysqlcppconn g2o_types_slam3d g2o_solver_cholmod cholmod)

############################## Local ###########################

rosbuild_add_executable(merge_maps src/merge_maps.cpp)
target_link_libraries(merge_maps ${PROJECT_NAME})

rosbuild_add_executable(optimize_map src/optimize_map.cpp)
target_link_libraries(optimize_map ${PROJECT_NAME})

rosbuild_add_executable(show_map src/show_map.cpp)
target_link_libraries(show_map ${VTK_LIBRARIES} ${PROJECT_NAME})

rosbuild_add_executable(optimize_panorama_map src/optimize_panorama_map.cpp)
target_link_libraries(optimize_panorama_map ${PROJECT_NAME})

rosbuild_add_executable(rotation_demo src/rotation_demo.cpp)
target_link_libraries(rotation_demo ${PROJECT_NAME})

rosbuild_add_executable(teleop_demo src/teleop_demo.cpp)
target_link_libraries(teleop_demo ${PROJECT_NAME})

############################## Parallel ###########################

#rosbuild_add_executable(worker src/worker.cpp)
#target_link_libraries(worker ${PROJECT_NAME} mysqlcppconn)

#rosbuild_add_executable(slam src/slam.cpp)
#target_link_libraries(slam ${PROJECT_NAME} mysqlcppconn)

#rosbuild_add_executable(worker_slam src/worker_slam.cpp)
#target_link_libraries(worker_slam ${PROJECT_NAME} mysqlcppconn)

#rosbuild_add_executable(panorama src/panorama.cpp)
#target_link_libraries(panorama ${PROJECT_NAME} mysqlcppconn)

#rosbuild_add_executable(test_util src/test_util.cpp)
#target_link_libraries(test_util ${PROJECT_NAME} mysqlcppconn)

#rosbuild_add_executable(binary2csv src/binary2csv.cpp)
#target_link_libraries(binary2csv ${PROJECT_NAME})

#rosbuild_add_executable(worker_g2o src/worker_g2o.cpp)
#target_link_libraries(worker_g2o ${PROJECT_NAME} mysqlcppconn)

#rosbuild_add_executable(master_g2o src/master_g2o.cpp)
#target_link_libraries(master_g2o ${PROJECT_NAME} mysqlcppconn g2o_types_slam3d g2o_solver_cholmod cholmod)


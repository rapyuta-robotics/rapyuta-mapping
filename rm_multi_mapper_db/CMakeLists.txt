cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_find_ros_package(actionlib_msgs)
include(${actionlib_msgs_PACKAGE_PATH}/cmake/actionbuild.cmake)
genaction()


rosbuild_init()

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
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

################ Library ##########################################

rosbuild_add_library(${PROJECT_NAME} src/util.cpp)
target_link_libraries(${PROJECT_NAME} tbb rm_localization rm_multi_mapper mysqlcppconn g2o_types_slam3d g2o_solver_cholmod cholmod)


################ Helper functions to work with database ###########

rosbuild_add_executable(load_map_db src/load_map_db.cpp)
target_link_libraries(load_map_db ${PROJECT_NAME} mysqlcppconn)

rosbuild_add_executable(show_map_db src/show_map_db.cpp)
target_link_libraries(show_map_db ${PROJECT_NAME} mysqlcppconn)

################# Map optimization #################################

rosbuild_add_executable(worker_g2o src/worker_g2o.cpp)
target_link_libraries(worker_g2o ${PROJECT_NAME} mysqlcppconn)

rosbuild_add_executable(master_g2o src/master_g2o.cpp)
target_link_libraries(master_g2o ${PROJECT_NAME} mysqlcppconn g2o_stuff g2o_core g2o_types_slam3d g2o_solver_cholmod cholmod)

################# Tests #################################

rosbuild_add_gtest(util_test test/util_test.cpp)
target_link_libraries(util_test ${PROJECT_NAME} mysqlcppconn)



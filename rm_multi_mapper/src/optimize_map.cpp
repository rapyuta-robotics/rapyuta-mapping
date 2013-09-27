/*
 * test_diem.cpp
 *
 *  Created on: Jul 29, 2013
 *      Author: vsu
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>

#include <keyframe_map.h>

int main(int argc, char **argv) {

	ros::init(argc, argv, "optimize_map");
	ros::NodeHandle nh;

	ros::Publisher pointcloud_pub = nh.advertise<
			pcl::PointCloud<pcl::PointXYZRGB> >("pointcloud", 1);

	keyframe_map map;
	map.load(argv[1]);

	std::cerr << map.frames.size() << std::endl;
	map.optimize_g2o();

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map.get_map_pointcloud();

	cloud->header.frame_id = "world";
	cloud->header.stamp = ros::Time::now();
	cloud->header.seq = 0;
	pointcloud_pub.publish(cloud);

	map.save(std::string(argv[1]) + "_optimized");

	return 0;

}

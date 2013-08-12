/*
 * test_diem.cpp
 *
 *  Created on: Jul 29, 2013
 *      Author: vsu
 */

#define MALLOC_CHECK_ 3

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/stitcher.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>

#include <icp_map.h>

int main(int argc, char **argv) {

	ros::init(argc, argv, "test_icp");
	ros::NodeHandle nh;

	ros::Publisher pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
			"/cloudbot0/keypoints", 1);

	icp_map map;
	map.load("icp_map_good3");

	std::cerr << map.frames.size() << std::endl;
	for (int level = 2; level >= 0; level--) {
		for (int i = 0; i < (level + 1) * (level + 1) * 12; i++) {
			map.optimize_rgb_3d(level);
			//map.optimize_rgb_3d_with_intrinsics(level);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
					map.get_map_pointcloud();
			cloud->header.frame_id = "/cloudbot0/map";
			cloud->header.stamp = ros::Time::now();
			cloud->header.seq = 0;
			pub.publish(cloud);
		}
	}

	{
		map.align_z_axis();
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map.get_map_pointcloud();
		cloud->header.frame_id = "/cloudbot0/map";
		cloud->header.stamp = ros::Time::now();
		cloud->header.seq = 0;
		pub.publish(cloud);
	}

	map.save("icp_map_good3_optimized");

	return 0;

}

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

	ros::init(argc, argv, "multi_mapper");
	ros::NodeHandle nh;

	ros::Publisher pointcloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
			"pointcloud", 1);

	keyframe_map map;
	map.load("keyframe_map");


	std::cerr << map.frames.size() << std::endl;
	for (int level = 0; level >= 0; level--) {
		for (int i = 0; i < (level + 1) * (level + 1) * 30; i++) {
			float max_update = map.optimize(level);

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
					map.get_map_pointcloud();

			cloud->header.frame_id = "odom";
			cloud->header.stamp = ros::Time::now();
			cloud->header.seq = 0;
			pointcloud_pub.publish(cloud);

			if (max_update < 1e-4)
				break;
		}
	}


	//map.save("keyframe_map_optimized");

	return 0;

}

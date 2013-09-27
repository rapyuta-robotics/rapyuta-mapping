/*
 * test_panorama.cpp
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

#include "ros/ros.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Int32.h"


int main(int argc, char **argv) {

	keyframe_map map;

	ros::init(argc, argv, "optimize_panorama_map");
	ros::NodeHandle nh;

	ros::Publisher pointcloud_pub = nh.advertise<
			pcl::PointCloud<pcl::PointXYZRGB> >("pointcloud", 1);
   	
	map.load(argv[1]);


	std::cerr << map.frames.size() << std::endl;
	for (int level = 2; level >= 0; level--) {
		for (int i = 0; i < (level + 1) * (level + 1) * 50; i++) {
	        float max_update = map.optimize_panorama(level);

	        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map.get_map_pointcloud();

	        		cloud->header.frame_id = "world";
	        		cloud->header.stamp = ros::Time::now();
	        		cloud->header.seq = 0;
	        		pointcloud_pub.publish(cloud);

			if (max_update < 1e-4)
			    break;
		}
	}

	map.save(std::string(argv[1]) + "_optimized");

	return 0;

}

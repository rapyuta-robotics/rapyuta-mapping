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

void print_map_positions(keyframe_map & map) {
	std::cout << "====================================" << std::endl;

	Eigen::Matrix4f Mbc;

	Mbc << -2.38419e-07,    -0.198669,     0.980067,    0.0391773,
	          -1, -2.38419e-07,            0,       0.0265,
	           0,    -0.980067,     -0.19867,     0.620467,
	           0,            0,            0,            1;

	Eigen::Affine3f Mbcc(Mbc);
	Sophus::SE3f Mbc_pos(Mbcc.rotation(), Mbcc.translation());

	for(int i=0; i<map.frames.size(); i++) {
		Sophus::SE3f pos = map.frames[i]->get_pos() * Mbc_pos.inverse();
		std::cout << i << " " << pos.translation().transpose() << " " << pos.unit_quaternion().coeffs().transpose() << std::endl;
	}
}


int main(int argc, char **argv) {

	keyframe_map map;

	ros::init(argc, argv, "multi_mapper");
	ros::NodeHandle nh;

	ros::Publisher pointcloud_pub = nh.advertise<
			pcl::PointCloud<pcl::PointXYZRGB> >("pointcloud", 1);
   	
	map.load(argv[1]);

	//cv::imshow("img", map.get_panorama_image());
	//cv::waitKey(3);

	//map.frames.resize(33);

	print_map_positions(map);

	std::cerr << map.frames.size() << std::endl;
	for (int level = 2; level >= 0; level--) {
		for (int i = 0; i < (level + 1) * (level + 1) * 50; i++) {
	        float max_update = map.optimize_panorama(level);

	        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map.get_map_pointcloud();

	        		cloud->header.frame_id = "world";
	        		cloud->header.stamp = ros::Time::now();
	        		cloud->header.seq = 0;
	        		pointcloud_pub.publish(cloud);

			//cv::imshow("img", map.get_panorama_image());
			//cv::waitKey(3);

			if (max_update < 1e-4)
			    break;
		}
	}

	print_map_positions(map);

	cv::imshow("img", map.get_panorama_image());
	cv::waitKey();

	map.save(std::string(argv[1]) + "_optimized");

	return 0;

}

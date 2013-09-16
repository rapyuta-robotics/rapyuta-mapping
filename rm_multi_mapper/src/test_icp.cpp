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

	ros::init(argc, argv, "multi_mapper");
	ros::NodeHandle nh;

	ros::Publisher pointcloud_pub = nh.advertise<
			pcl::PointCloud<pcl::PointXYZRGB> >("pointcloud", 1);

	keyframe_map map;
	map.load(argv[1]);
	//map.frames.resize(150);
	//map.align_z_axis();

	print_map_positions(map);


	std::cerr << map.frames.size() << std::endl;
	for (int i = 0; i < 100; i++) {
		map.optimize_g2o();

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map.get_map_pointcloud();

		cloud->header.frame_id = "world";
		cloud->header.stamp = ros::Time::now();
		cloud->header.seq = 0;
		pointcloud_pub.publish(cloud);


	}

	print_map_positions(map);

	return 0;

}

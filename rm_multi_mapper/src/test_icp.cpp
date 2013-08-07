/*
 * test_diem.cpp
 *
 *  Created on: Jul 29, 2013
 *      Author: vsu
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <cmath>

#include <icp_map.h>


int main() {

	icp_map map;

	for (int i = 1; i < 35; i++) {
		cv::Mat depth = cv::imread(
				"depth1/" + boost::lexical_cast<std::string>(i) + ".png",
				CV_LOAD_IMAGE_UNCHANGED);

		cv::Mat rgb = cv::imread(
				"rgb1/" + boost::lexical_cast<std::string>(i) + ".png",
				CV_LOAD_IMAGE_UNCHANGED);

		Sophus::SE3f transform(
				Eigen::AngleAxisf(-0.21 * (i - 1), Eigen::Vector3f(0, 1, 0)).matrix(),
				Eigen::Vector3f(0, 0, 0));

		map.add_frame(rgb, depth, transform);

	}

	pcl::visualization::PCLVisualizer vis;

	vis.removeAllPointClouds();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map.get_map_pointcloud();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
			cloud);
	vis.addPointCloud<pcl::PointXYZRGB>(cloud, rgb);

	vis.spin();

	for (int i = 0; i < 30; i++)
		map.optimize();

	vis.removeAllPointClouds();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 = map.get_map_pointcloud();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1(
			cloud1);
	vis.addPointCloud<pcl::PointXYZRGB>(cloud1, rgb1);

	vis.spin();

	return 0;

}

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

#include <pcl/visualization/pcl_visualizer.h>

#include <icp_map.h>
#include <panorama_map.h>

int main(int argc, char **argv) {
	icp_map map1, map2;
	panorama_map pmap1, pmap2;

	map1.load("icp_map_good3_optimized");
	map2.load("icp_map_good1_optimized");

	cv::Mat img1, img2, depth1, depth2, rgb1, rgb2;
	map1.get_panorama_image(img1, depth1, rgb1);
	map2.get_panorama_image(img2, depth2, rgb2);

	cv::imshow("img1", img1);
	cv::imshow("img2", img2);
	cv::imshow("depth1", depth1);
	cv::imshow("depth2", depth2);
	cv::imshow("rgb1", rgb1);
	cv::imshow("rgb2", rgb2);
	cv::waitKey();

	pmap1.add_frame(img1, depth1, rgb1,
			Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));
	pmap2.add_frame(img2, depth2, rgb2,
			Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

	pmap1.merge(pmap2);

	pcl::visualization::PCLVisualizer vis;

	vis.removeAllPointClouds();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = pmap1.get_pointcloud();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
			cloud);
	vis.addPointCloud<pcl::PointXYZRGB>(cloud, rgb);
	vis.spin();

	return 0;

}

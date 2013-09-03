/*
 * test.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <keyframe.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

int main() {

	Eigen::Vector3f intrinsics;
	intrinsics << 525.0, 319.5, 239.5;

	cv::Mat rgb1 = cv::imread("rgb1/1.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat depth1 = cv::imread("depth1/1.png", CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat rgb2 = cv::imread("rgb1/2.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat depth2 = cv::imread("depth1/2.png", CV_LOAD_IMAGE_UNCHANGED);

	Sophus::SE3f pos1(Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero());
	Sophus::SE3f pos2(Eigen::AngleAxisf(-0.2, Eigen::Vector3f::UnitY()).matrix(),
			Eigen::Vector3f::Zero());

	keyframe k1(rgb1, depth1, pos1, intrinsics);
	keyframe k2(rgb2, depth2, pos2, intrinsics);

	k1.estimate_position(k2);

	/*
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 = k1.get_pointcloud(0);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 = k2.get_pointcloud(0);

	pcl::transformPointCloud(*cloud2, *cloud2, pos2.inverse().matrix());


	pcl::visualization::PCLVisualizer vis;
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> r1(
			cloud1);
	vis.addPointCloud<pcl::PointXYZRGB>(cloud1, r1, "cloud1");
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> r2(
			cloud2);
	vis.addPointCloud<pcl::PointXYZRGB>(cloud2, r2, "cloud2");
	vis.spin();
	*/

	return 0;
}

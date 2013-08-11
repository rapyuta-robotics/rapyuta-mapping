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

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <cmath>

#include <icp_map.h>

int main() {

	icp_map map;
	map.load("icp_map1_optimized");

	std::cerr << map.frames.size() << std::endl;
	//map.frames.resize(map.frames.size() - 21);

	//map.frames.resize(map.frames.size() - 4);

	//cv::imshow("img", map.get_panorama_image() * 255);
	//cv::waitKey();
	pcl::visualization::PCLVisualizer vis;

	{
		vis.removeAllPointClouds();
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map.get_map_pointcloud();
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
				cloud);
		vis.addPointCloud<pcl::PointXYZRGB>(cloud, rgb);
		vis.spin();
	}

	/*
	for (int level = 2; level >= 0; level--) {
		for (int i = 0; i < (level + 1) * (level + 1) * 10; i++) {
			map.optimize_rgb_with_intrinsics(level);

		}
	}

	//for (int i = 0; i < 10; i++) {
	//	map.optimize_rgb_with_intrinsics(0);
	//}

	{
		vis.removeAllPointClouds();
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map.get_map_pointcloud();
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
				cloud);
		vis.addPointCloud<pcl::PointXYZRGB>(cloud, rgb);
		vis.spin();
	}

	map.save("icp_map1_optimized");
*/
	cv::imshow("img", map.get_panorama_image() * 255);
	cv::waitKey();

	for (int i = 0; i < 10; i++) {
		map.optimize();

		vis.removeAllPointClouds();
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 =
				map.get_map_pointcloud();
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1(
				cloud1);
		vis.addPointCloud<pcl::PointXYZRGB>(cloud1, rgb1);

		vis.spinOnce();

	}

	vis.spin();

	return 0;

}

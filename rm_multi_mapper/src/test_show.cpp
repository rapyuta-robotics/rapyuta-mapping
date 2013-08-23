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

#include <keyframe_map.h>

int main(int argc, char **argv) {

	keyframe_map map;
	map.load("keyframe_map_optimized");

	pcl::visualization::PCLVisualizer vis;

	vis.removeAllPointClouds();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map.get_map_pointcloud();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
			cloud);
	vis.addPointCloud<pcl::PointXYZRGB>(cloud, rgb);
	vis.spin();

	return 0;

}

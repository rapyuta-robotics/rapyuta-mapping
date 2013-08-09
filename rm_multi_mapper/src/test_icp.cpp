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
	map.load("icp_map1");
	map.save("icp_map2");

	pcl::visualization::PCLVisualizer vis;

	vis.removeAllPointClouds();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map.get_map_pointcloud();
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
			cloud);
	vis.addPointCloud<pcl::PointXYZRGB>(cloud, rgb);

	vis.spin();

	for (int level = 2; level >= 0; level--) {
		for (int i = 0; i < (level+1)*20; i++) {
			map.optimize_rgb(level);

			vis.removeAllPointClouds();
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 =
					map.get_map_pointcloud();
			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1(
					cloud1);
			vis.addPointCloud<pcl::PointXYZRGB>(cloud1, rgb1);

			vis.spinOnce();

		}
	}

	//vis.spin();

	for (int level = 1; level >= 0; level--) {
			for (int i = 0; i < (level+1)*20; i++) {
				map.optimize_rgb_with_intrinsics(level);

				vis.removeAllPointClouds();
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 =
						map.get_map_pointcloud();
				pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb1(
						cloud1);
				vis.addPointCloud<pcl::PointXYZRGB>(cloud1, rgb1);

				vis.spinOnce();

			}
		}

	vis.spin();

	return 0;

}

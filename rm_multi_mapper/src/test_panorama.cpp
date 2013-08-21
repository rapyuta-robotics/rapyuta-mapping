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
#include <opencv2/stitching/stitcher.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>

#include <icp_map.h>

int main(int argc, char **argv) {

	icp_map map;
	map.load("icp_map_good1");

	{
		cv::Mat img, depth, rgb;
		map.get_panorama_image(img, depth, rgb);
		cv::imshow("img", img);
		cv::waitKey();
	}

	std::cerr << map.frames.size() << std::endl;
	for (int level = 2; level >= 0; level--) {
		for (int i = 0; i < (level + 1) * (level + 1) * 12; i++) {
			float max_update = map.optimize_rgb(level);

			if (max_update < 1e-4)
				break;
		}
	}

	{
		cv::Mat img, depth, rgb;
		map.get_panorama_image(img, depth, rgb);
		cv::imshow("img", img);
		cv::waitKey();
	}

	map.save("icp_map_good2_optimized");

	return 0;

}

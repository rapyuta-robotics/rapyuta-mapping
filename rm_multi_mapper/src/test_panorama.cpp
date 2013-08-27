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

int main(int argc, char **argv) {

	keyframe_map map;
	map.load(argv[1]);

	cv::imshow("img", map.get_panorama_image());
	cv::waitKey(3);

	std::cerr << map.frames.size() << std::endl;
	for (int level = 2; level >= 0; level--) {
		for (int i = 0; i < (level + 1) * (level + 1) * 10; i++) {
			float max_update = map.optimize_panorama(level);

			cv::imshow("img", map.get_panorama_image());
			cv::waitKey(3);

			if (max_update < 1e-4)
				break;
		}
	}

	cv::imshow("img", map.get_panorama_image());
	cv::waitKey();

	map.save(std::string(argv[1]) + "_optimized");

	return 0;

}

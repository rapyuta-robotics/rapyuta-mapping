/*
 * test.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <robot_mapper.h>

int main(int argc, char **argv) {

	ros::init(argc, argv, "test_map");
	ros::NodeHandle nh;

	boost::shared_ptr<keypoint_map> map;

	Eigen::Vector4f intrinsics;
	intrinsics << 525.0, 525.0, 319.5, 239.5;

	Eigen::Affine3f transform;
	transform.setIdentity();

	for(int j=0; j<3; j++) {

	for (int i = 0; i < 36; i++) {

		cv::Mat rgb = cv::imread("rgb1/" + boost::lexical_cast<std::string>(i) + ".png",
					CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat depth = cv::imread("depth1/" + boost::lexical_cast<std::string>(i) + ".png",
					CV_LOAD_IMAGE_UNCHANGED);

		if (map.get()) {
				keypoint_map map1(rgb, depth, transform, intrinsics);
				map->merge_keypoint_map(map1, 50, 300);

			} else {
				map.reset(new keypoint_map(rgb, depth, transform, intrinsics));
			}
	}

	map->remove_bad_points(1);
	map->optimize();

	}

	map->save("test2_final_map");

	return 0;
}

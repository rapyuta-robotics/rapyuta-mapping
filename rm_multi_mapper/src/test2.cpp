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

	int num_robots = 2;
	std::string prefix = "cloudbot";
	std::vector<robot_mapper::Ptr> robot_mappers(num_robots);

	boost::thread_group tg;

	for (int i = 0; i < num_robots; i++) {
		robot_mappers[i].reset(new robot_mapper(nh, prefix, i + 1));
		robot_mappers[i]->map.reset(
				new keypoint_map(
						"maps/cloudbot"
								+ boost::lexical_cast<std::string>(i + 1)
								+ "_full"));
	}

	//for (int i = 0; i < num_robots; i++) {
	//		robot_mappers[i]->set_map();
	//}

	if (robot_mappers[0]->map->merge_keypoint_map(*robot_mappers[1]->map, 50,
			5000)) {
		ROS_INFO("Merged 2 maps");
		robot_mappers[0]->map->save("merged_map");
	} else {
		ROS_INFO("Could not merge 2 maps");
	}

	//ros::spin();

	return 0;
}

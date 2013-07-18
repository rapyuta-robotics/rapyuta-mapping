#include <ros/ros.h>

#include <robot_mapper.h>


int main(int argc, char **argv) {
	ros::init(argc, argv, "multi_mapper");

	ros::NodeHandle nh;

	int num_robots = 1;
	std::string prefix = "cloudbot";
	std::vector<robot_mapper::Ptr> robot_mappers(num_robots);

	boost::thread_group tg;

	for (int i = 0; i < num_robots; i++) {
		robot_mappers[i].reset(new robot_mapper(nh, prefix, i + 2));
	}

	for (int i = 0; i < num_robots; i++) {
		tg.create_thread(boost::bind(&robot_mapper::capture_sphere, robot_mappers[i].get()));
	}

	tg.join_all();

	ROS_INFO("All threads finished successfully");

	if(robot_mappers[0]->map->merge_keypoint_map(*robot_mappers[1]->map)) {
		ROS_INFO("Merged 2 maps");
		robot_mappers[0]->map->save("merged_map");
	}

	return 0;
}

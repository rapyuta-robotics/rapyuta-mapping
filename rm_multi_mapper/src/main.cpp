#include <ros/ros.h>

#include <robot_mapper.h>

const int robot_offset = 0;
const int num_robots = 1;
const std::string prefix = "cloudbot";
std::vector<robot_mapper::Ptr> robot_mappers(num_robots);
boost::thread_group tg;

bool start_capturing(std_srvs::Empty::Request &req,
		std_srvs::Empty::Response &res) {

	/*for (int i = 0; i < num_robots; i++) {
	 tg.create_thread(
	 boost::bind(&robot_mapper::move_straight,
	 robot_mappers[i].get()));
	 }*/


	for (int i = 0; i < num_robots; i++) {
		tg.create_thread(
				boost::bind(&robot_mapper::capture_sphere,
						robot_mappers[i].get()));
	}
	tg.join_all();

	for (int i = 0; i < num_robots; i++) {
		tg.create_thread(
				boost::bind(&robot_mapper::optmize_panorama,
						robot_mappers[i].get()));
	}
	tg.join_all();

		while (!robot_mappers[0]->merge(*robot_mappers[1])) {

		}
		ROS_INFO("Merged maps");


	return true;
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "multi_mapper");

	ros::NodeHandle nh;

	for (int i = 0; i < num_robots; i++) {
		robot_mappers[i].reset(new robot_mapper(nh, prefix, i + robot_offset));
	}

	ros::ServiceServer send_all_keyframes_service = nh.advertiseService(
			"start_moving", start_capturing);

	ros::AsyncSpinner spinner(4);
	spinner.start();

	//int iteration = 0;
	//while (!robot_mappers[0]->merge(*robot_mappers[1])) {
	//ROS_INFO("Merging iteration %d", iteration);
	//iteration++;
	//}
	//ROS_INFO("Merged maps");

	ros::waitForShutdown();

	return 0;
}

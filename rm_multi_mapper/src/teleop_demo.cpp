#include <ros/ros.h>

#include <robot_mapper.h>

const int robot_offset = 2;
const int num_robots = 1;
const std::string prefix = "cloudbot";
std::vector<robot_mapper::Ptr> robot_mappers(num_robots);

template<typename F>
void run_on_all_robots(F f) {
	boost::thread_group tg;

	for (int i = 0; i < num_robots; i++) {
		tg.create_thread(boost::bind(f, robot_mappers[i].get()));
	}

	tg.join_all();
}

template<typename F, typename P>
void run_on_all_robots(F f, P p) {
	boost::thread_group tg;

	for (int i = 0; i < num_robots; i++) {
		tg.create_thread(boost::bind(f, robot_mappers[i].get(), p));
	}

	tg.join_all();
}

bool start_capturing(std_srvs::Empty::Request &req,
		std_srvs::Empty::Response &res) {

	run_on_all_robots(&robot_mapper::save_map, "teleop_");

	return true;
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "multi_mapper");

	ros::NodeHandle nh;

	for (int i = 0; i < num_robots; i++) {
		robot_mappers[i].reset(new robot_mapper(nh, prefix, i + robot_offset));
	}

	ros::ServiceServer send_all_keyframes_service = nh.advertiseService(
			"save_map", start_capturing);

	ros::AsyncSpinner spinner(4);
	spinner.start();

	//run_on_all_robots(&robot_mapper::start_optimization_loop);

	ros::waitForShutdown();

	return 0;
}

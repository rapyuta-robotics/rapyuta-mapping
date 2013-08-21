#include <ros/ros.h>

#include <robot_mapper.h>

const int robot_offset = 0;
const int num_robots = 1;
const std::string prefix = "cloudbot";
std::vector<robot_mapper::Ptr> robot_mappers(num_robots);

void publishTf() {

	tf::TransformBroadcaster br;

	while (true) {
		for (int i = 0; i < num_robots; i++) {

			Eigen::Vector3f offset = robot_mappers[i]->visualization_offset;

			tf::Transform map_to_odom;
			map_to_odom.setIdentity();
			map_to_odom.setOrigin(tf::Vector3(offset[0], offset[1], offset[2]));

			br.sendTransform(
					tf::StampedTransform(map_to_odom, ros::Time::now(),
							"/world",
							"/cloudbot"
									+ boost::lexical_cast<std::string>(
											i + robot_offset) + "/map"));

			/*
			//robot_mappers[i]->update_map_to_odom();
			map_to_odom.setIdentity();
			br.sendTransform(
					tf::StampedTransform(map_to_odom,
							ros::Time::now(),
							"/cloudbot"
									+ boost::lexical_cast<std::string>(
											i + robot_offset) + "/map",
							"/cloudbot"
									+ boost::lexical_cast<std::string>(
											i + robot_offset)
									+ "/odom_combined"));
			*/

		}
		usleep(33000);
	}

}

int main(int argc, char **argv) {
	ros::init(argc, argv, "multi_mapper");

	ros::NodeHandle nh;

	boost::thread_group tg;

	for (int i = 0; i < num_robots; i++) {
		robot_mappers[i].reset(new robot_mapper(nh, prefix, i + robot_offset));
	}

	boost::thread t(publishTf);

	while (true) {
		for (int i = 0; i < num_robots; i++) {
			tg.create_thread(
					boost::bind(&robot_mapper::capture_sphere,
							robot_mappers[i].get()));
		}

		tg.join_all();

		for (int i = 0; i < num_robots; i++) {
			tg.create_thread(
					boost::bind(&robot_mapper::move_to_random_point,
							robot_mappers[i].get()));
		}
		tg.join_all();
	}

	return 0;
}

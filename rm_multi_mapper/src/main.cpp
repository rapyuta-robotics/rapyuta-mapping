#include <ros/ros.h>

#include <robot_mapper.h>

const int num_robots = 2;
const std::string prefix = "cloudbot";

void publishTf() {

	tf::TransformBroadcaster br;

	while (true) {
		for (int i = 0; i < num_robots; i++) {

			tf::Transform map_to_odom;
			map_to_odom.setIdentity();
			map_to_odom.setOrigin(tf::Vector3(0, i * 20, 0));

			br.sendTransform(
					tf::StampedTransform(map_to_odom, ros::Time::now(),
							"/world",
							"/cloudbot"
									+ boost::lexical_cast<std::string>(i + 1)
									+ "/map"));
		}
		usleep(33000);
	}

}

int main(int argc, char **argv) {
	ros::init(argc, argv, "multi_mapper");

	ros::NodeHandle nh;

	boost::thread t(publishTf);

	std::vector<robot_mapper::Ptr> robot_mappers(num_robots);
	boost::thread_group tg;

	for (int i = 0; i < num_robots; i++) {
		robot_mappers[i].reset(new robot_mapper(nh, prefix, i + 1));
	}

	while (true) {
		for (int i = 0; i < num_robots; i++) {
			tg.create_thread(
					boost::bind(&robot_mapper::capture_sphere,
							robot_mappers[i].get()));
		}

		tg.join_all();

		for (int i = 0; i < num_robots; i++) {
			tg.create_thread(
					boost::bind(&robot_mapper::set_map,
							robot_mappers[i].get()));
		}

		tg.join_all();

		for (int i = 0; i < num_robots; i++) {
			tg.create_thread(
					boost::bind(&robot_mapper::move_to_random_point,
							robot_mappers[i].get()));
		}

		ROS_INFO("All threads finished successfully");

		for (int i = 0; i < num_robots; i++) {
			robot_mappers[i]->map->save(
					"maps/cloudbot" + boost::lexical_cast<std::string>(i + 1)
							+ "_full");
		}

		for (int i = 0; i < num_robots; i++) {
			if (!robot_mappers[i]->merged) {
				for (int j = 0; j < num_robots; j++) {
					if (!robot_mappers[j]->merged && i < j) {
						robot_mappers[i]->merge(robot_mappers[j]);
					}

				}
			}

		}

	}

	return 0;
}

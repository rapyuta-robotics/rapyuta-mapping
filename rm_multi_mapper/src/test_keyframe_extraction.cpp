/*
 * test_keyframe_extraction.cpp
 *
 *  Created on: Aug 22, 2013
 *      Author: vsu
 */

#include <keyframe_map.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <turtlebot_actions/TurtlebotMoveAction.h>
#include <std_msgs/Float32.h>
#include <rm_localization/UpdateMap.h>

class TestKeyframeExtraction {

public:

	actionlib::SimpleActionClient<turtlebot_actions::TurtlebotMoveAction> action_client;

	boost::shared_ptr<keyframe_map> map;
	ros::Publisher pointcloud_pub;
	ros::Publisher servo_pub;
	ros::Subscriber keyframe_sub;

	ros::ServiceClient update_map_service;

	TestKeyframeExtraction(ros::NodeHandle & nh) :
			action_client("/turtlebot_move", true), map(new keyframe_map) {
		pointcloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
				"pointcloud", 1);

		servo_pub = nh.advertise<std_msgs::Float32>(
				"mobile_base/commands/servo_angle", 3);

		keyframe_sub = nh.subscribe("keyframe", 10,
				&TestKeyframeExtraction::chatterCallback, this);

		update_map_service = nh.serviceClient<rm_localization::UpdateMap>(
				"update_map");
	}

	void chatterCallback(const rm_localization::Keyframe::ConstPtr& msg) {
		ROS_INFO("Received keyframe");
		map->add_frame(msg);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
				map->get_map_pointcloud();
		cloud->header.frame_id = "odom";
		cloud->header.stamp = ros::Time::now();
		cloud->header.seq = 0;
		pointcloud_pub.publish(cloud);

	}

	void turn() {
		turtlebot_actions::TurtlebotMoveGoal goal;
		goal.forward_distance = 0;
		goal.turn_distance = M_PI;

		action_client.waitForServer();
		action_client.sendGoal(goal);

		//wait for the action to return
		bool finished_before_timeout = action_client.waitForResult(
				ros::Duration(30.0));

		if (finished_before_timeout) {
			actionlib::SimpleClientGoalState state = action_client.getState();
			ROS_INFO("Action finished: %s", state.toString().c_str());
		} else
			ROS_INFO("Action did not finish before the time out.");
	}

	void capture_sphere() {

		std_msgs::Float32 angle_msg;
		angle_msg.data = 0 * M_PI / 18;
		float stop_angle = -0.1 * M_PI / 18;
		float delta = M_PI / 18;

		sleep(3);

		for (; angle_msg.data > stop_angle; angle_msg.data -= delta) {

			servo_pub.publish(angle_msg);
			sleep(1);

			turn();
			turn();

		}

		map->save("keyframe_map");

	}

	void optmize_panorama() {

		for (int level = 2; level >= 0; level--) {
			for (int i = 0; i < (level + 1) * (level + 1) * 10; i++) {
				float max_update = map->optimize_panorama(level);

				rm_localization::UpdateMap update_map_msg;
				update_map_msg.request.idx.resize(map->frames.size());
				update_map_msg.request.transform.resize(map->frames.size());

				update_map_msg.request.intrinsics = {{0,0,0}};

				for (size_t i = 0; i < map->frames.size(); i++) {
					update_map_msg.request.idx[i] = map->idx[i];

					Sophus::SE3f position = map->frames[i]->get_pos();

					memcpy(
							update_map_msg.request.transform[i].unit_quaternion.data(),
							position.unit_quaternion().coeffs().data(),
							4 * sizeof(float));

					memcpy(update_map_msg.request.transform[i].position.data(),
							position.translation().data(), 3 * sizeof(float));

				}

				update_map_service.call(update_map_msg);

				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
						map->get_map_pointcloud();

				cloud->header.frame_id = "odom";
				cloud->header.stamp = ros::Time::now();
				cloud->header.seq = 0;
				pointcloud_pub.publish(cloud);

				usleep(100000);

				if (max_update < 1e-4)
					break;
			}
		}

	}

};

int main(int argc, char **argv) {
	ros::init(argc, argv, "multi_mapper");
	ros::NodeHandle nh;

	TestKeyframeExtraction t(nh);

	ros::AsyncSpinner spinner(4);
	spinner.start();

	t.capture_sphere();
	t.optmize_panorama();

	ros::waitForShutdown();

	return 0;
}

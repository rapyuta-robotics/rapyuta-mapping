/*
 * test_keyframe_extraction.cpp
 *
 *  Created on: Aug 22, 2013
 *      Author: vsu
 */

#include <keyframe_map.h>
#include <sensor_msgs/SetCameraInfo.h>
#include <sensor_msgs/distortion_models.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <turtlebot_actions/TurtlebotMoveAction.h>
#include <std_msgs/Float32.h>
#include <std_srvs/Empty.h>
#include <rm_localization/UpdateMap.h>

class TestKeyframeExtraction {

public:

	actionlib::SimpleActionClient<turtlebot_actions::TurtlebotMoveAction> action_client;

	boost::shared_ptr<keyframe_map> map;
	ros::Publisher pointcloud_pub;
	ros::Publisher servo_pub;
	ros::Subscriber keyframe_sub;

	ros::ServiceClient update_map_service;
	ros::ServiceClient clear_keyframes_service;
	ros::ServiceClient set_camera_info_service;

	TestKeyframeExtraction(ros::NodeHandle & nh) :
			action_client("/cloudbot1/turtlebot_move", true), map(
					new keyframe_map) {
		pointcloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
				"/cloudbot1/pointcloud", 1);

		servo_pub = nh.advertise<std_msgs::Float32>(
				"/cloudbot1/mobile_base/commands/servo_angle", 3);

		keyframe_sub = nh.subscribe("/cloudbot1/keyframe", 10,
				&TestKeyframeExtraction::chatterCallback, this);

		update_map_service = nh.serviceClient<rm_localization::UpdateMap>(
				"/cloudbot1/update_map");

		clear_keyframes_service = nh.serviceClient<std_srvs::Empty>(
				"/cloudbot1/clear_keyframes");

		set_camera_info_service = nh.serviceClient<sensor_msgs::SetCameraInfo>(
				"/cloudbot1/rgb/set_camera_info");

		std_srvs::Empty emp;
		clear_keyframes_service.call(emp);

	}

	void chatterCallback(const rm_localization::Keyframe::ConstPtr& msg) {
		ROS_INFO("Received keyframe");
		map->add_frame(msg);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
				map->get_map_pointcloud();
		cloud->header.frame_id = "/cloudbot1/odom";
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

	void move_straight() {

		float current_angle = 0;
		float desired_angle = 0;

		/*
		 if (map->frames.size() > 0) {
		 Sophus::SE3f position =
		 map->frames[map->frames.size() - 1]->get_pos();
		 Eigen::Vector3f current_heading = position.unit_quaternion()
		 * Eigen::Vector3f::UnitZ();

		 current_angle = std::atan2(-current_heading(1), current_heading(0));
		 desired_angle = std::asin(position.translation()(1)/10.0);
		 }*/

		turtlebot_actions::TurtlebotMoveGoal goal;
		goal.forward_distance = 25.0;
		goal.turn_distance = current_angle - desired_angle;

		action_client.waitForServer();
		action_client.sendGoal(goal);

		//wait for the action to return
		bool finished_before_timeout = action_client.waitForResult(
				ros::Duration(500.0));

		if (finished_before_timeout) {
			actionlib::SimpleClientGoalState state = action_client.getState();
			ROS_INFO("Action finished: %s", state.toString().c_str());
		} else
			ROS_INFO("Action did not finish before the time out.");

		map->save("corridor_map");

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

				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
						map->get_map_pointcloud();

				update_map();

				cloud->header.frame_id = "/cloudbot1/odom";
				cloud->header.stamp = ros::Time::now();
				cloud->header.seq = 0;
				pointcloud_pub.publish(cloud);

				usleep(100000);

				if (max_update < 1e-4)
					break;
			}
		}

		update_map(true);

	}

	void optmize() {

		if (map->frames.size() < 2)
			return;

		map->optimize_slam();

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
				map->get_map_pointcloud();

		cloud->header.frame_id = "odom";
		cloud->header.stamp = ros::Time::now();
		cloud->header.seq = 0;
		pointcloud_pub.publish(cloud);

		update_map();

	}

	void update_map(bool with_intrinsics = false) {

		rm_localization::UpdateMap update_map_msg;
		update_map_msg.request.idx.resize(map->frames.size());
		update_map_msg.request.transform.resize(map->frames.size());

		if (with_intrinsics) {

			Eigen::Vector3f intrinsics = map->frames[0]->get_intrinsics();

			sensor_msgs::SetCameraInfo s;
			s.request.camera_info.width = map->frames[0]->get_i(0).cols;
			s.request.camera_info.height = map->frames[0]->get_i(0).rows;

			// No distortion
			s.request.camera_info.D.resize(5, 0.0);
			s.request.camera_info.distortion_model =
					sensor_msgs::distortion_models::PLUMB_BOB;

			// Simple camera matrix: square pixels (fx = fy), principal point at center
			s.request.camera_info.K.assign(0.0);
			s.request.camera_info.K[0] = s.request.camera_info.K[4] =
					intrinsics[0];
			s.request.camera_info.K[2] = intrinsics[1];
			s.request.camera_info.K[5] = intrinsics[2];
			s.request.camera_info.K[8] = 1.0;

			// No separate rectified image plane, so R = I
			s.request.camera_info.R.assign(0.0);
			s.request.camera_info.R[0] = s.request.camera_info.R[4] =
					s.request.camera_info.R[8] = 1.0;

			// Then P=K(I|0) = (K|0)
			s.request.camera_info.P.assign(0.0);
			s.request.camera_info.P[0] = s.request.camera_info.P[5] =
					s.request.camera_info.K[0]; // fx, fy
			s.request.camera_info.P[2] = s.request.camera_info.K[2]; // cx
			s.request.camera_info.P[6] = s.request.camera_info.K[5]; // cy
			s.request.camera_info.P[10] = 1.0;

			set_camera_info_service.call(s);

			memcpy(update_map_msg.request.intrinsics.data(), intrinsics.data(),
					3 * sizeof(float));
		} else {
			update_map_msg.request.intrinsics = { {0,0,0}};
		}

		for (size_t i = 0; i < map->frames.size(); i++) {
			update_map_msg.request.idx[i] = map->idx[i];

			Sophus::SE3f position = map->frames[i]->get_pos();

			memcpy(update_map_msg.request.transform[i].unit_quaternion.data(),
					position.unit_quaternion().coeffs().data(),
					4 * sizeof(float));

			memcpy(update_map_msg.request.transform[i].position.data(),
					position.translation().data(), 3 * sizeof(float));

		}

		update_map_service.call(update_map_msg);

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

	//	t.move_straight();

	//while (true) {
	//	t.optmize();
	//}

	ros::waitForShutdown();

	return 0;
}

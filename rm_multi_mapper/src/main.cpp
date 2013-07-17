#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <turtlebot_actions/TurtlebotMoveAction.h>
#include <rm_capture_server/Capture.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <std_msgs/Float32.h>
#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/OccupancyGrid.h>
#include <octomap/OcTree.h>
#include <std_srvs/Empty.h>
#include <rm_localization/SetMap.h>

#include <move_base_msgs/MoveBaseAction.h>
#include <octomap_msgs/BoundingBoxQuery.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <tf/transform_broadcaster.h>
#include <pcl_ros/publisher.h>

#include <util.h>

class robot_mapper {

public:

	typedef boost::shared_ptr<robot_mapper> Ptr;

	robot_mapper(ros::NodeHandle & nh, const std::string & robot_prefix,
			const int robot_num) :
			prefix(
					"/" + robot_prefix
							+ boost::lexical_cast<std::string>(robot_num)), move_base_action_client(
					prefix + "/move_base", true) {

		capture_client = nh.serviceClient<rm_capture_server::Capture>(
				prefix + "/capture");

		clear_unknown_space_client = nh.serviceClient<std_srvs::Empty>(
				prefix + "/move_base/clear_unknown_space");

		servo_pub = nh.advertise<std_msgs::Float32>(
				prefix + "/mobile_base/commands/servo_angle", 1);

		pub_keypoints = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >(
				prefix + "/keypoints", 10);

		set_map_client = nh.serviceClient<rm_localization::SetMap>(
				prefix + "/set_map");

	}

	void capture_sphere() {

		for (int i = 0; i < 12; i++) {
			move_base_msgs::MoveBaseGoal goal;
			goal.target_pose.header.frame_id = "base_link";
			goal.target_pose.header.stamp = ros::Time::now();

			goal.target_pose.pose.position.x = 0;
			goal.target_pose.pose.position.y = 0;
			goal.target_pose.pose.position.z = 0;

			tf::Quaternion q;
			q.setRotation(tf::Vector3(0, 0, 1), M_PI / 6);
			tf::quaternionTFToMsg(q, goal.target_pose.pose.orientation);

			move_base_action_client.sendGoal(goal);

			//wait for the action to return
			bool finished_before_timeout =
					move_base_action_client.waitForResult(ros::Duration(30.0));

			if (finished_before_timeout) {
				actionlib::SimpleClientGoalState state =
						move_base_action_client.getState();
				ROS_INFO("Action finished: %s", state.toString().c_str());
			} else
				ROS_INFO("Action did not finish before the time out.");

			for (float angle = -M_PI / 4; angle <= M_PI / 4;
					angle += M_PI / 18) {

				std_msgs::Float32 angle_msg;
				angle_msg.data = angle;
				servo_pub.publish(angle_msg);
				sleep(1);

				rm_capture_server::Capture srv;
				srv.request.num_frames = 1;
				if (capture_client.call(srv)) {

					cv::Mat rgb = cv::imdecode(srv.response.rgb_png_data,
							CV_LOAD_IMAGE_UNCHANGED);

					cv::Mat depth = cv::imdecode(srv.response.depth_png_data,
							CV_LOAD_IMAGE_UNCHANGED);

					Eigen::Affine3d transform_d;
					tf::transformMsgToEigen(srv.response.transform,
							transform_d);
					Eigen::Affine3f transform = transform_d.cast<float>();

					if (map.get()) {
						keypoint_map map1(rgb, depth, transform);
						map->merge_keypoint_map(map1);
						map->keypoints3d.header.frame_id = "/map";
						map->keypoints3d.header.stamp = ros::Time::now();
						map->keypoints3d.header.seq = i;

						pub_keypoints.publish(map->keypoints3d);

					} else {
						map.reset(new keypoint_map(rgb, depth, transform));
						map->align_z_axis();
						map->save("map_aligned");
					}

					std::cerr << map->keypoints3d.size() << " "
							<< map->descriptors.rows << " "
							<< map->weights.size() << std::endl;

				} else {
					ROS_ERROR("Failed to call service /cloudbot1/capture");
					return;
				}

			}
		}

		std_msgs::Float32 angle_msg;
		angle_msg.data = 0;
		servo_pub.publish(angle_msg);

		map->optimize();

	}

	void set_map() {
		rm_localization::SetMap data;
		pcl::toROSMsg(map->keypoints3d, data.request.keypoints3d);

		cv_bridge::CvImage desc;
		desc.image = map->descriptors;
		desc.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
		data.request.descriptors = *(desc.toImageMsg());

		set_map_client.call(data);
	}

	void move_to_random_point() {

		while (true) {

			std_srvs::Empty empty;
			clear_unknown_space_client.call(empty);

			move_base_msgs::MoveBaseGoal goal;
			goal.target_pose.header.frame_id = "base_link";
			goal.target_pose.header.stamp = ros::Time::now();

			goal.target_pose.pose.position.x =
					1.0 * ((float) rand()) / RAND_MAX;
			goal.target_pose.pose.position.y =
					1.0 * ((float) rand()) / RAND_MAX;
			goal.target_pose.pose.position.z = 0;

			tf::Quaternion q;
			q.setEuler(0, 0, 0);
			tf::quaternionTFToMsg(q, goal.target_pose.pose.orientation);

			move_base_action_client.sendGoal(goal);

			//wait for the action to return
			bool finished_before_timeout =
					move_base_action_client.waitForResult(ros::Duration(30.0));

			if (finished_before_timeout) {
				actionlib::SimpleClientGoalState state =
						move_base_action_client.getState();
				ROS_INFO("Action finished: %s", state.toString().c_str());
				if (state == actionlib::SimpleClientGoalState::SUCCEEDED)
					break;
			} else
				ROS_INFO("Action did not finish before the time out.");

		}

	}

	std::string prefix;

	actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> move_base_action_client;
	ros::ServiceClient capture_client;
	ros::ServiceClient clear_unknown_space_client;
	ros::ServiceClient set_map_client;
	ros::Publisher servo_pub;
	ros::Publisher pub_keypoints;

	boost::shared_ptr<keypoint_map> map;

};

int main(int argc, char **argv) {
	ros::init(argc, argv, "multi_mapper");

	ros::NodeHandle nh;

	int num_robots = 2;
	std::string prefix = "cloudbot";
	std::vector<robot_mapper::Ptr> robot_mappers(2);

	boost::thread_group tg;

	for (int i = 0; i < num_robots; i++) {
		robot_mappers[i].reset(new robot_mapper(nh, prefix, i + 1));
	}

	for (int i = 0; i < num_robots; i++) {
		tg.create_thread(boost::bind(&robot_mapper::capture_sphere, robot_mappers[i].get()));
	}

	tg.join_all();

	ROS_INFO("All threads finished successfully");

	return 0;
}

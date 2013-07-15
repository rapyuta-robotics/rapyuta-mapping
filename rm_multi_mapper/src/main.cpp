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

int main(int argc, char **argv) {
	ros::init(argc, argv, "multi_mapper");

	ros::NodeHandle nh;

	boost::shared_ptr<keypoint_map> map;

	// create the action client
	// true causes the client to spin its own thread
	actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> ac(
			"/cloudbot1/move_base", true);

	ros::ServiceClient client = nh.serviceClient<rm_capture_server::Capture>(
			"/cloudbot1/capture");

	ros::Publisher servo_pub = nh.advertise<std_msgs::Float32>(
			"/cloudbot1/mobile_base/commands/servo_angle", 1);

	ros::Publisher pub_cloud =
			nh.advertise<pcl::PointCloud<pcl::PointXYZRGBA> >("/map_cloud", 10);

	ros::Publisher pub_keypoints =
			nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("keypoints", 10);

	ros::ServiceClient octomap_reset = nh.serviceClient<std_srvs::Empty>(
			"/octomap_server/reset");

	ros::ServiceClient set_map_client =
			nh.serviceClient<rm_localization::SetMap>("/cloudbot1/set_map");

	ROS_INFO("Waiting for action server to start.");
	// wait for the action server to start
	ac.waitForServer(); //will wait for infinite time

	ROS_INFO("Action server started, sending goal.");
	// send a goal to the action

	for (int j = 0; j < 3; j++) {
		for (int i = 0; i < 12; i++) {
			move_base_msgs::MoveBaseGoal goal;
			goal.target_pose.header.frame_id = "base_link";
			goal.target_pose.header.stamp = ros::Time::now();

			goal.target_pose.pose.position.x = 0;
			goal.target_pose.pose.position.y = 0;
			goal.target_pose.pose.position.z = 0;

			tf::Quaternion q;
			q.setRotation(tf::Vector3(0, 0, 1), M_PI / 4);
			tf::quaternionTFToMsg(q, goal.target_pose.pose.orientation);

			ac.sendGoal(goal);

			//wait for the action to return
			bool finished_before_timeout = ac.waitForResult(
					ros::Duration(30.0));

			if (finished_before_timeout) {
				actionlib::SimpleClientGoalState state = ac.getState();
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
				if (client.call(srv)) {

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
						map->keypoints3d.header.seq = 12 * j + i;

						pub_keypoints.publish(map->keypoints3d);

					} else {
						map.reset(new keypoint_map(rgb, depth, transform));
						map->align_z_axis();
						map->save("map_aligned");
					}

					std::cerr << map->keypoints3d.size() << " "
							<< map->descriptors.rows << " "
							<< map->weights.size() << std::endl;

					//vis.removeAllPointClouds();
					//vis.addPointCloud<pcl::PointXYZ>(
					//		accumulated_keypoints3d.makeShared(), "keypoints");
					//vis.spinOnce(2);

					//cv::imshow("Image", rgb);
					//cv::waitKey(2);

				} else {
					ROS_ERROR("Failed to call service /cloudbot1/capture");
					return 1;
				}

			}
		}

		std_msgs::Float32 angle_msg;
		angle_msg.data = 0;
		servo_pub.publish(angle_msg);
		sleep(1);

		move_base_msgs::MoveBaseGoal goal;
		goal.target_pose.header.frame_id = "/map";
		goal.target_pose.header.stamp = ros::Time::now();

		goal.target_pose.pose.position.x = j * 1.0;
		goal.target_pose.pose.position.y = 0;
		goal.target_pose.pose.position.z = 0;

		tf::Quaternion q;
		q.setEuler(0, 0, 0);
		tf::quaternionTFToMsg(q, goal.target_pose.pose.orientation);

		ac.sendGoal(goal);

		//turtlebot_actions::TurtlebotMoveGoal goal;
		//goal.forward_distance = 0.5;
		//goal.turn_distance = 0;
		//ac.sendGoal(goal);

		//wait for the action to return
		bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

		if (finished_before_timeout) {
			actionlib::SimpleClientGoalState state = ac.getState();
			ROS_INFO("Action finished: %s", state.toString().c_str());
		} else
			ROS_INFO("Action did not finish before the time out.");

		map->save("map_" + boost::lexical_cast<std::string>(j) + "_all");
		map->remove_bad_points(1);
		map->save("map_" + boost::lexical_cast<std::string>(j));
		map->optimize();

		std_srvs::Empty msg;
		octomap_reset.call(msg);
		map->publish_keypoints(pub_cloud);

		rm_localization::SetMap data;

		pcl::toROSMsg(map->keypoints3d, data.request.keypoints3d);

		cv_bridge::CvImage desc;
		desc.image = map->descriptors;
		desc.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
		data.request.descriptors = *(desc.toImageMsg());

		set_map_client.call(data);

	}

	pcl::visualization::PCLVisualizer vis;
	vis.removeAllPointClouds();
	vis.addPointCloud<pcl::PointXYZ>(map->keypoints3d.makeShared(),
			"keypoints");
	vis.spin();

	std::cerr << "Error " << map->compute_error() << " Mean error "
			<< map->compute_error() / map->observations.size() << std::endl;

	map->optimize();

	std::cerr << "Error " << map->compute_error() << " Mean error "
			<< map->compute_error() / map->observations.size() << std::endl;

	vis.removeAllPointClouds();
	vis.addPointCloud<pcl::PointXYZ>(map->keypoints3d.makeShared(),
			"keypoints");
	vis.spin();

//exit
	return 0;
}

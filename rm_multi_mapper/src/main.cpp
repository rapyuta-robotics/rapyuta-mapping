#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <turtlebot_actions/TurtlebotMoveAction.h>
#include <rm_capture_server/Capture.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <std_msgs/Float32.h>

#include <util.h>

int main(int argc, char **argv) {
	ros::init(argc, argv, "test_fibonacci");

	ros::NodeHandle nh;

	boost::shared_ptr<keypoint_map> map;

	pcl::visualization::PCLVisualizer vis;

	// create the action client
	// true causes the client to spin its own thread
	actionlib::SimpleActionClient<turtlebot_actions::TurtlebotMoveAction> ac(
			"/cloudbot1/turtlebot_move", true);

	ros::ServiceClient client = nh.serviceClient<rm_capture_server::Capture>(
			"/cloudbot1/capture");

	ros::Publisher servo_pub = nh.advertise<std_msgs::Float32>(
			"/cloudbot1/mobile_base/commands/servo_angle", 1);

	ROS_INFO("Waiting for action server to start.");
	// wait for the action server to start
	ac.waitForServer(); //will wait for infinite time

	ROS_INFO("Action server started, sending goal.");
	// send a goal to the action

	for (int i = 0; i < 18; i++) {
		turtlebot_actions::TurtlebotMoveGoal goal;
		goal.forward_distance = 0;
		goal.turn_distance = M_PI / 36;
		ac.sendGoal(goal);

		//wait for the action to return
		bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

		if (finished_before_timeout) {
			actionlib::SimpleClientGoalState state = ac.getState();
			ROS_INFO("Action finished: %s", state.toString().c_str());
		} else
			ROS_INFO("Action did not finish before the time out.");

		for (float angle = -M_PI / 4; angle <= M_PI / 4; angle += M_PI / 18) {

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

				if (map.get()) {
					keypoint_map map1(rgb, depth);
					map->merge_keypoint_map(map1);
				} else {
					map.reset(new keypoint_map(rgb, depth));
				}

				std::cerr << map->keypoints3d.size() << " "
						<< map->descriptors.rows << " " << map->weights.size()
						<< std::endl;

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

	vis.removeAllPointClouds();
	vis.addPointCloud<pcl::PointXYZ>(map->keypoints3d.makeShared(),
			"keypoints");
	vis.spin();

	map->remove_bad_points();

	vis.removeAllPointClouds();
	vis.addPointCloud<pcl::PointXYZ>(map->keypoints3d.makeShared(),
			"keypoints");
	vis.spin();

	map->optimize();

	vis.removeAllPointClouds();
	vis.addPointCloud<pcl::PointXYZ>(map->keypoints3d.makeShared(),
			"keypoints");
	vis.spin();

//exit
	return 0;
}

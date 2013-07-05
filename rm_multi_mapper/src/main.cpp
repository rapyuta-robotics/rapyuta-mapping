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

	cv::Ptr<cv::FeatureDetector> fd = new cv::SurfFeatureDetector;
	fd->setInt("hessianThreshold", 400);
	fd->setBool("extended", true);
	fd->setBool("upright", true);

	cv::Ptr<cv::DescriptorExtractor> de = new cv::SurfDescriptorExtractor;

	cv::Ptr<cv::DescriptorMatcher> dm = new cv::BFMatcher;

	pcl::PointCloud<pcl::PointXYZ> accumulated_keypoints3d;
	cv::Mat accumulated_descriptors;
	vector<float> accumulated_weights;

	pcl::visualization::PCLVisualizer vis;

	Eigen::Vector4f intrinsics;
	intrinsics << 525.0, 525.0, 319.5, 239.5;

	std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f> > camera_positions;
	std::vector<observation> observations;

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

	for (int i = 0; i < 2; i++) {
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

				if (accumulated_keypoints3d.size() == 0) {

					std::vector<cv::KeyPoint> keypoints;

					compute_features(rgb, depth, intrinsics, fd, de, keypoints,
							accumulated_keypoints3d, accumulated_descriptors);

					for (int keypoint_id = 0; keypoint_id < keypoints.size();
							keypoint_id++) {
						observation o;
						o.cam_id = 0;
						o.point_id = keypoint_id;
						o.coord << keypoints[keypoint_id].pt.x, keypoints[keypoint_id].pt.y;

						observations.push_back(o);
						accumulated_weights.push_back(1.0f);
					}

					camera_positions.push_back(Eigen::Affine3f::Identity());

				} else {

					std::vector<cv::KeyPoint> keypoints;
					pcl::PointCloud<pcl::PointXYZ> keypoints3d;
					cv::Mat descriptors;

					compute_features(rgb, depth, intrinsics, fd, de, keypoints,
							keypoints3d, descriptors);

					if (keypoints.size() < 3) {
						ROS_INFO("Not enough features... Skipping frame");
						continue;
					}

					std::vector<cv::DMatch> matches;
					dm->match(descriptors, accumulated_descriptors, matches);

					Eigen::Affine3f transform;
					std::vector<bool> inliers;

					bool res = estimate_transform_ransac(keypoints3d,
							accumulated_keypoints3d, matches, 3000, 0.03 * 0.03,
							20, transform, inliers);

					if(!res)
						continue;

					camera_positions.push_back(transform);

					for (int i = 0; i < matches.size(); i++) {

						observation o;
						o.cam_id = camera_positions.size() - 1;
						o.coord << keypoints[matches[i].queryIdx].pt.x, keypoints[matches[i].queryIdx].pt.y;

						if (inliers[i]) {
							o.point_id = matches[i].trainIdx;

							accumulated_descriptors.row(o.point_id) =
									(accumulated_descriptors.row(o.point_id)
											* accumulated_weights[o.point_id]
											+ descriptors.row(
													matches[i].queryIdx))
											/ (accumulated_weights[o.point_id]
													+ 1);

							accumulated_weights[o.point_id] += 1.0f;

						} else {

							pcl::PointXYZ p;
							p.getVector4fMap() = transform * keypoints3d[matches[i].queryIdx].getVector4fMap();

							accumulated_keypoints3d.push_back(p);

							o.point_id = accumulated_keypoints3d.size() - 1;

							accumulated_weights[o.point_id] += 1.0f;

							cv::vconcat(accumulated_descriptors,
									descriptors.row(matches[i].queryIdx),
									accumulated_descriptors);

							accumulated_weights.push_back(1.0f);

						}

						observations.push_back(o);

					}

				}

				std::cerr << accumulated_keypoints3d.size() << " "
						<< accumulated_descriptors.rows << " "
						<< accumulated_weights.size() << std::endl;

				vis.removeAllPointClouds();
				vis.addPointCloud<pcl::PointXYZ>(
						accumulated_keypoints3d.makeShared(), "keypoints");
				vis.spinOnce(2);

				//cv::imshow("Image", rgb);
				//cv::waitKey(2);

			} else {
				ROS_ERROR("Failed to call service /cloudbot1/capture");
				return 1;
			}

		}
	}

	vis.spin();

//exit
	return 0;
}

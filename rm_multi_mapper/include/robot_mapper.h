/*
 * robot_mapper.h
 *
 *  Created on: Jul 18, 2013
 *      Author: vsu
 */

#ifndef ROBOT_MAPPER_H_
#define ROBOT_MAPPER_H_

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <turtlebot_actions/TurtlebotMoveAction.h>
#include <rm_capture_server/Capture.h>
#include <pcl/common/transforms.h>
#include <std_msgs/Float32.h>
#include <eigen_conversions/eigen_msg.h>
#include <nav_msgs/OccupancyGrid.h>
#include <octomap/OcTree.h>
#include <std_srvs/Empty.h>
#include <rm_localization/SetMap.h>
#include <rm_localization/SetInitialPose.h>

#include <move_base_msgs/MoveBaseAction.h>
#include <octomap_msgs/BoundingBoxQuery.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <tf/transform_broadcaster.h>
#include <pcl_ros/publisher.h>
#include <tf_conversions/tf_eigen.h>
#include <octomap_server.h>

#include <keypoint_map.h>

class robot_mapper {

public:

	typedef boost::shared_ptr<robot_mapper> Ptr;

	robot_mapper(ros::NodeHandle & nh, const std::string & robot_prefix,
			const int robot_num);

	void capture_sphere();

	void set_map();
	void move_to_random_point();

	void merge(robot_mapper::Ptr & other);

	int robot_num;
	std::string prefix;

	bool merged;

	actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> move_base_action_client;
	ros::ServiceClient capture_client;
	ros::ServiceClient clear_unknown_space_client;
	ros::ServiceClient set_map_client;
	ros::ServiceClient set_intial_pose;
	ros::Publisher servo_pub;
	ros::Publisher pub_keypoints;
	ros::Publisher pub_cloud;

	boost::shared_ptr<keypoint_map> map;

	Eigen::Affine3f initial_transformation;

	RmOctomapServer::Ptr octomap_server;

};


#endif /* ROBOT_MAPPER_H_ */

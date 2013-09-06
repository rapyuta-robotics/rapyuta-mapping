/*
 * robot_mapper.h
 *
 *  Created on: Jul 18, 2013
 *      Author: vsu
 */

#ifndef ROBOT_MAPPER_H_
#define ROBOT_MAPPER_H_

#include <keyframe_map.h>
#include <sensor_msgs/SetCameraInfo.h>
#include <sensor_msgs/distortion_models.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <turtlebot_actions/TurtlebotMoveAction.h>
#include <std_msgs/Float32.h>
#include <std_srvs/Empty.h>
#include <rm_localization/UpdateMap.h>
#include <tf/transform_broadcaster.h>
#include <boost/thread.hpp>

class robot_mapper {

public:

	typedef boost::shared_ptr<robot_mapper> Ptr;

	robot_mapper(ros::NodeHandle & nh, const std::string & robot_prefix,
			const int robot_num);

	void move_straight(float distance);
	void full_rotation();
	void turn(float angle);
	void capture_sphere();
	void optmize_panorama();
	void optmize();
	void save_map(const std::string & dirname);

	bool merge(robot_mapper & other);

protected:

	void keyframeCallback(const rm_localization::Keyframe::ConstPtr& msg);
	void publish_tf();
	void update_map(bool with_intrinsics = false);
	void publish_empty_cloud();
	void publish_cloud();



	int robot_num;
	std::string prefix;
	bool merged;
	tf::Transform world_to_odom;

	actionlib::SimpleActionClient<turtlebot_actions::TurtlebotMoveAction> action_client;

	boost::shared_ptr<keyframe_map> map;
	ros::Publisher pointcloud_pub;
	ros::Publisher servo_pub;
	ros::Subscriber keyframe_sub;

	ros::ServiceClient update_map_service;
	ros::ServiceClient clear_keyframes_service;
	ros::ServiceClient set_camera_info_service;

	boost::mutex merge_mutex;


};

#endif /* ROBOT_MAPPER_H_ */

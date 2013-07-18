/*
 * test.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <keypoint_map.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <ros/ros.h>
#include <pcl_ros/publisher.h>

#include <tf/transform_broadcaster.h>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <rm_localization/SetMap.h>
#include <std_srvs/Empty.h>

int main(int argc, char **argv) {

	ros::init(argc, argv, "test_map");
	ros::NodeHandle nh;

	ros::Publisher pub_cloud =
			nh.advertise<pcl::PointCloud<pcl::PointXYZRGBA> >("/map_cloud", 10);

	ros::Publisher pub_keypoints =
			nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("keypoints", 10);

	ros::ServiceClient octomap_reset = nh.serviceClient<std_srvs::Empty>(
			"/octomap_server/reset");

	keypoint_map map("map_3");

	ros::ServiceClient client = nh.serviceClient<rm_localization::SetMap>(
			"/cloudbot2/set_map");

	rm_localization::SetMap data;

	pcl::toROSMsg(map.keypoints3d, data.request.keypoints3d);

	cv_bridge::CvImage desc;
	desc.image = map.descriptors;
	desc.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
	data.request.descriptors = *(desc.toImageMsg());

	client.call(data);

	std_srvs::Empty msg;
	octomap_reset.call(msg);
	map.publish_keypoints(pub_cloud);

	ros::Rate r(1);
	int seq = 0;
	while (ros::ok()) {
		map.keypoints3d.header.frame_id = "/map";
		map.keypoints3d.header.stamp = ros::Time::now();
		map.keypoints3d.header.seq = seq;

		pub_keypoints.publish(map.keypoints3d);

		seq++;
		r.sleep();
		ros::spinOnce();
	}

	return 0;

	//map.publish_keypoints(pub_cloud);

	return 0;
}

/*
 * test.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <util.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <ros/ros.h>
#include <pcl_ros/publisher.h>

#include <tf/transform_broadcaster.h>
#include <boost/thread.hpp>

void publish_transforms() {
	tf::TransformBroadcaster br;

	while (true) {
		tf::Transform transform;
		transform.setIdentity();

		br.sendTransform(
				tf::StampedTransform(transform, ros::Time::now(), "/map",
						"/cloudbot1/odom_combined"));

		usleep(33000);

	}

}

int main(int argc, char **argv) {

	ros::init(argc, argv, "test_map");
	ros::NodeHandle nh;

	octomap::OcTree tree(0.05);

	for (int x = -100; x <= 100; x++) {
		for (int y = -100; y <= 100; y++) {
			for (int z = -100; z <= 100; z++) {
				tree.updateNode(0.01 * x, 0.01 * y, 0.01 * z, false);
			}
		}
	}

	tree.prune();
	tree.writeBinary("free_space.bt");

	boost::thread t(publish_transforms);

	ros::spin();

	ros::Publisher pub = nh.advertise<nav_msgs::OccupancyGrid>("/map", 1);
	ros::Publisher pub_cloud =
			nh.advertise<pcl::PointCloud<pcl::PointXYZRGBA> >("/map_cloud", 10);
	keypoint_map map("map_2");

	/*
	 map.remove_bad_points(3);

	 pcl::visualization::PCLVisualizer vis;
	 vis.removeAllPointClouds();
	 vis.addPointCloud<pcl::PointXYZ>(map.keypoints3d.makeShared(), "keypoints");
	 vis.spin();

	 std::cerr << "Error " << map.compute_error() << " Mean error "
	 << map.compute_error() / map.observations.size() << std::endl;

	 for (int i = 0; i < 1; i++) {
	 map.optimize();

	 std::cerr << "Error " << map.compute_error() << " Mean error "
	 << map.compute_error() / map.observations.size() << std::endl;

	 vis.removeAllPointClouds();
	 vis.addPointCloud<pcl::PointXYZ>(map.keypoints3d.makeShared(),
	 "keypoints");
	 vis.spin();
	 }

	 map.extract_surface();
	 */

	map.publish_keypoints(pub_cloud);

	return 0;
}

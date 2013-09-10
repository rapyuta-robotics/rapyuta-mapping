/*
 * splitter.cpp
 *
 *  Created on: Sept 10, 2013
 *      Author: mayanks43
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>

#include <keyframe_map.h>

#include "ros/ros.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Int32.h"
#include "rm_multi_mapper/Range.h"

int main(int argc, char **argv) {

	int size = frames.size();

	tbb::concurrent_vector<std::pair<int, int> > overlaping_keyframes;

    ros::init(argc, argv, "splitter");
	ros::NodeHandle n;
	ros::Publisher reducer1 = n.advertise<std_msgs::Int32>("reducer1", 1000);
	ros::Publisher reducer2 = n.advertise<std_msgs::Int32>("reducer2", 1000);
	
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (i != j) {
				float angle =
						frames[i]->get_pos().unit_quaternion().angularDistance(
								frames[j]->get_pos().unit_quaternion());

				if (angle < M_PI / 6) {
					overlaping_keyframes.push_back(std::make_pair(i, j));
					//ROS_INFO("Images %d and %d intersect with angular distance %f", i, j, angle*180/M_PI);
				}
			}
		}
	}
}

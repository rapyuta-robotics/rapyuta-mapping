/*
 * reduce_jacobian_ros.h
 *
 *  Created on: Aug 10, 2013
 *      Author: vsu
 */

#ifndef REDUCE_JACOBIAN_ROS_H_
#define REDUCE_JACOBIAN_ROS_H_

#include <color_keyframe.h>

struct reduce_jacobian_ros {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;
	int size;
	int subsample_level;

	//std::vector<color_keyframe::Ptr> & frames;

	reduce_jacobian_ros();

};


#endif /* REDUCE_JACOBIAN_ROS_H_ */

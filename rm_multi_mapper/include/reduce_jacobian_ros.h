/*
 * reduce_jacobian_ros.h
 *
 *  Created on: Aug 10, 2013
 *      Author: vsu
 */

#ifndef REDUCE_JACOBIAN_ROS_H_
#define REDUCE_JACOBIAN_ROS_H_

#include <color_keyframe.h>
#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include "rm_multi_mapper/WorkerAction.h"

struct reduce_jacobian_ros {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;
	int size;
	int subsample_level;

	std::vector<color_keyframe::Ptr> & frames;

    reduce_jacobian_ros(std::vector<color_keyframe::Ptr> & frames,
			int size, int subsample_level);

	void compute_frame_jacobian(const Eigen::Vector3f & i,
			const Eigen::Matrix3f & Rwi, const Eigen::Matrix3f & Rwj,
			Eigen::Matrix<float, 9, 3> & Ji, Eigen::Matrix<float, 9, 3> & Jj, Eigen::Matrix<float, 9, 3> & Jk);

	void reduce(const rm_multi_mapper::WorkerGoalConstPtr &goal);

};


#endif /* REDUCE_JACOBIAN_ROS_H_ */

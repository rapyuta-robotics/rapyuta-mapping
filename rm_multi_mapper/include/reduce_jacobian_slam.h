/*
 * reduce_jacobian_slam.h
 *
 *  Created on: Aug 10, 2013
 *      Author: vsu
 */

#ifndef REDUCE_JACOBIAN_SLAM_H_
#define REDUCE_JACOBIAN_SLAM_H_

#include <color_keyframe.h>
#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include "rm_multi_mapper/WorkerSlamAction.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>

struct reduce_jacobian_slam {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;
	int size;

	typedef pcl::registration::TransformationEstimationPointToPlane<
			pcl::PointNormal, pcl::PointNormal> PointToPlane;
	pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;

	std::vector<color_keyframe::Ptr> & frames;

	reduce_jacobian_slam(std::vector<color_keyframe::Ptr> & frames, int size);

	void compute_frame_jacobian(const Eigen::Matrix4f & Mwi,
			const Eigen::Matrix4f & Miw, Eigen::Matrix<float, 6, 6> & Ji);

	void compute_floor_jacobian(float nx, float ny, float nz, float y, float x,
			Eigen::Matrix<float, 3, 6> & Ji);

	void add_icp_measurement(int i, int j);
	void add_rgbd_measurement(int i, int j);
	void add_floor_measurement(int i);

	void reduce(const rm_multi_mapper::WorkerSlamGoalConstPtr &goal);

};

#endif /* REDUCE_JACOBIAN_SLAM_H_ */

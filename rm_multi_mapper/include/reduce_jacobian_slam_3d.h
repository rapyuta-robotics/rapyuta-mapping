/*
 * reduce_jacobian_rgb.h
 *
 *  Created on: Aug 10, 2013
 *      Author: vsu
 */

#ifndef REDUCE_JACOBIAN_SLAM_3D_H_
#define REDUCE_JACOBIAN_SLAM_3D_H_

#include <color_keyframe.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>

struct reduce_jacobian_slam_3d {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;
	int size;

	typedef pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal> PointToPlane;
	pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;

	tbb::concurrent_vector<color_keyframe::Ptr> & frames;

	reduce_jacobian_slam_3d(tbb::concurrent_vector<color_keyframe::Ptr> & frames, int size);

	reduce_jacobian_slam_3d(reduce_jacobian_slam_3d & rb, tbb::split);

	void compute_frame_jacobian(
			const Eigen::Matrix4f & Mwi,
			const Eigen::Matrix4f & Miw,
			Eigen::Matrix<float, 6, 6> & Ji);

	void add_icp_measurement(int i, int j);
	void add_rgbd_measurement(int i, int j);
	void add_floor_measurement(int i);

	void operator()(
			const tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>& r);

	void join(reduce_jacobian_slam_3d& rb);

};

#endif /* REDUCE_JACOBIAN_SLAM_3D_H_ */

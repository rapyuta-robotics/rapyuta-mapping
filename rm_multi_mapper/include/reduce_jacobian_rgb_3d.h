/*
 * reduce_jacobian_rgb.h
 *
 *  Created on: Aug 10, 2013
 *      Author: vsu
 */

#ifndef REDUCE_JACOBIAN_RGB_3D_H_
#define REDUCE_JACOBIAN_RGB_3D_H_

#include <keyframe.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct reduce_jacobian_rgb_3d {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;
	int size;
	int intrinsics_size;
	int subsample_level;

	tbb::concurrent_vector<keyframe::Ptr> & frames;
	std::vector<Eigen::Vector3f> & intrinsics_vector;


	reduce_jacobian_rgb_3d(tbb::concurrent_vector<keyframe::Ptr> & frames, std::vector<Eigen::Vector3f> & intrinsics_vector,
			int size, int intrinsics_size, int subsample_level);

	reduce_jacobian_rgb_3d(reduce_jacobian_rgb_3d & rb, tbb::split);

	void compute_frame_jacobian(const Eigen::Vector3f & i,
			const Eigen::Matrix4f & Miw, const Eigen::Matrix4f & Mwj,
			Eigen::Matrix<float, 12, 6> & Ji, Eigen::Matrix<float, 12, 6> & Jj,
			Eigen::Matrix<float, 12, 3> & Jk);

	void warpImage(int i, int j, cv::Mat & intensity_i,
			cv::Mat & intensity_j, cv::Mat & intensity_j_warped, cv::Mat & idx_j_warped);

	void operator()(
			const tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>& r);

	void join(reduce_jacobian_rgb_3d& rb);

};


#endif /* REDUCE_JACOBIAN_RGB_3D_H_ */

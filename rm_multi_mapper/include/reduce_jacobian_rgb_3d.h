/*
 * reduce_jacobian_rgb.h
 *
 *  Created on: Aug 10, 2013
 *      Author: vsu
 */

#ifndef REDUCE_JACOBIAN_RGB_3D_H_
#define REDUCE_JACOBIAN_RGB_3D_H_

#include <color_keyframe.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct reduce_jacobian_rgb_3d {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;
	int size;
	int subsample_level;

	tbb::concurrent_vector<color_keyframe::Ptr> & frames;

	reduce_jacobian_rgb_3d(tbb::concurrent_vector<color_keyframe::Ptr> & frames, int size, int subsample_level);

	reduce_jacobian_rgb_3d(reduce_jacobian_rgb_3d & rb, tbb::split);

	void compute_frame_jacobian(const Eigen::Vector3f & i,
			const Eigen::Vector4f & p, const Eigen::Matrix<float, 4, 4, Eigen::ColMajor> & Miw,
			Eigen::Matrix<float, 2, 6> & J);

	void operator()(
			const tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>& r);

	void join(reduce_jacobian_rgb_3d& rb);

};

#endif /* REDUCE_JACOBIAN_RGB_3D_H_ */

/*
 * reduce_jacobian_rgb.h
 *
 *  Created on: Aug 10, 2013
 *      Author: vsu
 */

#ifndef REDUCE_JACOBIAN_RGB_H_
#define REDUCE_JACOBIAN_RGB_H_

#include <color_keyframe.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>

struct reduce_jacobian_rgb {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;
	int size;
	int subsample_level;

	tbb::concurrent_vector<color_keyframe::Ptr> & frames;

	reduce_jacobian_rgb(tbb::concurrent_vector<color_keyframe::Ptr> & frames,
			int size, int subsample_level);

	reduce_jacobian_rgb(reduce_jacobian_rgb& rb, tbb::split);

	void compute_frame_jacobian(const Eigen::Vector3f & i,
			const Eigen::Matrix3f & Rwi, const Eigen::Matrix3f & Rwj,
			Eigen::Matrix<float, 9, 3> & Ji, Eigen::Matrix<float, 9, 3> & Jj,
			Eigen::Matrix<float, 9, 3> & Jk);

	void operator()(
			const tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>& r);

	void join(reduce_jacobian_rgb& rb);

};

#endif /* REDUCE_JACOBIAN_RGB_H_ */

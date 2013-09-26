/*
 * reduce_jacobian.h
 *
 *  Created on: Sep 26, 2013
 *      Author: vsu
 */

#ifndef REDUCE_JACOBIAN_H_
#define REDUCE_JACOBIAN_H_

#include <sophus/se3.hpp>
#include <tbb/parallel_for.h>

struct reduce_jacobian {

	Sophus::Matrix6f JtJ;
	Sophus::Vector6f Jte;
	int num_points;
	float error_sum;

	const uint8_t * intencity;
	const int16_t * intencity_dx;
	const int16_t * intencity_dy;
	const float * intencity_warped;
	const float * depth_warped;
	const Eigen::Vector3f & intrinsics;
	const Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> & cloud;
	int cols;
	int rows;

	reduce_jacobian(const uint8_t * intencity, const int16_t * intencity_dx,
			const int16_t * intencity_dy, const float * intencity_warped,
			const float * depth_warped, const Eigen::Vector3f & intrinsics,
			const Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> & cloud,
			int cols, int rows);

	reduce_jacobian(reduce_jacobian & rb, tbb::split);

	void compute_jacobian(const Eigen::Vector4f & p, Eigen::Matrix<float, 2, 6> & J);

	void operator()(const tbb::blocked_range<int>& range);

	void join(reduce_jacobian& rb);

};



#endif /* REDUCE_JACOBIAN_H_ */

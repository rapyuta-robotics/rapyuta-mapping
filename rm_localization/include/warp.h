/*
 * warp.h
 *
 *  Created on: Sep 26, 2013
 *      Author: vsu
 */

#ifndef WARP_H_
#define WARP_H_

#include <tbb/blocked_range.h>

struct parallel_warp {
	const uint8_t * intencity;
	const uint16_t * depth;
	const Eigen::Matrix<float, 4, 4, Eigen::ColMajor> & transform;
	const Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> & cloud;
	const Eigen::Vector3f & intrinsics;
	int cols;
	int rows;
	float * intencity_warped;
	float * depth_warped;

	parallel_warp(const uint8_t * intencity, const uint16_t * depth,
			const Eigen::Matrix<float, 4, 4, Eigen::ColMajor> & transform,
			const Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> & cloud,
			const Eigen::Vector3f & intrinsics, int cols, int rows,
			float * intencity_warped, float * depth_warped) :
			intencity(intencity), depth(depth), transform(transform), cloud(
					cloud), intrinsics(intrinsics), cols(cols), rows(rows), intencity_warped(
					intencity_warped), depth_warped(depth_warped) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); i++) {

			Eigen::Vector4f p = cloud.col(i);
			if (p(3) > 0) {
				p = transform * p;

				float uw = p(0) * intrinsics[0] / p(2) + intrinsics[1];
				float vw = p(1) * intrinsics[0] / p(2) + intrinsics[2];

				if (uw >= 0 && uw < cols && vw >= 0 && vw < rows) {

					float val = interpolate(uw, vw, p(2));
					if (val > 0) {
						intencity_warped[i] = val;
						depth_warped[i] = p(2) * 1000;
					} else {
						intencity_warped[i] = 0;
						depth_warped[i] = 0;
					}

				} else {
					intencity_warped[i] = 0;
					depth_warped[i] = 0;
				}

			} else {
				intencity_warped[i] = 0;
				depth_warped[i] = 0;

			}

		}

	}

	float interpolate(float uw, float vw, float z) const {

		int u = uw;
		int v = vw;

		float u0 = uw - u;
		float v0 = vw - v;
		float u1 = 1 - u0;
		float v1 = 1 - v0;
		uint16_t z_p_eps = z * 1000 - 50;

		float val = 0;
		float sum = 0;

		size_t p00 = v * cols + u;
		if (depth[p00] != 0 && depth[p00] > z_p_eps) {
			val += u0 * v0 * intencity[p00];
			sum += u0 * v0;
		}

		size_t p01 = p00 + cols;
		if (depth[p01] != 0 && depth[p01] > z_p_eps) {
			val += u0 * v1 * intencity[p01];
			sum += u0 * v1;
		}

		size_t p10 = p00 + 1;
		if (depth[p10] != 0 && depth[p10] > z_p_eps) {
			val += u1 * v0 * intencity[p10];
			sum += u1 * v0;
		}

		size_t p11 = p01 + 1;
		if (depth[p11] != 0 && depth[p11] > z_p_eps) {
			val += u1 * v1 * intencity[p11];
			sum += u1 * v1;
		}

		return val / sum;

	}

};



#endif /* WARP_H_ */

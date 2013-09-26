/*
 * convert_depth_to_cloud.h
 *
 *  Created on: Sep 26, 2013
 *      Author: vsu
 */

#ifndef CONVERT_DEPTH_TO_CLOUD_H_
#define CONVERT_DEPTH_TO_CLOUD_H_

struct convert_depth_to_pointcloud {
	const uint8_t * intencity;
	const uint16_t * depth;
	const Eigen::Vector3f & intrinsics;
	int cols;
	int rows;
	Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> & cloud;
	int16_t * intencity_dx;
	int16_t * intencity_dy;

	convert_depth_to_pointcloud(const uint8_t * intencity,
			const uint16_t * depth, const Eigen::Vector3f & intrinsics,
			int cols, int rows,
			Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> & cloud,
			int16_t * intencity_dx, int16_t * intencity_dy) :
			intencity(intencity), depth(depth), intrinsics(intrinsics), cols(
					cols), rows(rows), cloud(cloud), intencity_dx(intencity_dx), intencity_dy(
					intencity_dy) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); i++) {
			int u = i % cols;
			int v = i / cols;

			int16_t dx, dy;
			if (u == 0) {
				dx = (int16_t) intencity[i + 1] - intencity[i];
			} else if (u == cols - 1) {
				dx = (int16_t) intencity[i] - intencity[i - 1];
			} else {
				dx = ((int16_t) intencity[i + 1] - intencity[i - 1]) / 2;
			}

			if (v == 0) {
				dy = (int16_t) intencity[i + cols] - intencity[i];
			} else if (v == rows - 1) {
				dy = (int16_t) intencity[i] - intencity[i - cols];
			} else {
				dy = ((int16_t) intencity[i + cols] - intencity[i - cols]) / 2;
			}

			intencity_dx[i] = dx;
			intencity_dy[i] = dy;

			Eigen::Vector4f p;
			p(2) = depth[i] / 1000.0;

			if (p(2) > 0 /* && (std::abs(dx) > 12 || std::abs(dy) > 12) */) {
				p(0) = (u - intrinsics[1]) * p(2) / intrinsics[0];
				p(1) = (v - intrinsics[2]) * p(2) / intrinsics[0];
				p(3) = 1.0f;

			} else {
				p(0) = p(1) = p(2) = p(3) = 0.0f;
			}

			cloud.col(i) = p;

		}

	}
};



#endif /* CONVERT_DEPTH_TO_CLOUD_H_ */

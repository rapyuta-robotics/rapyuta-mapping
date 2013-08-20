#ifndef FRAME_H_
#define FRAME_H_

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <tbb/parallel_for.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

struct convert {
	const uint8_t * yuv;
	uint8_t * intencity;

	convert(const uint8_t * yuv, uint8_t * intencity) :
			yuv(yuv), intencity(intencity) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); i++) {
			intencity[i] = yuv[2 * i + 1];
		}

	}

};

struct subsample {
	const uint8_t * prev_intencity;
	const uint16_t * prev_depth;
	int cols;
	int rows;
	uint8_t * current_intencity;
	uint16_t * current_depth;

	subsample(const uint8_t * prev_intencity, const uint16_t * prev_depth,
			int cols, int rows, uint8_t * current_intencity,
			uint16_t * current_depth) :
			prev_intencity(prev_intencity), prev_depth(prev_depth), cols(cols), rows(
					rows), current_intencity(current_intencity), current_depth(
					current_depth) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); i++) {
			int u = i % cols;
			int v = i / cols;

			int p1 = 4 * v * cols + 2 * u;
			int p2 = p1 + 2 * cols;
			int p3 = p1 + 1;
			int p4 = p2 + 1;

			int val = prev_intencity[p1];
			val += prev_intencity[p2];
			val += prev_intencity[p3];
			val += prev_intencity[p4];

			current_intencity[i] = val / 4;

			uint16_t values[4];
			values[0] = prev_depth[p1];
			values[1] = prev_depth[p2];
			values[2] = prev_depth[p3];
			values[3] = prev_depth[p4];
			std::sort(values, values + 4);

			current_depth[i] = values[2];

		}

	}

};

struct parallel_warp {
	const uint8_t * intencity;
	const uint16_t * depth;
	const Eigen::Affine3f & transform;
	const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud;
	const Eigen::Vector3f & intrinsics;
	int cols;
	int rows;
	uint8_t * intencity_warped;
	uint16_t * depth_warped;

	parallel_warp(const uint8_t * intencity, const uint16_t * depth,
			const Eigen::Affine3f & transform,
			const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
			const Eigen::Vector3f & intrinsics, int cols, int rows,
			uint8_t * intencity_warped, uint16_t * depth_warped) :
			intencity(intencity), depth(depth), transform(transform), cloud(
					cloud), intrinsics(intrinsics), cols(cols), rows(rows), intencity_warped(
					intencity_warped), depth_warped(depth_warped) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); i++) {
			int u = i % cols;
			int v = i / cols;

			pcl::PointXYZRGB p = cloud->at(u, v);
			if (std::isfinite(p.x) && std::isfinite(p.y)
					&& std::isfinite(p.z)) {
				p.getVector3fMap() = transform * p.getVector3fMap();

				float uw = p.x * intrinsics[0] / p.z + intrinsics[1];
				float vw = p.y * intrinsics[0] / p.z + intrinsics[2];

				if (uw >= 0 && uw < cols && vw >= 0 && vw < rows) {

					float val = interpolate(uw, vw, p.z);
					if (val > 0) {
						intencity_warped[i] = val;
						depth_warped[i] = p.z * 1000;
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
		uint16_t z_eps = z*1000 - 100;

		float val = 0;
		float sum = 0;

		size_t p00 = v * cols + u;
		if (depth[p00] != 0 && depth[p00] > z_eps) {
			val += u0 * v0 * intencity[p00];
			sum += u0 * v0;
		}

		size_t p01 = p00 + cols;
		if (depth[p01] != 0 && depth[p01] > z_eps) {
			val += u0 * v1 * intencity[p01];
			sum += u0 * v1;
		}

		size_t p10 = p00 + 1;
		if (depth[p10] != 0 && depth[p10] > z_eps) {
			val += u1 * v0 * intencity[p10];
			sum += u1 * v0;
		}

		size_t p11 = p01 + 1;
		if (depth[p11] != 0 && depth[p11] > z_eps) {
			val += u1 * v1 * intencity[p11];
			sum += u1 * v1;
		}

		return val / sum;

	}

};

class frame {

public:

	typedef boost::shared_ptr<frame> Ptr;

	frame(const cv::Mat & yuv, const cv::Mat & depth,
			const Sophus::SE3f & position, int max_level = 3);

	~frame();

	void warp(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
			const Eigen::Vector3f & intrinsics, const Sophus::SE3f & position,
			int level, cv::Mat & intencity_warped, cv::Mat & depth_warped);

	inline cv::Mat get_i(int level) {
		return cv::Mat(rows / (1 << level), cols / (1 << level), CV_8U,
				intencity_pyr[level]);
	}

	inline cv::Mat get_d(int level) {
		return cv::Mat(rows / (1 << level), cols / (1 << level), CV_16U,
				depth_pyr[level]);
	}

	inline Sophus::SE3f get_pos() {
		return position;
	}

protected:

	uint8_t ** intencity_pyr;
	uint16_t ** depth_pyr;
	Sophus::SE3f position;

	int max_level;
	int cols;
	int rows;

	friend class keyframe;

};

#endif /* KEYFRAME_H_ */

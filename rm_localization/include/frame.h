#ifndef FRAME_H_
#define FRAME_H_

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <tbb/parallel_for.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

struct convert {
	const cv::Mat & yuv;
	const cv::Mat & depth;
	cv::Mat & intencity;
	cv::Mat & depth_f;

	convert(const cv::Mat & yuv, const cv::Mat & depth, cv::Mat & intencity,
			cv::Mat & depth_f) :
			yuv(yuv), depth(depth), intencity(intencity), depth_f(depth_f) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); i++) {
			int u = i % yuv.cols;
			int v = i / yuv.cols;

			cv::Vec2b val = yuv.at<cv::Vec2b>(v, u);
			intencity.at<float>(v, u) = val[1] / 255.0f;
			depth_f.at<float>(v, u) = depth.at<unsigned short>(v, u) / 1000.0f;
		}

	}

};

struct subsample {
	const cv::Mat & prev_intencity;
	const cv::Mat & prev_depth;
	cv::Mat & current_intencity;
	cv::Mat & current_depth;

	subsample(const cv::Mat & prev_intencity, const cv::Mat & prev_depth,
			cv::Mat & current_intencity, cv::Mat & current_depth) :
			prev_intencity(prev_intencity), prev_depth(prev_depth), current_intencity(
					current_intencity), current_depth(current_depth) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); i++) {
			int u = i % current_intencity.cols;
			int v = i / current_intencity.cols;

			float val = prev_intencity.at<float>(2 * v, 2 * u);
			val += prev_intencity.at<float>(2 * v + 1, 2 * u);
			val += prev_intencity.at<float>(2 * v, 2 * u + 1);
			val += prev_intencity.at<float>(2 * v + 1, 2 * u + 1);

			current_intencity.at<float>(v, u) = val / 4.0f;

			float values[4];
			values[0] = prev_depth.at<float>(2 * v, 2 * u);
			values[1] = prev_depth.at<float>(2 * v + 1, 2 * u);
			values[2] = prev_depth.at<float>(2 * v, 2 * u + 1);
			values[3] = prev_depth.at<float>(2 * v + 1, 2 * u + 1);
			std::sort(values, values + 4);

			current_depth.at<float>(v, u) = values[2];

		}

	}

};

struct parallel_warp {
	const cv::Mat & intencity;
	const cv::Mat & depth;
	const Sophus::SE3f & transform;
	const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud;
	const Eigen::Vector3f & intrinsics;
	cv::Mat & intencity_warped;
	cv::Mat & depth_warped;

	parallel_warp(const cv::Mat & intencity, const cv::Mat & depth,
			const Sophus::SE3f & transform,
			const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
			const Eigen::Vector3f & intrinsics, cv::Mat & intencity_warped,
			cv::Mat & depth_warped) :
			intencity(intencity), depth(depth), transform(transform), cloud(
					cloud), intrinsics(intrinsics), intencity_warped(
					intencity_warped), depth_warped(depth_warped) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		for (int i = range.begin(); i != range.end(); i++) {
			int u = i % intencity_warped.cols;
			int v = i / intencity_warped.cols;

			pcl::PointXYZ p = cloud->at(u, v);
			if (std::isfinite(p.x) && std::isfinite(p.y)
					&& std::isfinite(p.z)) {
				p.getVector3fMap() = transform * p.getVector3fMap();

				float uw = p.x * intrinsics[0] / p.z + intrinsics[1];
				float vw = p.y * intrinsics[0] / p.z + intrinsics[2];

				if (uw >= 0 && uw < intencity_warped.cols && vw >= 0
						&& vw < intencity_warped.rows) {

					float val = interpolate(uw, vw, p.z);
					if (val > 0) {
						intencity_warped.at<float>(v, u) = val;
						depth_warped.at<float>(v, u) = p.z;
					} else {
						intencity_warped.at<float>(v, u) = 0;
						depth_warped.at<float>(v, u) = 0;
					}

				} else {
					intencity_warped.at<float>(v, u) = 0;
					depth_warped.at<float>(v, u) = 0;
				}

			} else {
				intencity_warped.at<float>(v, u) = 0;
				depth_warped.at<float>(v, u) = 0;

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
		float z_eps = z - 0.05;

		float val = 0;
		float sum = 0;

		if (depth.at<float>(v, u) != 0 && depth.at<float>(v, u) > z_eps) {
			val += u0 * v0 * intencity.at<float>(v, u);
			sum += u0 * v0;
		}

		if (depth.at<float>(v + 1, u) != 0
				&& depth.at<float>(v + 1, u) > z_eps) {
			val += u0 * v1 * intencity.at<float>(v + 1, u);
			sum += u0 * v1;
		}

		if (depth.at<float>(v, u + 1) != 0
				&& depth.at<float>(v, u + 1) > z_eps) {
			val += u1 * v0 * intencity.at<float>(v, u + 1);
			sum += u1 * v0;
		}

		if (depth.at<float>(v + 1, u + 1) != 0
				&& depth.at<float>(v + 1, u + 1) > z_eps) {
			val += u1 * v1 * intencity.at<float>(v + 1, u + 1);
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

	void warp(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
			const Eigen::Vector3f & intrinsics, const Sophus::SE3f & position,
			int level, cv::Mat & intencity_warped, cv::Mat & depth_warped);

	inline cv::Mat get_i(int level) {
		return get_subsampled(intencity_pyr, level);
	}

	inline cv::Mat get_d(int level) {
		return get_subsampled(depth_pyr, level);
	}

	inline Sophus::SE3f get_pos() {
		return position;
	}

protected:

	cv::Mat get_subsampled(cv::Mat & image_pyr, int level) const;

	cv::Mat intencity_pyr;
	cv::Mat depth_pyr;
	Sophus::SE3f position;

	int max_level;
	int cols;
	int rows;

	friend class keyframe;

};

#endif /* KEYFRAME_H_ */

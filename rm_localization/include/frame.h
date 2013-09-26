#ifndef FRAME_H_
#define FRAME_H_

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <convert.h>
#include <subsample.h>
#include <warp.h>

class frame {

public:

	typedef boost::shared_ptr<frame> Ptr;

	frame(const cv::Mat & yuv, const cv::Mat & depth,
			const Sophus::SE3f & position, const Eigen::Vector3f & intrinsics,
			int max_level = 3);

	~frame();

	void warp(
			const Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> & cloud,
			const Sophus::SE3f & position, int level,
			cv::Mat & intencity_warped, cv::Mat & depth_warped);

	inline cv::Mat get_i(int level) {
		return cv::Mat(rows / (1 << level), cols / (1 << level), CV_8U,
				intencity_pyr[level]);
	}

	inline cv::Mat get_d(int level) {
		return cv::Mat(rows / (1 << level), cols / (1 << level), CV_16U,
				depth_pyr[level]);
	}

	inline Sophus::SE3f & get_pos() {
		return position;
	}

	inline Eigen::Vector3f get_intrinsics(int level) {
		return intrinsics / (1 << level);
	}

	inline Eigen::Vector3f & get_intrinsics() {
		return intrinsics;
	}

protected:

	uint8_t ** intencity_pyr;
	uint16_t ** depth_pyr;
	Sophus::SE3f position;
	Eigen::Vector3f intrinsics;

	int max_level;
	int cols;
	int rows;

	friend class keyframe;

};

#endif /* KEYFRAME_H_ */

#ifndef KEYFRAME_H_
#define KEYFRAME_H_

#include <frame.h>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <cv_bridge/cv_bridge.h>
#include <rm_localization/Keyframe.h>

#include <convert_depth_to_cloud.h>
#include <reduce_jacobian.h>

#include <ros/time.h>

class keyframe: public frame {

public:

	typedef boost::shared_ptr<keyframe> Ptr;

	keyframe(const cv::Mat & yuv, const cv::Mat & depth,
			const Sophus::SE3f & position, const Eigen::Vector3f & intrinsics,
			int max_level = 3);

	~keyframe();

	bool estimate_position(frame & f);
	bool estimate_relative_position(frame & f, Sophus::SE3f & Mrc);

	void update_intrinsics(const Eigen::Vector3f & intrinsics);

	void set_timestamp(ros::Time stamp);
	inline ros::Time get_timestamp() {
		return timestamp;
	}

	inline cv::Mat get_i_dx(int level) {
		return cv::Mat(rows / (1 << level), cols / (1 << level), CV_16S,
				intencity_pyr_dx[level]);
	}

	inline cv::Mat get_i_dy(int level) {
		return cv::Mat(rows / (1 << level), cols / (1 << level), CV_16S,
				intencity_pyr_dy[level]);
	}

	rm_localization::Keyframe::Ptr to_msg(
			const cv_bridge::CvImageConstPtr & yuv2, int idx);

	inline long int get_id() {
		return id;
	}

	inline void set_id(long int id) {
		this->id = id;
	}

protected:

	long int id;
	ros::Time timestamp;

	int16_t ** intencity_pyr_dx;
	int16_t ** intencity_pyr_dy;

	std::vector<Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> > clouds;

};

#endif /* KEYFRAME_H_ */

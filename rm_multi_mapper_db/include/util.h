
#ifndef UTIL_H
#define UTIL_H

#include <opencv2/core/core.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <keyframe_map.h>

class util {
public:

	typedef boost::shared_ptr<util> Ptr;

	util() {}
	virtual ~util(){}

	struct measurement {
		long first;
		long second;
		Sophus::SE3f transform;
		std::string type;
	};

	struct position {
		long idx;
		Sophus::SE3f transform;
	};

	virtual int get_new_robot_id() = 0;
	virtual void add_keyframe(int robot_id, const color_keyframe::Ptr & k) = 0;
	virtual void add_measurement(long first, long second,
			const Sophus::SE3f & transform, const std::string & type) = 0;

	virtual void add_keypoints(const color_keyframe::Ptr & k) = 0;
	virtual void get_keypoints(long frame_id,
			pcl::PointCloud<pcl::PointXYZ> & keypoints3d,
			cv::Mat & desctriptors) = 0;

	virtual color_keyframe::Ptr get_keyframe(long frame_id) = 0;

	virtual boost::shared_ptr<keyframe_map> get_robot_map(int robot_id) = 0;

	virtual void get_overlapping_pairs(int map_id,
			std::vector<std::pair<long, long> > & overlapping_keyframes) = 0;

	virtual void load_measurements(long keyframe_id,
			std::vector<measurement> & m) = 0;
	virtual void load_positions(int map_id, std::vector<position> & p) = 0;
	virtual void update_position(const position & p) = 0;

	virtual void compute_features(const cv::Mat & rgb, const cv::Mat & depth,
			const Eigen::Vector3f & intrinsics,
			std::vector<cv::KeyPoint> & filtered_keypoints,
			pcl::PointCloud<pcl::PointXYZ> & keypoints3d,
			cv::Mat & descriptors) = 0;
};

#endif

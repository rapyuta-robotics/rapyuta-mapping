#ifndef COLOR_KEYFRAME_H_
#define COLOR_KEYFRAME_H_

#include <keyframe.h>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <rm_localization/Keyframe.h>

class color_keyframe: public keyframe {

public:

	typedef boost::shared_ptr<color_keyframe> Ptr;

	color_keyframe(const cv::Mat & rgb, const cv::Mat & gray,
			const cv::Mat & depth, const Sophus::SE3f & position,
			const Eigen::Vector3f & intrinsics, int max_level = 3);

	pcl::PointCloud<pcl::PointXYZ>::Ptr get_pointcloud(int subsample = 1,
			bool transformed = true,
			float min_height = std::numeric_limits<float>::min(),
			float max_height = std::numeric_limits<float>::max()) const;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_colored_pointcloud(
			int subsample = 1) const;
	pcl::PointCloud<pcl::PointNormal>::Ptr get_pointcloud_with_normals(
			int subsample = 1, bool transformed = true) const;

	inline cv::Mat get_rgb() {
		return rgb;
	}

	inline Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> & get_cloud(
			int level) {
		return clouds[level];
	}

	inline Eigen::Vector3f get_centroid() {
		return position * centroid;
	}

	static Ptr from_msg(const rm_localization::Keyframe::ConstPtr & k);

protected:
	cv::Mat rgb;
	Eigen::Vector3f centroid;

};

#endif /* COLOR_KEYFRAME_H_ */

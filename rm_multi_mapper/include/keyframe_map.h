#ifndef KEYFRAME_MAP_H_
#define KEYFRAME_MAP_H_

#include <opencv2/core/core.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <color_keyframe.h>

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>

#include <rm_localization/Keyframe.h>

class keyframe_map {
public:

	keyframe_map();

	void add_frame(const rm_localization::Keyframe::ConstPtr & k);
	float optimize_panorama(int level);
	float optimize_slam(int skip_n = 1);
	void align_z_axis();

	//void optimize_g2o();

	cv::Mat get_panorama_image();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_map_pointcloud();

	bool find_transform(const keyframe_map & other, Sophus::SE3f & t) const;
	void merge(keyframe_map & other, const Sophus::SE3f & t);

	void save(const std::string & dir_name);
	void load(const std::string & dir_name);

	tbb::concurrent_vector<color_keyframe::Ptr> frames;
	tbb::concurrent_vector<int> idx;
};

#endif /* KEYFRAME_MAP_H_ */

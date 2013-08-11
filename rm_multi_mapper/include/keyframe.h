/*
 * keyframe.h
 *
 *  Created on: Aug 10, 2013
 *      Author: vsu
 */

#ifndef KEYFRAME_H_
#define KEYFRAME_H_

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sophus/se3.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

class keyframe {

public:

	typedef boost::shared_ptr<keyframe> Ptr;

	keyframe(const cv::Mat & rgb, const cv::Mat & depth,
			const Sophus::SE3f & position, std::vector<Eigen::Vector3f> & intrinsics_vector, int intrinsics_idx);

	Eigen::Vector3f get_centroid() const;
	pcl::PointCloud<pcl::PointXYZ>::Ptr get_pointcloud() const;
	pcl::PointCloud<pcl::PointXYZ>::Ptr get_pointcloud(float min_height,
			float max_height) const;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_colored_pointcloud() const;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_colored_pointcloud(
			float min_height, float max_height) const;

	pcl::PointCloud<pcl::PointXYZ>::Ptr get_original_pointcloud() const;
	pcl::PointCloud<pcl::PointNormal>::Ptr get_original_pointcloud_with_normals() const;
	pcl::PointCloud<pcl::PointXYZ>::Ptr get_transformed_pointcloud(const Sophus::SE3f & transform) const;
	pcl::PointCloud<pcl::PointNormal>::Ptr get_transformed_pointcloud_with_normals(const Sophus::SE3f & transform) const;

	Sophus::SE3f & get_position();
	Sophus::SE3f & get_initial_position();

	cv::Mat get_subsampled_intencity(int level) const;
	Eigen::Vector3f get_subsampled_intrinsics(int level) const;
	int get_intrinsics_idx();

	cv::Mat rgb;
	cv::Mat depth;
	cv::Mat intencity;

protected:
	Sophus::SE3f position;
	Sophus::SE3f initial_position;
	Eigen::Vector3f centroid;

	std::vector<Eigen::Vector3f> & intrinsics_vector;
	int intrinsics_idx;

};



#endif /* KEYFRAME_H_ */

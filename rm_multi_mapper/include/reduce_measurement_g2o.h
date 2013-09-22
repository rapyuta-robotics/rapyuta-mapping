/*
 * reduce_jacobian_rgb.h
 *
 *  Created on: Aug 10, 2013
 *      Author: vsu
 */

#ifndef REDUCE_MEASUREMENT_G2O_H_
#define REDUCE_MEASUREMENT_G2O_H_

#include <color_keyframe.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>

struct reduce_measurement_g2o {

	enum measurement_type {
		ICP, RANSAC, DVO
	};

	struct measurement {
		int i;
		int j;
		Sophus::SE3f transform;
		measurement_type mt;
	};

	std::vector<measurement> m;
	int size;

	typedef pcl::registration::TransformationEstimationPointToPlane<
			pcl::PointNormal, pcl::PointNormal> PointToPlane;
	pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;

	const tbb::concurrent_vector<color_keyframe::Ptr> & frames;

	reduce_measurement_g2o(
			const tbb::concurrent_vector<color_keyframe::Ptr> & frames,
			int size);

	reduce_measurement_g2o(reduce_measurement_g2o & rb, tbb::split);

	void add_icp_measurement(int i, int j);
	void add_rgbd_measurement(int i, int j);
	void add_ransac_measurement(int i, int j);
	void add_floor_measurement(int i);

	void operator()(const tbb::blocked_range<
			tbb::concurrent_vector<std::pair<int, int> >::iterator> & r);

	void join(reduce_measurement_g2o& rb);

	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;
	cv::Ptr<cv::DescriptorMatcher> dm;

	void init_feature_detector();

	bool estimate_transform_ransac(const pcl::PointCloud<pcl::PointXYZ> & src,
			const pcl::PointCloud<pcl::PointXYZ> & dst,
			const std::vector<cv::DMatch> matches, int num_iter,
			float distance2_threshold, int min_num_inliers,
			Eigen::Affine3f & trans, std::vector<bool> & inliers);

	void compute_features(const cv::Mat & rgb, const cv::Mat & depth,
			const Eigen::Vector3f & intrinsics,
			cv::Ptr<cv::FeatureDetector> & fd,
			cv::Ptr<cv::DescriptorExtractor> & de,
			std::vector<cv::KeyPoint> & filtered_keypoints,
			pcl::PointCloud<pcl::PointXYZ> & keypoints3d,
			cv::Mat & descriptors);

	bool find_transform(const color_keyframe::Ptr & fi, const color_keyframe::Ptr & fj,
			Sophus::SE3f & t);

};

#endif /* REDUCE_JACOBIAN_SLAM_G2O_H_ */

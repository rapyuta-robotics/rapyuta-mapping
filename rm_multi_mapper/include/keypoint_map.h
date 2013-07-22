/*
 * keypoint_map.h
 *
 *  Created on: Jul 18, 2013
 *      Author: vsu
 */

#ifndef KEYPOINT_MAP_H_
#define KEYPOINT_MAP_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>

#include <octomap/OcTree.h>
#include <octomap/ColorOcTree.h>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/publisher.h>


typedef struct {
	int cam_id;
	int point_id;
	Eigen::Vector2f coord;
} observation;

class keypoint_map {

public:

	keypoint_map(cv::Mat & rgb, cv::Mat & depth, Eigen::Affine3f & transform);
	keypoint_map(const std::string & dir_name);

	bool merge_keypoint_map(const keypoint_map & other, int min_num_inliers);

	void remove_bad_points(int min_num_observations);

	void optimize();

	void get_octree(octomap::OcTree & tree);

	void extract_surface();

	float compute_error();

	void align_z_axis();

	void save(const std::string & dir_name);

	void publish_keypoints(ros::Publisher & pub);

	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;
	cv::Ptr<cv::DescriptorMatcher> dm;

	pcl::PointCloud<pcl::PointXYZ> keypoints3d;
	cv::Mat descriptors;
	std::vector<float> weights;

	std::vector<Eigen::Affine3f> camera_positions;
	std::vector<observation> observations;

	Eigen::Vector4f intrinsics;

	std::vector<cv::Mat> rgb_imgs;
	std::vector<cv::Mat> depth_imgs;

	Eigen::Vector3f offset;

};

#endif /* KEYPOINT_MAP_H_ */

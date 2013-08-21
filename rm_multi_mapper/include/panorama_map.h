/*
 * icp_map.h
 *
 *  Created on: Aug 7, 2013
 *      Author: vsu
 */

#ifndef PANORAMA_MAP_H_
#define PANORAMA_MAP_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>

#include <octomap_server.h>
#include <reduce_jacobian_icp.h>
#include <reduce_jacobian_icp_p2p.h>
#include <reduce_jacobian_rgb.h>
#include <reduce_jacobian_rgb_3d.h>

class panorama_map {
public:

	panorama_map();

	void add_frame(const cv::Mat & gray_f, const cv::Mat & depth, const cv::Mat & rgb,
			const Sophus::SE3f & transform);

	bool merge(panorama_map & other);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_pointcloud();
	void set_octomap(RmOctomapServer::Ptr & server);

	std::vector<cv::Mat> intencity_panoramas;
	std::vector<cv::Mat> rgb_panoramas;
	std::vector<cv::Mat> depth_panoramas;
	std::vector<Sophus::SE3f> position_panoramas;

};

#endif /* PANORAMA_MAP_H_ */

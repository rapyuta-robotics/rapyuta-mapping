/*
 * icp_map.h
 *
 *  Created on: Aug 7, 2013
 *      Author: vsu
 */

#ifndef ICP_MAP_H_
#define ICP_MAP_H_

#include <opencv2/core/core.hpp>
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

class icp_map {
public:

	typedef tbb::concurrent_vector<keyframe::Ptr>::iterator keyframe_reference;

	icp_map();

	keyframe_reference add_frame(const cv::Mat rgb, const cv::Mat depth,
			const Sophus::SE3f & transform);
	void optimize();
	void optimize_p2p();
	void optimize_rgb(int level);
	void optimize_rgb_with_intrinsics(int level);
	void optimize_rgb_3d(int level);
	void optimize_rgb_3d_with_intrinsics(int level);
	void set_octomap(RmOctomapServer::Ptr & server);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_map_pointcloud();
	cv::Mat get_panorama_image();

	void align_z_axis();

	void optimization_loop();

	void save(const std::string & dir_name);
	void load(const std::string & dir_name);

	tbb::concurrent_vector<keyframe::Ptr> frames;
	std::vector<Eigen::Vector3f> intrinsics_vector;
	boost::mutex position_modification_mutex;
	boost::thread optimization_loop_thread;
};

#endif /* ICP_MAP_H_ */

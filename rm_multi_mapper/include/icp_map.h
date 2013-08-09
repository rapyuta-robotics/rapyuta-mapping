/*
 * icp_map.h
 *
 *  Created on: Aug 7, 2013
 *      Author: vsu
 */

#ifndef ICP_MAP_H_
#define ICP_MAP_H_

#include <opencv2/core/core.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>

#include <octomap_server.h>

class keyframe {

public:

	typedef boost::shared_ptr<keyframe> Ptr;

	keyframe(const cv::Mat & rgb, const cv::Mat & depth,
			const Sophus::SE3f & position);

	Eigen::Vector3f get_centroid() const;
	pcl::PointCloud<pcl::PointXYZ>::Ptr get_pointcloud() const;
	pcl::PointCloud<pcl::PointXYZ>::Ptr get_pointcloud(float min_height,
			float max_height) const;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_colored_pointcloud() const;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_colored_pointcloud(
			float min_height, float max_height) const;
	Sophus::SE3f & get_position();
	Sophus::SE3f & get_initial_position();

	cv::Mat get_subsampled_intencity(int level);
	Eigen::Vector3f get_subsampled_intrinsics(int level);

	cv::Mat rgb;
	cv::Mat depth;
	cv::Mat intencity;

protected:
	Sophus::SE3f position;
	Sophus::SE3f initial_position;
	Eigen::Vector3f centroid;
	Eigen::Vector3f intrinsics;

};

struct reduce_jacobian_icp {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;
	int size;

	tbb::concurrent_vector<keyframe::Ptr> & frames;

	pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> ce;
	pcl::registration::CorrespondenceRejectorOneToOne cr;

	reduce_jacobian_icp(tbb::concurrent_vector<keyframe::Ptr> & frames,
			int size);

	reduce_jacobian_icp(reduce_jacobian_icp& rb, tbb::split);

	void operator()(
			const tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>& r);

	void join(reduce_jacobian_icp& rb);

};

struct reduce_jacobian_rgb {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;
	int size;
	int subsample_level;

	tbb::concurrent_vector<keyframe::Ptr> & frames;

	pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> ce;
	pcl::registration::CorrespondenceRejectorOneToOne cr;

	reduce_jacobian_rgb(tbb::concurrent_vector<keyframe::Ptr> & frames,
			int size, int subsample_level);

	reduce_jacobian_rgb(reduce_jacobian_rgb& rb, tbb::split);

	void compute_frame_jacobian(const Eigen::Vector3f & i,
			const Eigen::Matrix3f & Rwi, const Eigen::Matrix3f & Rwj,
			Eigen::Matrix<float, 9, 3> & Ji, Eigen::Matrix<float, 9, 3> & Jj);

	void operator()(
			const tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>& r);

	void join(reduce_jacobian_rgb& rb);

};

class icp_map {
public:

	typedef tbb::concurrent_vector<keyframe::Ptr>::iterator keyframe_reference;

	icp_map();

	keyframe_reference add_frame(const cv::Mat rgb, const cv::Mat depth,
			const Sophus::SE3f & transform);
	void optimize();
	void optimize_rgb(int level);

	void set_octomap(RmOctomapServer::Ptr & server);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_map_pointcloud();

	void optimization_loop();

	void save(const std::string & dir_name);
	void load(const std::string & dir_name);

	tbb::concurrent_vector<keyframe::Ptr> frames;
	boost::mutex position_modification_mutex;
	boost::thread optimization_loop_thread;
};

#endif /* ICP_MAP_H_ */

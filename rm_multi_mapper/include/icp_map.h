/*
 * icp_map.h
 *
 *  Created on: Aug 7, 2013
 *      Author: vsu
 */

#ifndef ICP_MAP_H_
#define ICP_MAP_H_

#include <opencv2/core/core.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>

#include <sophus/se3.hpp>

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>

class keyframe {

public:

	typedef boost::shared_ptr<keyframe> Ptr;

	keyframe(const cv::Mat & rgb, const cv::Mat & depth,
			const Sophus::SE3f & position);

	Eigen::Vector3f get_centroid() const;
	pcl::PointCloud<pcl::PointXYZ>::Ptr get_pointcloud() const;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_colored_pointcloud() const;
	Sophus::SE3f & get_position();

protected:
	cv::Mat rgb;
	cv::Mat depth;
	Sophus::SE3f position;
	Eigen::Vector3f centroid;
	Eigen::Vector3f intrinsics;

};

struct reduce_jacobian {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;

	tbb::concurrent_vector<keyframe::Ptr> & frames;

	pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> ce;
	pcl::registration::CorrespondenceRejectorOneToOne cr;

	reduce_jacobian(tbb::concurrent_vector<keyframe::Ptr> & frames);

	reduce_jacobian(reduce_jacobian& rb, tbb::split);

	void operator()(
			const tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>& r);

	void join(reduce_jacobian& rb);

};

class icp_map {
public:

	icp_map();

	void add_frame(const cv::Mat rgb, const cv::Mat depth,
			const Sophus::SE3f & transform);
	void optimize();

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_map_pointcloud();

	tbb::concurrent_vector<keyframe::Ptr> frames;
};

#endif /* ICP_MAP_H_ */

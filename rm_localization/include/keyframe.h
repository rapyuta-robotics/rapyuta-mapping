#ifndef KEYFRAME_H_
#define KEYFRAME_H_

#include <frame.h>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

struct convert_depth_to_pointcloud {
	const cv::Mat & depth;
	const Eigen::Vector3f & intrinsics;
	pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud;

	convert_depth_to_pointcloud(const cv::Mat & depth,
			const Eigen::Vector3f & intrinsics,
			pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) :
			depth(depth), intrinsics(intrinsics), cloud(cloud) {
	}

	void operator()(const tbb::blocked_range<int>& range) const {
		pcl::PointXYZ bad_point;
		bad_point.x = bad_point.y = bad_point.z =
				std::numeric_limits<float>::quiet_NaN();
		for (int i = range.begin(); i != range.end(); i++) {
			int u = i % depth.cols;
			int v = i / depth.cols;

			pcl::PointXYZ p;
			p.z = depth.at<float>(v, u);

			if (p.z > 0) {
				p.x = (u - intrinsics[1]) * p.z / intrinsics[0];
				p.y = (v - intrinsics[2]) * p.z / intrinsics[0];

				cloud->at(u, v) = p;
			} else {
				cloud->at(u, v) = bad_point;
			}

		}

	}
};

struct reduce_jacobian {

	Sophus::Matrix6f JtJ;
	Sophus::Vector6f Jte;

	const cv::Mat & intencity;
	const cv::Mat & intencity_dx;
	const cv::Mat & intencity_dy;
	const cv::Mat & intencity_warped;
	const cv::Mat & depth_warped;
	const Eigen::Vector3f & intrinsics;
	const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud;

	reduce_jacobian(const cv::Mat & intencity, const cv::Mat & intencity_dx,
			const cv::Mat & intencity_dy, const cv::Mat & intencity_warped,
			const cv::Mat & depth_warped, const Eigen::Vector3f & intrinsics,
			const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) :
			intencity(intencity), intencity_dx(intencity_dx), intencity_dy(
					intencity_dy), intencity_warped(intencity_warped), depth_warped(
					depth_warped), intrinsics(intrinsics), cloud(cloud) {

		JtJ.setZero();
		Jte.setZero();

	}

	reduce_jacobian(reduce_jacobian & rb, tbb::split) :
			intencity(rb.intencity), intencity_dx(rb.intencity_dx), intencity_dy(
					rb.intencity_dy), intencity_warped(rb.intencity_warped), depth_warped(
					rb.depth_warped), intrinsics(rb.intrinsics), cloud(rb.cloud) {
		JtJ.setZero();
		Jte.setZero();
	}

	inline void compute_jacobian(const pcl::PointXYZ & p,
			Eigen::Matrix<float, 2, 6> & J) {

		J(0,0) = 1/p.z;
		J(0,1) = 0;
		J(0,2) = -p.x/p.z*p.z;
		J(0,3) = -p.x*p.y/p.z*p.z;
		J(0,4) = (p.x*p.x + p.z*p.z)/p.z*p.z;
		J(0,5) = -p.y/p.z;
		J(1,0) = 0;
		J(1,1) = 1/p.z;
		J(1,2) = -p.y/p.z*p.z;
		J(1,3) = -(p.y*p.z + p.z*p.z)/p.z*p.z;
		J(1,4) = p.x*p.y/p.z*p.z;
		J(1,5) = p.x/p.z;

	}

	void operator()(const tbb::blocked_range<int>& range) {
		for (int i = range.begin(); i != range.end(); i++) {
			int u = i % depth_warped.cols;
			int v = i / depth_warped.cols;

			pcl::PointXYZ p = cloud->at(u, v);
			if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)
					&& depth_warped.at<float>(v, u) != 0) {

				float error = intencity.at<float>(v, u) - intencity_warped.at<float>(v, u);

				Eigen::Matrix<float, 1, 2> Ji;
				Eigen::Matrix<float, 2, 6> Jw;
				Eigen::Matrix<float, 1, 6> J;
				Ji[0] = intencity_dx.at<float>(v, u) * intrinsics[0];
				Ji[1] = intencity_dy.at<float>(v, u) * intrinsics[0];

				compute_jacobian(p, Jw);

				J = Ji*Jw;

				JtJ += J.transpose()*J;
				Jte += J.transpose()*error;

			}

		}

	}

	void join(reduce_jacobian& rb) {
		JtJ += rb.JtJ;
		Jte += rb.Jte;
	}

};

class keyframe: public frame {

public:

	typedef boost::shared_ptr<keyframe> Ptr;

	keyframe(const cv::Mat & yuv, const cv::Mat & depth,
			const Sophus::SE3f & position, const Eigen::Vector3f & intrinsics,
			int max_level = 3);

	void estimate_position(frame & f);

	inline cv::Mat get_i_dx(int level) {
		return get_subsampled(intencity_pyr_dx, level);
	}

	inline cv::Mat get_i_dy(int level) {
		return get_subsampled(intencity_pyr_dy, level);
	}

	inline Eigen::Vector3f get_intrinsics(int level) {
		return intrinsics / (1 << level);
	}

	inline pcl::PointCloud<pcl::PointXYZ>::Ptr get_pointcloud(int level) {
		return clouds[level];
	}

protected:
	cv::Mat intencity_pyr_dx;
	cv::Mat intencity_pyr_dy;
	Eigen::Vector3f intrinsics;

	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;

};

#endif /* KEYFRAME_H_ */

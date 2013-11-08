/*
 * g2o_worker.cpp
 *
 *  Created on: Sept 29, 2013
 *      Author: mayanks43
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>

#include <keyframe_map.h>
#include <reduce_measurement_g2o_dist.h>
#include <util.h>

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <rm_multi_mapper_db/G2oWorkerAction.h>

class G2oWorkerAction {
protected:
	ros::NodeHandle nh_;
	actionlib::SimpleActionServer<rm_multi_mapper_db::G2oWorkerAction> as_;
	std::string action_name_;
	rm_multi_mapper_db::G2oWorkerFeedback feedback_;
	rm_multi_mapper_db::G2oWorkerResult result_;
	boost::shared_ptr<keyframe_map> map;
	cv::Ptr<cv::DescriptorMatcher> dm;
	util U;

public:

	G2oWorkerAction(std::string name) :
			as_(nh_, name, boost::bind(&G2oWorkerAction::executeCB, this, _1),
					false), action_name_(name) {
		dm = new cv::FlannBasedMatcher;
		as_.start();

	}

	~G2oWorkerAction(void) {
	}

	void executeCB(const rm_multi_mapper_db::G2oWorkerGoalConstPtr & goal) {
		ros::Rate r(1);
		bool success = true;

		if (as_.isPreemptRequested() || !ros::ok()) {
			ROS_INFO("%s: Preempted", action_name_.c_str());
			as_.setPreempted();
			success = false;
		}

		for (size_t i = 0; i < goal->Overlap.size(); i++) {

			pcl::PointCloud<pcl::PointXYZ> keypoints3d1, keypoints3d2;
			cv::Mat descriptors1, descriptors2;

			//std::cerr << "Trying to match " << goal->Overlap[i].first << " "
			//		<< goal->Overlap[i].second << std::endl;

			U.get_keypoints(goal->Overlap[i].first, keypoints3d1, descriptors1);
			U.get_keypoints(goal->Overlap[i].second, keypoints3d2,
					descriptors2);

			Sophus::SE3f t;
			if (find_transform(keypoints3d1, keypoints3d2, descriptors1,
					descriptors2, t)) {
				U.add_measurement(goal->Overlap[i].first,
						goal->Overlap[i].second, t, "RANSAC");
			}

		}

		std::cout << "Done";
		if (success) {
			ROS_INFO("%s: Succeeded", action_name_.c_str());
			result_.reply = true;
			as_.setSucceeded(result_);
		}
	}

	bool find_transform(const pcl::PointCloud<pcl::PointXYZ> & keypoints3d_i,
			const pcl::PointCloud<pcl::PointXYZ> & keypoints3d_j,
			const cv::Mat & descriptors_i, const cv::Mat & descriptors_j,
			Sophus::SE3f & t) const {

		std::vector<cv::DMatch> matches, matches_filtered;
		dm->match(descriptors_j, descriptors_i, matches);

		Eigen::Affine3f transform;
		std::vector<bool> inliers;

		bool res = estimate_transform_ransac(keypoints3d_j, keypoints3d_i,
				matches, 5000, 0.03 * 0.03, 20, transform, inliers);

		if (res) {
			t = Sophus::SE3f(transform.rotation(), transform.translation());
			return true;

		}

		return false;

	}

	bool estimate_transform_ransac(const pcl::PointCloud<pcl::PointXYZ> & src,
			const pcl::PointCloud<pcl::PointXYZ> & dst,
			const std::vector<cv::DMatch> matches, int num_iter,
			float distance2_threshold, int min_num_inliers,
			Eigen::Affine3f & trans, std::vector<bool> & inliers) const {

		int max_inliers = 0;

		if (matches.size() < min_num_inliers)
			return false;

		for (int iter = 0; iter < num_iter; iter++) {

			int rand_idx[3];
			// Select 3 random points
			for (int i = 0; i < 3; i++) {
				rand_idx[i] = rand() % matches.size();
			}

			while (rand_idx[0] == rand_idx[1] || rand_idx[0] == rand_idx[2]
					|| rand_idx[1] == rand_idx[2]) {
				for (int i = 0; i < 3; i++) {
					rand_idx[i] = rand() % matches.size();
				}
			}

			//std::cerr << "Random idx " << rand_idx[0] << " " << rand_idx[1] << " "
			//		<< rand_idx[2] << " " << matches.size() << std::endl;

			Eigen::Matrix3f src_rand, dst_rand;

			for (int i = 0; i < 3; i++) {
				src_rand.col(i) =
						src[matches[rand_idx[i]].queryIdx].getVector3fMap();
				dst_rand.col(i) =
						dst[matches[rand_idx[i]].trainIdx].getVector3fMap();

			}

			Eigen::Affine3f transformation;
			transformation = Eigen::umeyama(src_rand, dst_rand, false);

			//std::cerr << "src_rand " << std::endl << src_rand << std::endl;
			//std::cerr << "dst_rand " << std::endl << dst_rand << std::endl;
			//std::cerr << "src_rand_trans " << std::endl << transformation * src_rand
			//		<< std::endl;
			//std::cerr << "trans " << std::endl << transformation.matrix()
			//		<< std::endl;

			int current_num_inliers = 0;
			std::vector<bool> current_inliers;
			current_inliers.resize(matches.size());
			for (size_t i = 0; i < matches.size(); i++) {

				Eigen::Vector4f distance_vector = transformation
						* src[matches[i].queryIdx].getVector4fMap()
						- dst[matches[i].trainIdx].getVector4fMap();

				current_inliers[i] = distance_vector.squaredNorm()
						< distance2_threshold;
				if (current_inliers[i])
					current_num_inliers++;

			}

			if (current_num_inliers > max_inliers) {
				max_inliers = current_num_inliers;
				inliers = current_inliers;
			}
		}

		if (max_inliers < min_num_inliers) {
			return false;
		}

		Eigen::Matrix3Xf src_rand(3, max_inliers), dst_rand(3, max_inliers);

		int col_idx = 0;
		for (size_t i = 0; i < inliers.size(); i++) {
			if (inliers[i]) {
				src_rand.col(col_idx) =
						src[matches[i].queryIdx].getVector3fMap();
				dst_rand.col(col_idx) =
						dst[matches[i].trainIdx].getVector3fMap();
				col_idx++;
			}

		}

		trans = Eigen::umeyama(src_rand, dst_rand, false);
		trans.makeAffine();

		std::cerr << max_inliers << std::endl;

		return true;

	}

};

int main(int argc, char** argv) {
	ros::init(argc, argv, argv[1]);
	G2oWorkerAction worker(ros::this_node::getName());
	ros::spin();

	return 0;
}


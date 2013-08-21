/*
 * icp_map.cpp
 *
 *  Created on: Aug 7, 2013
 *      Author: vsu
 */

#include <panorama_map.h>
#include <boost/filesystem.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <fstream>

void init_feature_detector(cv::Ptr<cv::FeatureDetector> & fd,
		cv::Ptr<cv::DescriptorExtractor> & de,
		cv::Ptr<cv::DescriptorMatcher> & dm);

bool estimate_transform_ransac(const pcl::PointCloud<pcl::PointXYZ> & src,
		const pcl::PointCloud<pcl::PointXYZ> & dst,
		const std::vector<cv::DMatch> matches, int num_iter,
		float distance2_threshold, int min_num_inliers, Eigen::Affine3f & trans,
		std::vector<bool> & inliers);

void get_panorama_features(const cv::Mat & gray, const cv::Mat & depth,
		std::vector<cv::KeyPoint> & filtered_keypoints,
		pcl::PointCloud<pcl::PointXYZ> & keypoints3d, cv::Mat & descriptors);

panorama_map::panorama_map() {

}

void panorama_map::add_frame(const cv::Mat & gray_f, const cv::Mat & depth,
		const cv::Mat & rgb, const Sophus::SE3f & transform) {
	cv::Mat gray;
	gray_f.convertTo(gray, CV_8U, 255);
	intencity_panoramas.push_back(gray);
	rgb_panoramas.push_back(rgb);
	depth_panoramas.push_back(depth);
	position_panoramas.push_back(transform);
}

bool panorama_map::merge(panorama_map & other) {

	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;
	cv::Ptr<cv::DescriptorMatcher> dm;

	init_feature_detector(fd, de, dm);

	bool result = false;

	for (int iter = 0; iter < 10; iter++) {

		int i = rand() % intencity_panoramas.size();
		int j = rand() % other.intencity_panoramas.size();

		std::vector<cv::KeyPoint> keypoints_i, keypoints_j;
		pcl::PointCloud<pcl::PointXYZ> keypoints3d_i, keypoints3d_j;
		cv::Mat descriptors_i, descriptors_j;

		cv::Mat gray_i, depth_i, gray_j, depth_j;
		gray_i = intencity_panoramas[i];
		depth_i = depth_panoramas[i];
		gray_j = other.intencity_panoramas[j];
		depth_j = other.depth_panoramas[j];

		get_panorama_features(gray_i, depth_i, keypoints_i, keypoints3d_i,
				descriptors_i);

		get_panorama_features(gray_j, depth_j, keypoints_j, keypoints3d_j,
				descriptors_j);

		std::vector<cv::DMatch> matches, matches_filtered;
		dm->match(descriptors_j, descriptors_i, matches);

		Eigen::Affine3f transform;
		std::vector<bool> inliers;

		bool res = estimate_transform_ransac(keypoints3d_j, keypoints3d_i,
				matches, 2000, 0.05 * 0.05, 20, transform, inliers);

		if (res) {
			for (size_t k = 0; k < matches.size(); k++) {
				if (inliers[k]) {
					matches_filtered.push_back(matches[k]);
				}
			}

			cv::Mat matches_img;
			cv::drawMatches(gray_j, keypoints_j, gray_i, keypoints_i,
					matches_filtered, matches_img, cv::Scalar(0, 255, 0));

			cv::imshow("Matches", matches_img);
			cv::waitKey(0);

			Sophus::SE3f t(transform.rotation(), transform.translation());
			Sophus::SE3f Mw1w2 = position_panoramas[i] * t
					* other.position_panoramas[j].inverse();

			for (size_t k = 0; k < other.intencity_panoramas.size(); k++) {
				intencity_panoramas.push_back(other.intencity_panoramas[k]);
				depth_panoramas.push_back(other.depth_panoramas[k]);
				position_panoramas.push_back(
						Mw1w2 * other.position_panoramas[k]);
			}

			break;

		} else {
			std::cerr << "Could not merge maps" << std::endl;
			cv::Mat matches_img;
			cv::drawMatches(gray_j, keypoints_j, gray_i, keypoints_i, matches,
					matches_img, cv::Scalar(0, 255, 0));

			cv::imshow("Matches", matches_img);
			cv::waitKey(0);
		}
	}

	return result;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr panorama_map::get_pointcloud() {

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZRGB>);

	float cx = intencity_panoramas[0].cols / 2.0;
	float cy = intencity_panoramas[0].rows / 2.0;

	float scale_x = 2 * M_PI / intencity_panoramas[0].cols;
	float scale_y = 0.5 * M_PI / intencity_panoramas[0].rows;

	for (size_t i = 0; i < intencity_panoramas.size(); i++) {

		for (int v = 0; v < intencity_panoramas[i].rows; v++) {
			for (int u = 0; u < intencity_panoramas[i].cols; u++) {

				if (depth_panoramas[i].at<float>(v, u) != 0) {
					float phi = (u - cx) * scale_x;
					float theta = (v - cy) * scale_y;

					Eigen::Vector3f vec(cos(theta) * cos(phi),
							-cos(theta) * sin(phi), -sin(theta));

					vec *= depth_panoramas[i].at<float>(v, u);
					vec = position_panoramas[i] * vec;
					pcl::PointXYZRGB p;
					p.getVector3fMap() = vec;
					p.r = p.g = p.b = 255; //intencity_panoramas[i].at<float>(v, u) * 255;

					cloud->push_back(p);
				}
			}
		}

	}

	return cloud;
}

void panorama_map::set_octomap(RmOctomapServer::Ptr & server) {

	float cx = intencity_panoramas[0].cols / 2.0;
	float cy = intencity_panoramas[0].rows / 2.0;

	float scale_x = 2 * M_PI / intencity_panoramas[0].cols;
	float scale_y = 0.5 * M_PI / intencity_panoramas[0].rows;

	for (size_t i = 0; i < intencity_panoramas.size(); i++) {

		pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
				new pcl::PointCloud<pcl::PointXYZ>);

		for (int v = 0; v < intencity_panoramas[i].rows; v++) {
			for (int u = 0; u < intencity_panoramas[i].cols; u++) {

				if (depth_panoramas[i].at<float>(v, u) != 0) {
					float phi = (u - cx) * scale_x;
					float theta = (v - cy) * scale_y;

					Eigen::Vector3f vec(cos(theta) * cos(phi),
							-cos(theta) * sin(phi), -sin(theta));

					vec *= depth_panoramas[i].at<float>(v, u);
					vec = position_panoramas[i] * vec;
					pcl::PointXYZ p;
					p.getVector3fMap() = vec;

					if (p.z > 0.1 && p.z < 0.8)
						point_cloud->push_back(p);
				}
			}
		}
		Eigen::Vector3f pos = position_panoramas[i].translation();
		server->insertScan(tf::Point(pos[0], pos[1], pos[2]),
				pcl::PointCloud<pcl::PointXYZ>(), *point_cloud);

	}

	server->publishAll(ros::Time::now());
}

/*
 * test.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <util.h>
#include <set>

int main() {

	Eigen::Vector4f intrinsics;
	intrinsics << 525.0, 525.0, 319.5, 239.5;

	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;
	cv::Ptr<cv::DescriptorMatcher> dm;

	init_feature_detector(fd, de, dm);


	cv::Mat rgb1 = cv::imread("../panorama5/rgb/1372788772.32563458.png",
			CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat depth1 = cv::imread("../panorama5/depth/1372788772.32563458.png",
			CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat rgb2 = cv::imread("../panorama5/rgb/1372788773.373165709.png",
			CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat depth2 = cv::imread("../panorama5/depth/1372788773.373165709.png",
			CV_LOAD_IMAGE_UNCHANGED);

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	pcl::PointCloud<pcl::PointXYZ> keypoints3d1, keypoints3d2;
	cv::Mat descriptors1, descriptors2;

	compute_features(rgb1, depth1, intrinsics, fd, de, keypoints1, keypoints3d1,
			descriptors1);

	compute_features(rgb2, depth2, intrinsics, fd, de, keypoints2, keypoints3d2,
			descriptors2);

	std::vector<cv::DMatch> matches;
	dm->match(descriptors1, descriptors2, matches);

	std::set<int> point_set;
	for (size_t i = 0; i < matches.size(); i++) {
		if (point_set.find(matches[i].trainIdx) == point_set.end()) {
			point_set.insert(matches[i].trainIdx);
		} else {
			std::cerr << "Index " << matches[i].trainIdx
					<< " repeats multiple times" << std::endl;
		}
	}

	Eigen::Affine3f transform;
	std::vector<bool> inliers;
	estimate_transform_ransac(keypoints3d1, keypoints3d2, matches, 1000,
			0.03 * 0.03, 20, transform, inliers);

	std::cerr << "After RANSAC " << std::endl << std::endl;

	point_set.clear();
	std::set<int> bad_idx;
	for (size_t i = 0; i < matches.size(); i++) {
		if (inliers[i])
			if (point_set.find(matches[i].trainIdx) == point_set.end()) {
				point_set.insert(matches[i].trainIdx);
			} else {
				std::cerr << "Index " << matches[i].trainIdx
						<< " repeats multiple times" << std::endl;
				bad_idx.insert(matches[i].trainIdx);
			}
	}

	std::vector<cv::DMatch> bad_matches;
	for (std::set<int>::iterator it = bad_idx.begin(); it != bad_idx.end();
			it++) {
		for (size_t i = 0; i < matches.size(); i++) {
			if (inliers[i])
				if (matches[i].trainIdx == *it) {
					bad_matches.push_back(matches[i]);
				}
		}

	}

	std::vector<cv::DMatch> filtered_matches;
	for (int i = 0; i < matches.size(); i++) {
		if (inliers[i])
			filtered_matches.push_back(matches[i]);
	}

	cv::Mat res;
	cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, matches, res,
			cv::Scalar(0, 255, 0));

	cv::imshow("Matches", res);
	cv::waitKey(0);

	cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, filtered_matches, res,
			cv::Scalar(0, 255, 0));

	cv::imshow("Matches", res);
	cv::waitKey(0);

	cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, bad_matches, res,
			cv::Scalar(0, 255, 0));

	cv::imshow("Matches", res);
	cv::waitKey(0);

	std::cerr << descriptors1.cols << " " << descriptors1.rows << std::endl;

	return 0;
}

/*
 * test.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <util.h>
#include <set>
#include <opencv2/calib3d/calib3d.hpp>

int main() {

	Eigen::Vector4f intrinsics;
	intrinsics << 525.0, 525.0, 319.5, 239.5;

	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;
	cv::Ptr<cv::DescriptorMatcher> dm;

	init_feature_detector(fd, de, dm);

	cv::Mat rgb1 = cv::imread("rgb1/1.png", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat depth1 = cv::imread("depth1/1.png", CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat rgb2 = cv::imread("rgb1/2.png", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat depth2 = cv::imread("depth1/2.png", CV_LOAD_IMAGE_UNCHANGED);

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	pcl::PointCloud<pcl::PointXYZ> keypoints3d1, keypoints3d2;
	cv::Mat descriptors1, descriptors2;

	compute_features(rgb1, depth1, intrinsics, fd, de, keypoints1, keypoints3d1,
			descriptors1);

	compute_features(rgb2, depth2, intrinsics, fd, de, keypoints2, keypoints3d2,
			descriptors2);

	std::vector<cv::DMatch> matches, matched_filtered;
	dm->match(descriptors1, descriptors2, matches);

	std::vector<cv::Vec2f> vec1, vec2;

	cv::Mat res;
	cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, matched_filtered, res,
			cv::Scalar(0, 255, 0));

	cv::imshow("Matches", res);
	cv::waitKey(0);

	std::cerr << descriptors1.cols << " " << descriptors1.rows << std::endl;

	return 0;
}

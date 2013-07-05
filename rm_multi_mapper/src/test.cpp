/*
 * test.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <util.h>

int main() {

	Eigen::Vector4f intrinsics;
	intrinsics << 525.0, 525.0, 319.5, 239.5;

	cv::Ptr<cv::FeatureDetector> fd = new cv::SurfFeatureDetector;
	fd->setInt("hessianThreshold", 400);
	fd->setBool("extended", true);
	fd->setBool("upright", true);

	cv::Ptr<cv::DescriptorExtractor> de = new cv::SurfDescriptorExtractor;

	cv::Ptr<cv::DescriptorMatcher> dm = new cv::BFMatcher;

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

	Eigen::Affine3f transform;
	std::vector<bool> inliers;
	estimate_transform_ransac(keypoints3d1, keypoints3d2, matches, 1000,
			0.03 * 0.03, 20, transform, inliers);

	std::vector<cv::DMatch> filtered_matches;
	for (int i = 0; i < matches.size(); i++) {
		if (inliers[i])
			filtered_matches.push_back(matches[i]);
	}



	cv::Mat res;
	cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, filtered_matches, res,
			cv::Scalar(0, 255, 0));

	cv::imshow("Matches", res);
	cv::waitKey(0);

	cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, matches, res,
			cv::Scalar(0, 255, 0));

	cv::imshow("Matches", res);
	cv::waitKey(0);

	std::cerr << descriptors1.cols << " " << descriptors1.rows << std::endl;

	return 0;
}

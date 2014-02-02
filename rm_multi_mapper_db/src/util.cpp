#include <util.h>

using namespace std;

util::util() {

	// Classes for feature extraction
	de = new cv::SurfDescriptorExtractor;
	fd = new cv::SurfFeatureDetector;

	fd->setInt("hessianThreshold", 400);
	fd->setInt("extended", 1);
	fd->setInt("upright", 1);

	de->setInt("hessianThreshold", 400);
	de->setInt("extended", 1);
	de->setInt("upright", 1);

	dm = new cv::FlannBasedMatcher;

}

util::~util() {
}


void util::compute_features(const cv::Mat & rgb, const cv::Mat & depth,
		const Eigen::Vector3f & intrinsics,
		std::vector<cv::KeyPoint> & filtered_keypoints,
		pcl::PointCloud<pcl::PointXYZ> & keypoints3d, cv::Mat & descriptors) {
	cv::Mat gray;

	if (rgb.channels() != 1) {
		cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
	} else {
		gray = rgb.clone();
	}

	cv::GaussianBlur(gray, gray, cv::Size(3, 3), 3);

	int threshold = 400;
	fd->setInt("hessianThreshold", threshold);

	//int threshold = 100;
	//fd->setInt("thres", threshold);

	std::vector<cv::KeyPoint> keypoints;

	cv::Mat mask(depth.size(), CV_8UC1);
	depth.convertTo(mask, CV_8U);

	fd->detect(gray, keypoints, mask);

	for (int i = 0; i < 5; i++) {
		if (keypoints.size() < 300) {
			threshold = threshold / 2;
			fd->setInt("hessianThreshold", threshold);
			//fd->setInt("thres", threshold);
			keypoints.clear();
			fd->detect(gray, keypoints, mask);
		} else {
			break;
		}
	}

	if (keypoints.size() > 400)
		keypoints.resize(400);

	filtered_keypoints.clear();
	keypoints3d.clear();

	for (size_t i = 0; i < keypoints.size(); i++) {
		if (depth.at<unsigned short>(keypoints[i].pt) != 0) {
			filtered_keypoints.push_back(keypoints[i]);

			pcl::PointXYZ p;
			p.z = depth.at<unsigned short>(keypoints[i].pt) / 1000.0f;
			p.x = (keypoints[i].pt.x - intrinsics[1]) * p.z / intrinsics[0];
			p.y = (keypoints[i].pt.y - intrinsics[2]) * p.z / intrinsics[0];

			//ROS_INFO("Point %f %f %f from  %f %f ", p.x, p.y, p.z, keypoints[i].pt.x, keypoints[i].pt.y);

			keypoints3d.push_back(p);

		}
	}

	de->compute(gray, filtered_keypoints, descriptors);
}

bool util::find_transform(const pcl::PointCloud<pcl::PointXYZ> & keypoints3d_i,
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

bool util::estimate_transform_ransac(const pcl::PointCloud<pcl::PointXYZ> & src,
		const pcl::PointCloud<pcl::PointXYZ> & dst,
		const std::vector<cv::DMatch> matches, int num_iter,
		float distance2_threshold, size_t min_num_inliers,
		Eigen::Affine3f & trans, std::vector<bool> & inliers) const {

	size_t max_inliers = 0;

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

		size_t current_num_inliers = 0;
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

	std::cerr << "max inliers" << max_inliers << std::endl;

	return true;

}



#include <reduce_measurement_g2o.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

void reduce_measurement_g2o::init_feature_detector() {
	de = new cv::SurfDescriptorExtractor;
	dm = new cv::FlannBasedMatcher;
	fd = new cv::SurfFeatureDetector;

	fd->setInt("hessianThreshold", 400);
	fd->setInt("extended", 1);
	fd->setInt("upright", 1);

	de->setInt("extended", 1);
	de->setInt("upright", 1);

}

bool reduce_measurement_g2o::estimate_transform_ransac(
		const pcl::PointCloud<pcl::PointXYZ> & src,
		const pcl::PointCloud<pcl::PointXYZ> & dst,
		const std::vector<cv::DMatch> matches, int num_iter,
		float distance2_threshold, int min_num_inliers, Eigen::Affine3f & trans,
		std::vector<bool> & inliers) {

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
			src_rand.col(col_idx) = src[matches[i].queryIdx].getVector3fMap();
			dst_rand.col(col_idx) = dst[matches[i].trainIdx].getVector3fMap();
			col_idx++;
		}

	}

	trans = Eigen::umeyama(src_rand, dst_rand, false);
	trans.makeAffine();

	std::cerr << max_inliers << std::endl;

	return true;

}
void reduce_measurement_g2o::compute_features(const cv::Mat & rgb,
		const cv::Mat & depth, const Eigen::Vector3f & intrinsics,
		cv::Ptr<cv::FeatureDetector> & fd,
		cv::Ptr<cv::DescriptorExtractor> & de,
		std::vector<cv::KeyPoint> & filtered_keypoints,
		pcl::PointCloud<pcl::PointXYZ> & keypoints3d, cv::Mat & descriptors) {
	cv::Mat gray;

	if (rgb.channels() != 1) {
		cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
	} else {
		gray = rgb;
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

bool reduce_measurement_g2o::find_transform(const color_keyframe::Ptr & fi,
		const color_keyframe::Ptr & fj, Sophus::SE3f & t) {

	std::vector<cv::KeyPoint> keypoints_i, keypoints_j;
	pcl::PointCloud<pcl::PointXYZ> keypoints3d_i, keypoints3d_j;
	cv::Mat descriptors_i, descriptors_j;

	compute_features(fi->get_i(0), fi->get_d(0), fi->get_intrinsics(0), fd, de,
			keypoints_i, keypoints3d_i, descriptors_i);

	compute_features(fj->get_i(0), fj->get_d(0), fj->get_intrinsics(0), fd, de,
			keypoints_j, keypoints3d_j, descriptors_j);

	std::vector<cv::DMatch> matches, matches_filtered;
	dm->match(descriptors_j, descriptors_i, matches);

	Eigen::Affine3f transform;
	std::vector<bool> inliers;

	bool res = estimate_transform_ransac(keypoints3d_j, keypoints3d_i, matches,
			100, 0.03 * 0.03, 20, transform, inliers);

	t = Sophus::SE3f(transform.rotation(), transform.translation());

	return res;
}

reduce_measurement_g2o::reduce_measurement_g2o(
		const tbb::concurrent_vector<color_keyframe::Ptr> & frames, int size) :
		size(size), frames(frames) {

	init_feature_detector();

	icp.setMaxCorrespondenceDistance(0.5);
	boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
	icp.setTransformationEstimation(point_to_plane);

}

reduce_measurement_g2o::reduce_measurement_g2o(reduce_measurement_g2o& rb,
		tbb::split) :
		size(rb.size), frames(rb.frames) {

	init_feature_detector();

	icp.setMaxCorrespondenceDistance(0.5);
	boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
	icp.setTransformationEstimation(point_to_plane);
}

void reduce_measurement_g2o::add_icp_measurement(int i, int j) {

	Sophus::SE3f Mij = frames[i]->get_pos().inverse() * frames[j]->get_pos();

	pcl::PointCloud<pcl::PointNormal>::Ptr Final(
			new pcl::PointCloud<pcl::PointNormal>);

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_j =
			frames[j]->get_pointcloud_with_normals(8, false);
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_i =
			frames[i]->get_pointcloud_with_normals(8, false);

	pcl::transformPointCloudWithNormals(*cloud_j, *cloud_j, Mij.matrix());

	icp.setInputCloud(cloud_j);
	icp.setInputTarget(cloud_i);
	icp.align(*Final);
	if (icp.hasConverged()) {

		Eigen::Affine3f tm(icp.getFinalTransformation());
		Mij = Sophus::SE3f(tm.rotation(), tm.translation()) * Mij;

		measurement meas;
		meas.i = i;
		meas.j = j;
		meas.transform = Mij;
		meas.mt = ICP;

		m.push_back(meas);

	}

}

void reduce_measurement_g2o::add_rgbd_measurement(int i, int j) {

	Sophus::SE3f Mij;

	if (frames[i]->estimate_relative_position(*frames[j], Mij)) {

		measurement meas;
		meas.i = i;
		meas.j = j;
		meas.transform = Mij;
		meas.mt = DVO;

		m.push_back(meas);

	}

}

void reduce_measurement_g2o::add_ransac_measurement(int i, int j) {

	Sophus::SE3f Mij;

	if (find_transform(frames[i], frames[j], Mij)) {

		ROS_INFO("Found correspondances between %d and %d", i, j);

		measurement meas;
		meas.i = i;
		meas.j = j;
		meas.transform = Mij;
		meas.mt = RANSAC;

		m.push_back(meas);

	}

}

void reduce_measurement_g2o::operator()(
		const tbb::blocked_range<
				tbb::concurrent_vector<std::pair<int, int> >::iterator>& r) {
	for (tbb::concurrent_vector<std::pair<int, int> >::iterator it = r.begin();
			it != r.end(); it++) {
		int i = it->first;
		int j = it->second;

		add_ransac_measurement(i, j);

	}

}

void reduce_measurement_g2o::join(reduce_measurement_g2o& rb) {
	m.insert(m.end(), rb.m.begin(), rb.m.end());
}

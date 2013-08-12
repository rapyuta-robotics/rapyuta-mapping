/*
 * icp_map.cpp
 *
 *  Created on: Aug 7, 2013
 *      Author: vsu
 */

#include <icp_map.h>
#include <boost/filesystem.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <util.h>

icp_map::icp_map()
// : optimization_loop_thread(boost::bind(&icp_map::optimization_loop, this)) {
{
	Eigen::Vector3f intrinsics;
	intrinsics << 525.0, 319.5, 239.5;
	intrinsics /= 2;
	intrinsics_vector.push_back(intrinsics);
}

icp_map::keyframe_reference icp_map::add_frame(const cv::Mat rgb,
		const cv::Mat depth, const Sophus::SE3f & transform) {
	keyframe::Ptr k(new keyframe(rgb, depth, transform, intrinsics_vector, 0));
	return frames.push_back(k);
}

void icp_map::optimize() {

	int size = frames.size();

	tbb::concurrent_vector<std::pair<int, int> > overlaping_keyframes;

	for (int i = 0; i < size; i++) {

		for (int j = 0; j < size; j++) {

			if (i != j) {

				float centroid_distance = (frames[i]->get_centroid()
						- frames[j]->get_centroid()).squaredNorm();

				Eigen::Quaternionf diff_quat =
						frames[i]->get_position().unit_quaternion()
								* frames[j]->get_position().unit_quaternion().inverse();

				float angle = 2 * std::acos(std::abs(diff_quat.w()));

				if (angle < M_PI / 6) {
					overlaping_keyframes.push_back(std::make_pair(i, j));
					//std::cerr << i << " and " << j
					//		<< " are connected with angle distance " << angle
					//		<< std::endl;
				}
			}
		}
	}

	reduce_jacobian_icp rj(frames, size);

	tbb::parallel_reduce(
			tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>(
					overlaping_keyframes.begin(), overlaping_keyframes.end()),
			rj);

	Eigen::VectorXf update =
			-rj.JtJ.block(6, 6, (size - 1) * 6, (size - 1) * 6).ldlt().solve(
					rj.Jte.segment(6, (size - 1) * 6));

	std::cerr << "Max update " << update.maxCoeff() << " " << update.minCoeff()
			<< std::endl;

	position_modification_mutex.lock();
	for (int i = 0; i < size - 1; i++) {

		frames[i + 1]->get_position() = Sophus::SE3f::exp(
				update.segment<6>(i * 6)) * frames[i + 1]->get_position();

	}
	position_modification_mutex.unlock();

}

void icp_map::optimize_p2p() {

	int size = frames.size();

	tbb::concurrent_vector<std::pair<int, int> > overlaping_keyframes;

	for (int i = 0; i < size; i++) {

		for (int j = 0; j < size; j++) {

			if (i != j) {

				float centroid_distance = (frames[i]->get_centroid()
						- frames[j]->get_centroid()).squaredNorm();

				Eigen::Quaternionf diff_quat =
						frames[i]->get_position().unit_quaternion()
								* frames[j]->get_position().unit_quaternion().inverse();

				float angle = 2 * std::acos(std::abs(diff_quat.w()));

				if (angle < M_PI / 6) {
					overlaping_keyframes.push_back(std::make_pair(i, j));
					//std::cerr << i << " and " << j
					//		<< " are connected with angle distance " << angle
					//		<< std::endl;
				}
			}
		}
	}

	reduce_jacobian_icp_p2p rj(frames, size);

	tbb::parallel_reduce(
			tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>(
					overlaping_keyframes.begin(), overlaping_keyframes.end()),
			rj);

	Eigen::VectorXf update =
			-rj.JtJ.block(6, 6, (size - 1) * 6, (size - 1) * 6).ldlt().solve(
					rj.Jte.segment(6, (size - 1) * 6));

	std::cerr << "Max update " << update.maxCoeff() << " " << update.minCoeff()
			<< std::endl;

	position_modification_mutex.lock();
	for (int i = 0; i < size - 1; i++) {

		frames[i + 1]->get_position() = Sophus::SE3f::exp(
				update.segment<6>(i * 6)) * frames[i + 1]->get_position();

	}
	position_modification_mutex.unlock();

}

void icp_map::optimize_rgb(int level) {

	int size = frames.size();
	int intrinsics_size = intrinsics_vector.size();

	tbb::concurrent_vector<std::pair<int, int> > overlaping_keyframes;

	for (int i = 0; i < size; i++) {

		for (int j = 0; j < size; j++) {

			if (i != j) {

				float centroid_distance = (frames[i]->get_centroid()
						- frames[j]->get_centroid()).squaredNorm();

				Eigen::Quaternionf diff_quat =
						frames[i]->get_position().unit_quaternion()
								* frames[j]->get_position().unit_quaternion().inverse();

				float angle = 2 * std::acos(std::abs(diff_quat.w()));

				if (angle < M_PI / 6) {
					overlaping_keyframes.push_back(std::make_pair(i, j));
					if (i == 0 || j == 0)
						std::cerr << i << " and " << j
								<< " are connected with angle distance "
								<< angle << std::endl;
				}
			}
		}
	}

	reduce_jacobian_rgb rj(frames, intrinsics_vector, size, intrinsics_size,
			level);

	tbb::parallel_reduce(
			tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>(
					overlaping_keyframes.begin(), overlaping_keyframes.end()),
			rj);

	/*
	 rj(
	 tbb::blocked_range<
	 tbb::concurrent_vector<std::pair<int, int> >::iterator>(
	 overlaping_keyframes.begin(), overlaping_keyframes.end()));
	 */

	Eigen::VectorXf update =
			-rj.JtJ.block(3, 3, (size - 1) * 3, (size - 1) * 3).ldlt().solve(
					rj.Jte.segment(3, (size - 1) * 3));

	std::cerr << "Max update " << update.maxCoeff() << " " << update.minCoeff()
			<< std::endl;

	position_modification_mutex.lock();
	for (int i = 0; i < size - 1; i++) {

		Sophus::SO3f current_rotation = frames[i + 1]->get_position().so3();
		current_rotation = Sophus::SO3f::exp(update.segment<3>(i * 3))
				* current_rotation;

		frames[i + 1]->get_position().so3() = current_rotation;

	}

	position_modification_mutex.unlock();

	for (int i = 0; i < intrinsics_size; i++) {
		std::cerr << "Intrinsics" << std::endl << intrinsics_vector[i]
				<< std::endl;
	}

}

void icp_map::optimize_rgb_with_intrinsics(int level) {

	int size = frames.size();
	int intrinsics_size = intrinsics_vector.size();

	tbb::concurrent_vector<std::pair<int, int> > overlaping_keyframes;

	for (int i = 0; i < size; i++) {

		for (int j = 0; j < size; j++) {

			if (i != j) {

				float centroid_distance = (frames[i]->get_centroid()
						- frames[j]->get_centroid()).squaredNorm();

				Eigen::Quaternionf diff_quat =
						frames[i]->get_position().unit_quaternion()
								* frames[j]->get_position().unit_quaternion().inverse();

				float angle = 2 * std::acos(std::abs(diff_quat.w()));

				if (angle < M_PI / 4) {
					overlaping_keyframes.push_back(std::make_pair(i, j));
					if (i == 4)
						std::cerr << i << " and " << j
								<< " are connected with angle distance "
								<< angle << std::endl;
				}
			}
		}
	}

	reduce_jacobian_rgb rj(frames, intrinsics_vector, size, intrinsics_size,
			level);

	tbb::parallel_reduce(
			tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>(
					overlaping_keyframes.begin(), overlaping_keyframes.end()),
			rj);

	/*
	 rj(
	 tbb::blocked_range<
	 tbb::concurrent_vector<std::pair<int, int> >::iterator>(
	 overlaping_keyframes.begin(), overlaping_keyframes.end()));

	 */

	Eigen::VectorXf update = -rj.JtJ.block(3, 3,
			(size - 1) * 3 + intrinsics_size * 3,
			(size - 1) * 3 + intrinsics_size * 3).ldlt().solve(
			rj.Jte.segment(3, (size - 1) * 3 + intrinsics_size * 3));

	std::cerr << "Max update " << update.maxCoeff() << " " << update.minCoeff()
			<< std::endl;

	position_modification_mutex.lock();
	for (int i = 0; i < size - 1; i++) {

		Sophus::SO3f current_rotation = frames[i + 1]->get_position().so3();
		current_rotation = Sophus::SO3f::exp(update.segment<3>(i * 3))
				* current_rotation;

		frames[i + 1]->get_position().so3() = current_rotation;

	}

	for (int i = 0; i < intrinsics_size; i++) {
		intrinsics_vector[i] =
				update.segment<3>((size - 1) * 3 + i * 3).array().exp()
						* intrinsics_vector[i].array();
	}

	position_modification_mutex.unlock();

	for (int i = 0; i < intrinsics_size; i++) {
		std::cerr << "Intrinsics" << std::endl << intrinsics_vector[i]
				<< std::endl;
	}

}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr icp_map::get_map_pointcloud() {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr res(
			new pcl::PointCloud<pcl::PointXYZRGB>);

	for (size_t i = 0; i < frames.size(); i++) {

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
				frames[i]->get_colored_pointcloud();

		*res += *cloud;

	}

	return res;

}

void icp_map::optimize_rgb_3d(int level) {

	int size = frames.size();
	int intrinsics_size = intrinsics_vector.size();

	tbb::concurrent_vector<std::pair<int, int> > overlaping_keyframes;

	for (int i = 0; i < size; i++) {

		for (int j = 0; j < size; j++) {

			if (i != j) {

				float centroid_distance = (frames[i]->get_centroid()
						- frames[j]->get_centroid()).squaredNorm();

				Eigen::Quaternionf diff_quat =
						frames[i]->get_position().unit_quaternion()
								* frames[j]->get_position().unit_quaternion().inverse();

				float angle = 2 * std::acos(std::abs(diff_quat.w()));

				if (angle < M_PI / 6) {
					overlaping_keyframes.push_back(std::make_pair(i, j));
					if (i == 0 || j == 0)
						std::cerr << i << " and " << j
								<< " are connected with angle distance "
								<< angle << std::endl;
				}
			}
		}
	}

	reduce_jacobian_rgb_3d rj(frames, intrinsics_vector, size, intrinsics_size,
			level);

	/*
	 tbb::parallel_reduce(
	 tbb::blocked_range<
	 tbb::concurrent_vector<std::pair<int, int> >::iterator>(
	 overlaping_keyframes.begin(), overlaping_keyframes.end()),
	 rj);
	 */

	rj(
			tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>(
					overlaping_keyframes.begin(), overlaping_keyframes.end()));

	Eigen::VectorXf update =
			-rj.JtJ.block(6, 6, (size - 1) * 6, (size - 1) * 6).ldlt().solve(
					rj.Jte.segment(6, (size - 1) * 6));

	std::cerr << "Max update " << update.maxCoeff() << " " << update.minCoeff()
			<< std::endl;

	position_modification_mutex.lock();
	for (int i = 0; i < size - 1; i++) {

		frames[i + 1]->get_position() = Sophus::SE3f::exp(
				update.segment<6>(i * 6)) * frames[i + 1]->get_position();

	}

	position_modification_mutex.unlock();

}

void icp_map::optimize_rgb_3d_with_intrinsics(int level) {

	int size = frames.size();
	int intrinsics_size = intrinsics_vector.size();

	tbb::concurrent_vector<std::pair<int, int> > overlaping_keyframes;

	for (int i = 0; i < size; i++) {

		for (int j = 0; j < size; j++) {

			if (i != j) {

				float centroid_distance = (frames[i]->get_centroid()
						- frames[j]->get_centroid()).squaredNorm();

				Eigen::Quaternionf diff_quat =
						frames[i]->get_position().unit_quaternion()
								* frames[j]->get_position().unit_quaternion().inverse();

				float angle = 2 * std::acos(std::abs(diff_quat.w()));

				if (angle < M_PI / 4) {
					overlaping_keyframes.push_back(std::make_pair(i, j));
					if (i == 0 || j == 0)
						std::cerr << i << " and " << j
								<< " are connected with angle distance "
								<< angle << std::endl;
				}
			}
		}
	}

	reduce_jacobian_rgb_3d rj(frames, intrinsics_vector, size, intrinsics_size,
			level);

	/*
	 tbb::parallel_reduce(
	 tbb::blocked_range<
	 tbb::concurrent_vector<std::pair<int, int> >::iterator>(
	 overlaping_keyframes.begin(), overlaping_keyframes.end()),
	 rj);
	 */

	rj(
			tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>(
					overlaping_keyframes.begin(), overlaping_keyframes.end()));

	Eigen::VectorXf update = -rj.JtJ.block(6, 6,
			(size - 1) * 6 + intrinsics_size * 3,
			(size - 1) * 6 + intrinsics_size * 3).ldlt().solve(
			rj.Jte.segment(6, (size - 1) * 6 + intrinsics_size * 3));

	std::cerr << "Max update " << update.maxCoeff() << " " << update.minCoeff()
			<< std::endl;

	position_modification_mutex.lock();
	for (int i = 0; i < size - 1; i++) {

		frames[i + 1]->get_position() = Sophus::SE3f::exp(
				update.segment<6>(i * 6)) * frames[i + 1]->get_position();

	}

	for (int i = 0; i < intrinsics_size; i++) {
		intrinsics_vector[i] =
				update.segment<3>((size - 1) * 6 + i * 3).array().exp()
						* intrinsics_vector[i].array();
	}

	position_modification_mutex.unlock();

	for (int i = 0; i < intrinsics_size; i++) {
		std::cerr << "Intrinsics" << std::endl << intrinsics_vector[i]
				<< std::endl;
	}

}

void icp_map::get_panorama_image(cv::Mat & res, cv::Mat & res_depth) {
	res = cv::Mat::zeros(400, 1600, CV_32F);
	res_depth = cv::Mat::zeros(res.size(), res.type());
	cv::Mat w = cv::Mat::zeros(res.size(), res.type());
	cv::Mat w_depth = cv::Mat::zeros(res.size(), res.type());

	cv::Mat map_x(res.size(), res.type()), map_y(res.size(), res.type()),
			img_projected(res.size(), res.type()), mask(res.size(), res.type()),
			depth_projected(res.size(), res.type());

	cv::Mat image_weight(frames[0]->intencity.size(), res.type()),
			intencity_weighted(image_weight.size(), image_weight.type());
	cv::Mat depth_weight(image_weight.size(), image_weight.type()),
			depth_weighted(image_weight.size(), image_weight.type());

	float cx = image_weight.cols / 2.0;
	float cy = image_weight.rows / 2.0;
	float max_r = cy * cy;
	for (int v = 0; v < image_weight.rows; v++) {
		for (int u = 0; u < image_weight.cols; u++) {
			image_weight.at<float>(v, u) = std::max(
					1.0 - ((u - cx) * (u - cx) + (v - cy) * (v - cy)) / max_r,
					0.0);
		}
	}

	Eigen::Vector3f intrinsics = frames[0]->get_subsampled_intrinsics(0);
	for (int v = 0; v < depth_weight.rows; v++) {
		for (int u = 0; u < depth_weight.cols; u++) {
			float s1 = (u - intrinsics[1]) / intrinsics[0];
			float s2 = (v - intrinsics[2]) / intrinsics[0];
			depth_weight.at<float>(v, u) = sqrt(s1 * s1 + s2 * s2 + 1);
		}
	}

	//cv::imshow("image_weight", image_weight);
	//cv::waitKey(0);

	cx = res.cols / 2.0;
	cy = res.rows / 2.0;

	float scale_x = 2 * M_PI / res.cols;
	float scale_y = 0.5 * M_PI / res.rows;

	for (size_t i = 0; i < frames.size(); i++) {

		Eigen::Vector3f intrinsics = frames[i]->get_subsampled_intrinsics(0);
		Eigen::Quaternionf Qiw =
				frames[i]->get_position().unit_quaternion().inverse();

		Eigen::Matrix3f K;
		K << intrinsics[0], 0, intrinsics[1], 0, intrinsics[0], intrinsics[2], 0, 0, 1;
		Eigen::Matrix3f H = K * Qiw.matrix();

		for (int v = 0; v < map_x.rows; v++) {
			for (int u = 0; u < map_x.cols; u++) {

				float phi = (u - cx) * scale_x;
				float theta = (v - cy) * scale_y;

				Eigen::Vector3f vec(cos(theta) * cos(phi),
						-cos(theta) * sin(phi), -sin(theta));
				vec = H * vec;

				if (vec[2] > 0.01) {
					map_x.at<float>(v, u) = vec[0] / vec[2];
					map_y.at<float>(v, u) = vec[1] / vec[2];
				} else {
					map_x.at<float>(v, u) = -1;
					map_y.at<float>(v, u) = -1;
				}
			}
		}

		img_projected = 0.0f;
		mask = 0.0f;
		cv::multiply(frames[i]->intencity, image_weight, intencity_weighted);
		cv::multiply(frames[i]->depth, depth_weight, depth_weighted, 1 / 1000.0,
				CV_32F);
		cv::remap(intencity_weighted, img_projected, map_x, map_y,
				CV_INTER_LINEAR, cv::BORDER_TRANSPARENT, 0);
		cv::remap(image_weight, mask, map_x, map_y, CV_INTER_LINEAR,
				cv::BORDER_TRANSPARENT, 0);
		cv::remap(depth_weighted, depth_projected, map_x, map_y, CV_INTER_NN,
				cv::BORDER_TRANSPARENT, 0);
		res += img_projected;
		w += mask;

		res_depth += depth_projected;
		mask = 0.0f;
		mask.setTo(1.0, depth_projected > 0);
		w_depth += mask;

		//cv::imshow("img_projected", img_projected);
		//cv::imshow("mask", mask);
		//cv::waitKey(0);

	}

	res /= w;
	res_depth /= w_depth;
}

void icp_map::get_panorama_features(cv::Mat & gray, cv::Mat & depth,
		std::vector<cv::KeyPoint> & filtered_keypoints,
		pcl::PointCloud<pcl::PointXYZ> & keypoints3d, cv::Mat & descriptors) {

	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;
	cv::Ptr<cv::DescriptorMatcher> dm;

	init_feature_detector(fd, de, dm);

	cv::Mat gray_f;
	get_panorama_image(gray_f, depth);
	gray_f.convertTo(gray, CV_8U, 255);

	int threshold = 400;
	fd->setInt("hessianThreshold", threshold);

	std::vector<cv::KeyPoint> keypoints;

	cv::Mat mask(depth.size(), CV_8UC1);
	depth.convertTo(mask, CV_8U);

	fd->detect(gray, keypoints, mask);

	for (int i = 0; i < 5; i++) {
		if (keypoints.size() < 900) {
			threshold = threshold / 2;
			fd->setInt("hessianThreshold", threshold);
			keypoints.clear();
			fd->detect(gray, keypoints, mask);
		} else {
			break;
		}
	}

	if (keypoints.size() > 1000)
		keypoints.resize(1000);

	filtered_keypoints.clear();
	keypoints3d.clear();

	float cx = depth.cols / 2.0;
	float cy = depth.rows / 2.0;
	float scale_x = 2 * M_PI / depth.cols;
	float scale_y = 0.5 * M_PI / depth.rows;

	for (size_t i = 0; i < keypoints.size(); i++) {
		if (depth.at<float>(keypoints[i].pt) != 0) {
			filtered_keypoints.push_back(keypoints[i]);
			float phi = (keypoints[i].pt.x - cx) * scale_x;
			float theta = (keypoints[i].pt.y - cy) * scale_y;

			Eigen::Vector3f vec(cos(theta) * cos(phi), -cos(theta) * sin(phi),
					-sin(theta));
			vec *= depth.at<float>(keypoints[i].pt);
			pcl::PointXYZ p;
			p.getVector3fMap() = vec;

			keypoints3d.push_back(p);

		}
	}

	de->compute(gray, filtered_keypoints, descriptors);

}

bool icp_map::merge(icp_map & other) {

	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;
	cv::Ptr<cv::DescriptorMatcher> dm;

	init_feature_detector(fd, de, dm);

	bool result = false;

	std::vector<cv::KeyPoint> keypoints_i, keypoints_j;
	pcl::PointCloud<pcl::PointXYZ> keypoints3d_i, keypoints3d_j;
	cv::Mat descriptors_i, descriptors_j;

	cv::Mat gray_i, depth_i, gray_j, depth_j;

	get_panorama_features(gray_i, depth_i, keypoints_i, keypoints3d_i,
			descriptors_i);

	other.get_panorama_features(gray_j, depth_j, keypoints_j, keypoints3d_j,
			descriptors_j);

	std::vector<cv::DMatch> matches, matches_filtered;
	dm->match(descriptors_j, descriptors_i, matches);

	Eigen::Affine3f transform;
	std::vector<bool> inliers;

	bool res = estimate_transform_ransac(keypoints3d_j, keypoints3d_i, matches,
			20000, 0.05 * 0.05, 20, transform, inliers);

	if (res) {
		for (int i = 0; i < matches.size(); i++) {
			if (inliers[i]) {
				matches_filtered.push_back(matches[i]);
			}
		}

		cv::Mat matches_img;
		cv::drawMatches(gray_j, keypoints_j, gray_i, keypoints_i,
				matches_filtered, matches_img, cv::Scalar(0, 255, 0));

		cv::imshow("Matches", matches_img);
		cv::waitKey(0);

		Sophus::SE3f t(transform.rotation(), transform.translation());

		for(int i=0; i<other.frames.size(); i++) {
			other.frames[i]->get_position() = t * other.frames[i]->get_position();
			frames.push_back(frames[i]);
		}


	} else {
		std::cerr << "Could not merge maps" << std::endl;
		cv::Mat matches_img;
		cv::drawMatches(gray_j, keypoints_j, gray_i, keypoints_i, matches,
				matches_img, cv::Scalar(0, 255, 0));

		cv::imshow("Matches", matches_img);
		cv::waitKey(0);
	}

	return result;
}

void icp_map::align_z_axis() {

	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
			new pcl::PointCloud<pcl::PointXYZ>);

	for (size_t i = 0; i < frames.size(); i++) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = frames[i]->get_pointcloud(
				-0.2, 0.2);

		*point_cloud += *cloud;
	}

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.02);
	seg.setProbability(0.99);
	seg.setMaxIterations(5000);

	seg.setInputCloud(point_cloud);
	seg.segment(*inliers, *coefficients);

	std::cerr << "Model coefficients: " << coefficients->values[0] << " "
			<< coefficients->values[1] << " " << coefficients->values[2] << " "
			<< coefficients->values[3] << " Num inliers "
			<< inliers->indices.size() << std::endl;

	Eigen::Affine3f transform = Eigen::Affine3f::Identity();
	if (coefficients->values[2] > 0) {
		transform.matrix().coeffRef(0, 2) = coefficients->values[0];
		transform.matrix().coeffRef(1, 2) = coefficients->values[1];
		transform.matrix().coeffRef(2, 2) = coefficients->values[2];
	} else {
		transform.matrix().coeffRef(0, 2) = -coefficients->values[0];
		transform.matrix().coeffRef(1, 2) = -coefficients->values[1];
		transform.matrix().coeffRef(2, 2) = -coefficients->values[2];
	}

	transform.matrix().col(0).head<3>() =
			transform.matrix().col(1).head<3>().cross(
					transform.matrix().col(2).head<3>());
	transform.matrix().col(1).head<3>() =
			transform.matrix().col(2).head<3>().cross(
					transform.matrix().col(0).head<3>());

	transform = transform.inverse();

	transform.matrix().coeffRef(2, 3) = coefficients->values[3];

	Sophus::SE3f t(transform.rotation(), transform.translation());

	for (size_t i = 0; i < frames.size(); i++) {
		frames[i]->get_position() = t * frames[i]->get_position();
	}

}

void icp_map::set_octomap(RmOctomapServer::Ptr & server) {

	for (size_t i = 0; i < frames.size(); i++) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud =
				frames[i]->get_pointcloud(0.1, 0.8);
		Eigen::Vector3f pos = frames[i]->get_position().translation();

		server->insertScan(tf::Point(pos[0], pos[1], pos[2]),
				pcl::PointCloud<pcl::PointXYZ>(), *point_cloud);
	}

	server->publishAll(ros::Time::now());
}

void icp_map::optimization_loop() {
	while (true) {
		if (frames.size() < 50) {
			sleep(5);
		} else {
			optimize();
		}
	}
}

void icp_map::save(const std::string & dir_name) {

	if (boost::filesystem::exists(dir_name)) {
		boost::filesystem::remove_all(dir_name);
	}

	boost::filesystem::create_directory(dir_name);
	boost::filesystem::create_directory(dir_name + "/rgb");
	boost::filesystem::create_directory(dir_name + "/depth");
	boost::filesystem::create_directory(dir_name + "/intencity");
	boost::filesystem::create_directory(dir_name + "/intencity_sub_1");
	boost::filesystem::create_directory(dir_name + "/intencity_sub_2");

	for (size_t i = 0; i < frames.size(); i++) {
		cv::imwrite(
				dir_name + "/rgb/" + boost::lexical_cast<std::string>(i)
						+ ".png", frames[i]->rgb);
		cv::imwrite(
				dir_name + "/depth/" + boost::lexical_cast<std::string>(i)
						+ ".png", frames[i]->depth);

		cv::imwrite(
				dir_name + "/intencity/" + boost::lexical_cast<std::string>(i)
						+ ".png", frames[i]->intencity * 255);

		cv::imwrite(
				dir_name + "/intencity_sub_1/"
						+ boost::lexical_cast<std::string>(i) + ".png",
				frames[i]->get_subsampled_intencity(1) * 255);

		cv::imwrite(
				dir_name + "/intencity_sub_2/"
						+ boost::lexical_cast<std::string>(i) + ".png",
				frames[i]->get_subsampled_intencity(2) * 255);

	}

	std::ofstream f((dir_name + "/positions.txt").c_str(),
			std::ios_base::binary);
	for (size_t i = 0; i < frames.size(); i++) {
		Eigen::Quaternionf q = frames[i]->get_position().unit_quaternion();
		Eigen::Vector3f t = frames[i]->get_position().translation();
		int intrinsics_idx = frames[i]->get_intrinsics_idx();

		f.write((char *) q.coeffs().data(), sizeof(float) * 4);
		f.write((char *) t.data(), sizeof(float) * 3);
		f.write((char *) &intrinsics_idx, sizeof(int));

		//f << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
		//		<< t.x() << " " << t.y() << " " << t.z() << std::endl;
	}
	f.close();

	std::ofstream f1((dir_name + "/intrinsics.txt").c_str(),
			std::ios_base::binary);
	for (size_t i = 0; i < intrinsics_vector.size(); i++) {
		Eigen::Vector3f intrinsics = intrinsics_vector[i];
		f1.write((char *) intrinsics.data(), sizeof(float) * 3);
	}
	f1.close();

}

void icp_map::load(const std::string & dir_name) {

	std::vector<std::pair<Sophus::SE3f, int> > positions;

	std::ifstream f((dir_name + "/positions.txt").c_str(),
			std::ios_base::binary);
	while (f) {
		Eigen::Quaternionf q;
		Eigen::Vector3f t;
		int intrinsics_idx;

		f.read((char *) q.coeffs().data(), sizeof(float) * 4);
		f.read((char *) t.data(), sizeof(float) * 3);
		f.read((char *) &intrinsics_idx, sizeof(int));

		//std::cerr << "sizeof(q.coeffs().data()) " << sizeof(q.coeffs().data()) << " sizeof(t.data()) " << sizeof(t.data()) << std::endl;

		//f >> q.x() >> q.y() >> q.z() >> q.w() >> t.x() >> t.y() >> t.z();
		positions.push_back(std::make_pair(Sophus::SE3f(q, t), intrinsics_idx));
	}

	positions.pop_back();
	std::cerr << "Loaded " << positions.size() << " positions" << std::endl;

	std::ifstream f1((dir_name + "/intrinsics.txt").c_str(),
			std::ios_base::binary);
	intrinsics_vector.clear();
	while (f1) {
		Eigen::Vector3f intrinsics;
		f1.read((char *) intrinsics.data(), sizeof(float) * 3);
		intrinsics_vector.push_back(intrinsics);
	}
	intrinsics_vector.pop_back();
	std::cerr << "Loaded " << intrinsics_vector.size() << " intrinsics"
			<< std::endl;

	for (size_t i = 0; i < positions.size(); i++) {
		cv::Mat rgb = cv::imread(
				dir_name + "/rgb/" + boost::lexical_cast<std::string>(i)
						+ ".png", CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat depth = cv::imread(
				dir_name + "/depth/" + boost::lexical_cast<std::string>(i)
						+ ".png", CV_LOAD_IMAGE_UNCHANGED);

		keyframe::Ptr k(
				new keyframe(rgb, depth, positions[i].first, intrinsics_vector,
						positions[i].second));
		frames.push_back(k);
	}

}


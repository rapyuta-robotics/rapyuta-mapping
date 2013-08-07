/*
 * icp_map.cpp
 *
 *  Created on: Aug 7, 2013
 *      Author: vsu
 */

#include <icp_map.h>

keyframe::keyframe(const cv::Mat & rgb, const cv::Mat & depth,
		const Sophus::SE3f & position) :
		rgb(rgb), depth(depth), position(position) {
	intrinsics << 525.0, 319.5, 239.5;

	int num_points = 0;
	centroid.setZero();

	for (int v = 0; v < depth.rows; v += 2) {
		for (int u = 0; u < depth.cols; u += 2) {
			if (depth.at<unsigned short>(v, u) != 0) {
				pcl::PointXYZ p;
				p.z = depth.at<unsigned short>(v, u) / 1000.0f;
				p.x = (u - intrinsics[1]) * p.z / intrinsics[0];
				p.y = (v - intrinsics[2]) * p.z / intrinsics[0];

				centroid += p.getVector3fMap();
				num_points++;

			}
		}
	}

	centroid /= num_points;

}

Eigen::Vector3f keyframe::get_centroid() const {
	return position * centroid;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr keyframe::get_pointcloud() const {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZ>);

	for (int v = 0; v < depth.rows; v += 2) {
		for (int u = 0; u < depth.cols; u += 2) {
			if (depth.at<unsigned short>(v, u) != 0) {
				pcl::PointXYZ p;
				p.z = depth.at<unsigned short>(v, u) / 1000.0f;
				p.x = (u - intrinsics[1]) * p.z / intrinsics[0];
				p.y = (v - intrinsics[2]) * p.z / intrinsics[0];

				p.getVector3fMap() = position * p.getVector3fMap();

				cloud->push_back(p);

			}
		}
	}

	return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyframe::get_colored_pointcloud() const {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZRGB>);

	for (int v = 0; v < depth.rows; v += 2) {
		for (int u = 0; u < depth.cols; u += 2) {
			if (depth.at<unsigned short>(v, u) != 0) {

				cv::Vec3b color = rgb.at<cv::Vec3b>(v, u);

				pcl::PointXYZRGB p;
				p.z = depth.at<unsigned short>(v, u) / 1000.0f;
				p.x = (u - intrinsics[1]) * p.z / intrinsics[0];
				p.y = (v - intrinsics[2]) * p.z / intrinsics[0];

				p.b = color[0];
				p.g = color[1];
				p.r = color[2];

				p.getVector3fMap() = position * p.getVector3fMap();

				cloud->push_back(p);

			}
		}
	}

	return cloud;
}

Sophus::SE3f & keyframe::get_position() {
	return position;
}

reduce_jacobian::reduce_jacobian(tbb::concurrent_vector<keyframe::Ptr> & frames) :
		frames(frames) {

	int size = frames.size();
	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);

}

reduce_jacobian::reduce_jacobian(reduce_jacobian& rb, tbb::split) :
		frames(rb.frames) {
	int size = frames.size();
	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);
}

void reduce_jacobian::operator()(
		const tbb::blocked_range<
				tbb::concurrent_vector<std::pair<int, int> >::iterator>& r) {
	for (tbb::concurrent_vector<std::pair<int, int> >::iterator it = r.begin();
			it != r.end(); it++) {
		int i = it->first;
		int j = it->second;

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_i, cloud_j;

		cloud_i = frames[i]->get_pointcloud();
		cloud_j = frames[j]->get_pointcloud();

		pcl::Correspondences cor;
		ce.setInputCloud(cloud_j);
		ce.setInputTarget(cloud_i);
		ce.determineCorrespondences(cor, 0.5);

		cr.getRemainingCorrespondences(cor, cor);

		for (size_t k = 0; k < cor.size(); k++) {
			if (cor[k].index_match >= 0) {
				pcl::PointXYZ & pi = cloud_i->at(cor[k].index_match);
				pcl::PointXYZ & pj = cloud_j->at(cor[k].index_query);

				Eigen::Vector3f error = pi.getVector3fMap()
						- pj.getVector3fMap();
				Eigen::Matrix<float, 3, 6> Ji, Jj;
				Ji.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
				Jj.block<3, 3>(0, 0) = -Eigen::Matrix3f::Identity();

				Ji.block<3, 3>(0, 3) << 0, pi.z, -pi.y, -pi.z, 0, pi.x, pi.y, -pi.x, 0;
				Jj.block<3, 3>(0, 3) << 0, -pj.z, pj.y, pj.z, 0, -pj.x, -pj.y, pj.x, 0;

				JtJ.block<6, 6>(i * 6, i * 6) += Ji.transpose() * Ji;
				JtJ.block<6, 6>(j * 6, j * 6) += Jj.transpose() * Jj;
				JtJ.block<6, 6>(i * 6, j * 6) += Ji.transpose() * Jj;
				JtJ.block<6, 6>(j * 6, i * 6) += Jj.transpose() * Ji;

				Jte.segment<6>(i * 6) += Ji.transpose() * error;
				Jte.segment<6>(j * 6) += Jj.transpose() * error;

			}

		}
	}
}

void reduce_jacobian::join(reduce_jacobian& rb) {
	JtJ += rb.JtJ;
	Jte += rb.Jte;
}

icp_map::icp_map() {
}

void icp_map::add_frame(const cv::Mat rgb, const cv::Mat depth,
		const Sophus::SE3f & transform) {

	keyframe::Ptr k(new keyframe(rgb, depth, transform));

	frames.push_back(k);
}

void icp_map::optimize() {

	int size = frames.size();
	Eigen::MatrixXf JtJ = Eigen::MatrixXf::Zero(size * 6, size * 6);
	Eigen::VectorXf Jte = Eigen::VectorXf::Zero(size * 6);

	tbb::concurrent_vector<std::pair<int, int> > overlaping_keyframes;

	for (int i = 0; i < size; i++) {

		for (int j = 0; j < i; j++) {
			Eigen::Quaternionf diff_quat =
					frames[i]->get_position().unit_quaternion()
							* frames[j]->get_position().unit_quaternion().inverse();

			float angle = 2 * std::acos(std::abs(diff_quat.w()));

			if (angle < 0.5) {
				overlaping_keyframes.push_back(std::make_pair(i, j));
				std::cerr << i << " and " << j
						<< " are connected with angle distance " << angle
						<< std::endl;
			}
		}
	}

	reduce_jacobian rj(frames);

	tbb::parallel_reduce(
			tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>(
					overlaping_keyframes.begin(), overlaping_keyframes.end()),
			rj);

	Eigen::VectorXf update = -rj.JtJ.ldlt().solve(rj.Jte);

	std::cerr << "Max update " << update.maxCoeff() << " " << update.minCoeff()
			<< std::endl;

	for (int i = 0; i < size; i++) {

		frames[i]->get_position() = Sophus::SE3f::exp(update.segment<6>(i * 6))
				* frames[i]->get_position();

	}

}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr icp_map::get_map_pointcloud() {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr res(
			new pcl::PointCloud<pcl::PointXYZRGB>);

	for (size_t i = 0; i < frames.size(); i++) {

		std::cerr << "Adding pointcloud " << i << std::endl;

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
				frames[i]->get_colored_pointcloud();

		*res += *cloud;

	}

	return res;

}


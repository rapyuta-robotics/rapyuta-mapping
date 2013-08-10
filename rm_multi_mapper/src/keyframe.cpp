
#include <keyframe.h>

keyframe::keyframe(const cv::Mat & rgb, const cv::Mat & depth,
		const Sophus::SE3f & position,
		std::vector<Eigen::Vector3f> & intrinsics_vector, int intrinsics_idx) :
		rgb(rgb), depth(depth), position(position), initial_position(position), intrinsics_vector(
				intrinsics_vector), intrinsics_idx(intrinsics_idx) {

	Eigen::Vector3f & intrinsics = intrinsics_vector[intrinsics_idx];

	int num_points = 0;
	centroid.setZero();

	for (int v = 0; v < depth.rows; v++) {
		for (int u = 0; u < depth.cols; u++) {
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

	cv::Mat tmp;
	cv::cvtColor(rgb, tmp, CV_RGB2GRAY);
	tmp.convertTo(intencity, CV_32F, 1 / 255.0);

}

Eigen::Vector3f keyframe::get_centroid() const {
	return position * centroid;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr keyframe::get_pointcloud() const {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZ>);

	Eigen::Vector3f & intrinsics = intrinsics_vector[intrinsics_idx];

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

pcl::PointCloud<pcl::PointXYZ>::Ptr keyframe::get_pointcloud(float min_height,
		float max_height) const {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZ>);

	Eigen::Vector3f & intrinsics = intrinsics_vector[intrinsics_idx];

	for (int v = 0; v < depth.rows; v += 2) {
		for (int u = 0; u < depth.cols; u += 2) {
			if (depth.at<unsigned short>(v, u) != 0) {
				pcl::PointXYZ p;
				p.z = depth.at<unsigned short>(v, u) / 1000.0f;
				p.x = (u - intrinsics[1]) * p.z / intrinsics[0];
				p.y = (v - intrinsics[2]) * p.z / intrinsics[0];

				p.getVector3fMap() = position * p.getVector3fMap();

				if (p.z > min_height && p.z < max_height)
					cloud->push_back(p);

			}
		}
	}

	return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyframe::get_colored_pointcloud() const {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZRGB>);

	Eigen::Vector3f & intrinsics = intrinsics_vector[intrinsics_idx];

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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyframe::get_colored_pointcloud(
		float min_height, float max_height) const {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZRGB>);

	Eigen::Vector3f & intrinsics = intrinsics_vector[intrinsics_idx];

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

				if (p.z > min_height && p.z < max_height)
					cloud->push_back(p);

			}
		}
	}

	return cloud;
}

Sophus::SE3f & keyframe::get_position() {
	return position;
}

Sophus::SE3f & keyframe::get_initial_position() {
	return initial_position;
}

cv::Mat keyframe::get_subsampled_intencity(int level) {

	if (level > 0) {
		int size_reduction = 1 << level;
		cv::Mat res(rgb.rows / size_reduction, rgb.cols / size_reduction,
				intencity.type());

		for (int v = 0; v < res.rows; v++) {
			for (int u = 0; u < res.cols; u++) {

				float value = 0;

				for (int i = 0; i < size_reduction; i++) {
					for (int j = 0; j < size_reduction; j++) {
						value += intencity.at<float>(size_reduction * v + i,
								size_reduction * u + j);
					}
				}

				value /= size_reduction * size_reduction;
				res.at<float>(v, u) = value;

			}
		}

		//std::cerr << "Res size" << res.rows << " " << res.cols << std::endl;
		return res;

	} else {
		return intencity;
	}

}
Eigen::Vector3f keyframe::get_subsampled_intrinsics(int level) {

	Eigen::Vector3f & intrinsics = intrinsics_vector[intrinsics_idx];
	int size_reduction = 1 << level;
	return intrinsics / size_reduction;
}

int keyframe::get_intrinsics_idx() {
	return intrinsics_idx;
}

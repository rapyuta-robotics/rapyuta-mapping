#include <keyframe.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>

keyframe::keyframe(const cv::Mat & rgb, const cv::Mat & depth,
		const Sophus::SE3f & position,
		std::vector<Eigen::Vector3f> & intrinsics_vector, int intrinsics_idx,
		int max_level) :
		rgb(rgb), depth(depth), max_level(max_level), position(position), initial_position(
				position), intrinsics_vector(intrinsics_vector), intrinsics_idx(
				intrinsics_idx) {

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

	intencity_pyr = cv::Mat::zeros(rgb.rows, rgb.cols + rgb.cols / 2, CV_32F);
	build_pyr();
	cv::Sobel(intencity_pyr, intencity_pyr_dx, CV_32F, 1, 0);
	cv::Sobel(intencity_pyr, intencity_pyr_dy, CV_32F, 0, 1);

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

void keyframe::build_pyr() {

	if (rgb.channels() == 3) {
		cv::Mat res = get_subsampled_intencity(0);
		cv::Mat tmp;
		cv::cvtColor(rgb, tmp, CV_RGB2GRAY);
		tmp.convertTo(res, CV_32F, 1 / 255.0);
	} else if (rgb.channels() == 2) {
		cv::Mat res = get_subsampled_intencity(0);
		cv::Mat tmp;
		cv::cvtColor(rgb, tmp, CV_YUV2GRAY_UYVY);
		tmp.convertTo(res, CV_32F, 1 / 255.0);
	}

	for (int level = 1; level < max_level; level++) {
		cv::Mat prev = get_subsampled_intencity(level - 1);
		cv::Mat current = get_subsampled_intencity(level);

		for (int v = 0; v < current.rows; v++) {
			for (int u = 0; u < current.cols; u++) {
				float value = 0;
				value += prev.at<float>(2 * v + 0, 2 * u + 0);
				value += prev.at<float>(2 * v + 1, 2 * u + 0);
				value += prev.at<float>(2 * v + 0, 2 * u + 1);
				value += prev.at<float>(2 * v + 1, 2 * u + 1);
				value /= 4;
				current.at<float>(v, u) = value;
			}
		}
	}


}

cv::Mat keyframe::get_subsampled_intencity(int level) const {
	cv::Mat res;
	if (level == 0) {
		res = intencity_pyr(cv::Rect(0, 0, rgb.cols, rgb.rows));
	} else if (level == 1) {
		cv::Rect r(rgb.cols, 0, rgb.cols / 2, rgb.rows / 2);
		res = intencity_pyr(r);
	} else if (level < max_level) {
		int size_reduction = 1 << level;
		int u = rgb.cols;
		int v = (rgb.rows / 2) * (pow(0.5, level - 1) - 1) / (0.5 - 1);
		cv::Rect r(u, v, rgb.cols / size_reduction, rgb.rows / size_reduction);
		res = intencity_pyr(r);
	} else {
		std::cerr << "Requested level " << level << " is bigger than availible max level"
				<< std::endl;
	}

	return res;
}

cv::Mat keyframe::get_subsampled_intencity_dx(int level) const {
	cv::Mat res;
	if (level == 0) {
		res = intencity_pyr_dx(cv::Rect(0, 0, rgb.cols, rgb.rows));
	} else if (level == 1) {
		cv::Rect r(rgb.cols, 0, rgb.cols / 2, rgb.rows / 2);
		res = intencity_pyr_dx(r);
	} else if (level < max_level) {
		int size_reduction = 1 << level;
		int u = rgb.cols;
		int v = (rgb.rows / 2) * (pow(0.5, level - 1) - 1) / (0.5 - 1);
		cv::Rect r(u, v, rgb.cols / size_reduction, rgb.rows / size_reduction);
		res = intencity_pyr_dx(r);
	} else {
		std::cerr << "Requested level " << level << " is bigger than availible max level"
				<< std::endl;
	}

	return res;
}

cv::Mat keyframe::get_subsampled_intencity_dy(int level) const {
	cv::Mat res;
	if (level == 0) {
		res = intencity_pyr_dy(cv::Rect(0, 0, rgb.cols, rgb.rows));
	} else if (level == 1) {
		cv::Rect r(rgb.cols, 0, rgb.cols / 2, rgb.rows / 2);
		res = intencity_pyr_dy(r);
	} else if (level < max_level) {
		int size_reduction = 1 << level;
		int u = rgb.cols;
		int v = (rgb.rows / 2) * (pow(0.5, level - 1) - 1) / (0.5 - 1);
		cv::Rect r(u, v, rgb.cols / size_reduction, rgb.rows / size_reduction);
		res = intencity_pyr_dy(r);
	} else {
		std::cerr << "Requested level " << level << " is bigger than availible max level"
				<< std::endl;
	}

	return res;
}

Eigen::Vector3f keyframe::get_subsampled_intrinsics(int level) const {

	Eigen::Vector3f & intrinsics = intrinsics_vector[intrinsics_idx];
	int size_reduction = 1 << level;
	return intrinsics / size_reduction;
}

int keyframe::get_intrinsics_idx() {
	return intrinsics_idx;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr keyframe::get_original_pointcloud() const {
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

				cloud->push_back(p);

			}
		}
	}

	return cloud;
}

pcl::PointCloud<pcl::PointNormal>::Ptr keyframe::get_original_pointcloud_with_normals() const {
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud(
			new pcl::PointCloud<pcl::PointNormal>);

	Eigen::Vector3f & intrinsics = intrinsics_vector[intrinsics_idx];

	for (int v = 0; v < depth.rows; v += 2) {
		for (int u = 0; u < depth.cols; u += 2) {
			if (depth.at<unsigned short>(v, u) != 0) {
				pcl::PointNormal p;
				p.z = depth.at<unsigned short>(v, u) / 1000.0f;
				p.x = (u - intrinsics[1]) * p.z / intrinsics[0];
				p.y = (v - intrinsics[2]) * p.z / intrinsics[0];

				cloud->push_back(p);

			}
		}
	}

	pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> ne;
	ne.setInputCloud(cloud);

	pcl::search::KdTree<pcl::PointNormal>::Ptr tree(
			new pcl::search::KdTree<pcl::PointNormal>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.03);
	ne.compute(*cloud);

	return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr keyframe::get_transformed_pointcloud(
		const Sophus::SE3f & transform) const {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZ>);

	Eigen::Vector3f & intrinsics = intrinsics_vector[intrinsics_idx];

	Sophus::SE3f t = transform * position;

	for (int v = 0; v < depth.rows; v += 2) {
		for (int u = 0; u < depth.cols; u += 2) {
			if (depth.at<unsigned short>(v, u) != 0) {
				pcl::PointXYZ p;
				p.z = depth.at<unsigned short>(v, u) / 1000.0f;
				p.x = (u - intrinsics[1]) * p.z / intrinsics[0];
				p.y = (v - intrinsics[2]) * p.z / intrinsics[0];

				p.getVector3fMap() = t * p.getVector3fMap();

				cloud->push_back(p);

			}
		}
	}

	return cloud;

}

pcl::PointCloud<pcl::PointNormal>::Ptr keyframe::get_transformed_pointcloud_with_normals(
		const Sophus::SE3f & transform) const {
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud =
			get_original_pointcloud_with_normals();
	pcl::transformPointCloudWithNormals(*cloud, *cloud, transform.matrix());

	return cloud;

}

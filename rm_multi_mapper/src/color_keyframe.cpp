#include <color_keyframe.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

color_keyframe::color_keyframe(const cv::Mat & rgb, const cv::Mat & gray,
		const cv::Mat & depth, const Sophus::SE3f & position,
		const Eigen::Vector3f & intrinsics, int max_level) :
		keyframe(gray, depth, position, intrinsics, max_level), rgb(rgb) {

}

pcl::PointCloud<pcl::PointXYZ>::Ptr color_keyframe::get_pointcloud() const {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZ>);

	Eigen::Matrix<float, 4, 4, Eigen::ColMajor> transform = position.matrix();

	int size = cols * rows;
	for (int i = 0; i < size; i++) {
		Eigen::Vector4f vec = clouds[0].col(i);
		if (vec(3) > 0) {
			pcl::PointXYZ p;
			p.getVector4fMap() = transform * vec;
			cloud->push_back(p);
		}
	}

	return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_keyframe::get_colored_pointcloud(
		int subsample) const {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZRGB>);

	Eigen::Matrix<float, 4, 4, Eigen::ColMajor> transform = position.matrix();

	for (int v = 0; v < rows; v += subsample) {
		for (int u = 0; u < cols; u += subsample) {
			int i = v*cols + u;
			Eigen::Vector4f vec = clouds[0].col(i);
			if (vec(3) > 0) {

				cv::Vec3b color = rgb.at<cv::Vec3b>(v, u);

				pcl::PointXYZRGB p;
				p.getVector4fMap() = transform * vec;
				p.r = color[0];
				p.g = color[1];
				p.b = color[2];

				cloud->push_back(p);
			}
		}
	}

	return cloud;
}

color_keyframe::Ptr color_keyframe::from_msg(
		const rm_localization::Keyframe::ConstPtr & k) {

	cv::Mat rgb, gray, depth;

	rgb = cv::imdecode(k->rgb_png_data, CV_LOAD_IMAGE_UNCHANGED);
	depth = cv::imdecode(k->depth_png_data, CV_LOAD_IMAGE_UNCHANGED);
	cv::cvtColor(rgb, gray, CV_RGB2GRAY);

	Eigen::Quaternionf orientation;
	Eigen::Vector3f position, intrinsics;

	memcpy(intrinsics.data(), k->intrinsics.data(), 3 * sizeof(float));
	memcpy(orientation.coeffs().data(), k->unit_quaternion.data(),
			4 * sizeof(float));
	memcpy(position.data(), k->position.data(), 3 * sizeof(float));

	color_keyframe::Ptr res(
			new color_keyframe(rgb, gray, depth,
					Sophus::SE3f(orientation, position), intrinsics));
	return res;

}

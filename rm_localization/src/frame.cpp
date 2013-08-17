#include <frame.h>
#include <cassert>
//#include <opencv2/highgui/highgui.hpp>

frame::frame(const cv::Mat & yuv, const cv::Mat & depth,
		const Sophus::SE3f & position, int max_level) {

	assert(yuv.cols == depth.cols && yuv.rows == depth.rows);

	cols = yuv.cols;
	rows = yuv.rows;
	this->max_level = max_level;

	intencity_pyr = cv::Mat::zeros(rows, cols + cols / 2, CV_32F);
	depth_pyr = cv::Mat::zeros(rows, cols + cols / 2, CV_32F);

	cv::Mat intecity_0 = get_i(0);
	cv::Mat depth_0 = get_d(0);

	convert cvt(yuv, depth, intecity_0, depth_0);
	tbb::parallel_for(tbb::blocked_range<int>(0, cols * rows), cvt);

	for (int level = 1; level < max_level; level++) {
		cv::Mat prev_intencity = get_i(level - 1);
		cv::Mat current_intencity = get_i(level);

		cv::Mat prev_depth = get_d(level - 1);
		cv::Mat current_depth = get_d(level);

		subsample sub(prev_intencity, prev_depth, current_intencity,
				current_depth);
		tbb::parallel_for(
				tbb::blocked_range<int>(0,
						current_intencity.cols * current_intencity.rows), sub);

	}

	//cv::imshow("intencity_pyr",intencity_pyr);
	//cv::imshow("depth_pyr", depth_pyr);
	//cv::waitKey();

}

cv::Mat frame::get_subsampled(cv::Mat & image_pyr, int level) const {
	cv::Mat res;
	if (level == 0) {
		cv::Rect r(0, 0, cols, rows);
		res = image_pyr(r);
	} else if (level == 1) {
		cv::Rect r(cols, 0, cols / 2, rows / 2);
		res = image_pyr(r);
	} else if (level < max_level) {
		int size_reduction = 1 << level;
		int u = cols;
		int v = (rows / 2) * (pow(0.5, level - 1) - 1) / (0.5 - 1);
		cv::Rect r(u, v, cols / size_reduction, rows / size_reduction);
		res = image_pyr(r);

	} else {
		std::cerr << "Requested level " << level
				<< " is bigger than available max level" << std::endl;
	}

	return res;
}

void frame::warp(const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
		const Eigen::Vector3f & intrinsics, const Sophus::SE3f & position, int level,
		cv::Mat & intencity_warped, cv::Mat & depth_warped) {

	cv::Mat intencity = get_i(level);
	cv::Mat depth = get_d(level);

	assert(cloud->height == intencity.rows && cloud->width == intencity.cols);

	Sophus::SE3f transform = this->position * position.inverse();

	parallel_warp w(intencity, depth, transform, cloud, intrinsics, intencity_warped,
			depth_warped);

	tbb::parallel_for(
			tbb::blocked_range<int>(0, intencity.cols * intencity.rows), w);

}

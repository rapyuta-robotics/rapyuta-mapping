#include <frame.h>
#include <cassert>
#include <tbb/parallel_for.h>
//#include <opencv2/highgui/highgui.hpp>

frame::frame(const cv::Mat & yuv, const cv::Mat & depth,
		const Sophus::SE3f & position, const Eigen::Vector3f & intrinsics,
		int max_level) {

	assert(yuv.cols == depth.cols && yuv.rows == depth.rows);

	this->position = position;
	this->intrinsics = intrinsics;

	cols = yuv.cols;
	rows = yuv.rows;
	this->max_level = max_level;

	intencity_pyr = new uint8_t *[max_level];
	depth_pyr = new uint16_t *[max_level];

	for (int level = 0; level < max_level; level++) {
		intencity_pyr[level] = new uint8_t[(cols * rows) >> (2 * level)];
		depth_pyr[level] = new uint16_t[(cols * rows) >> (2 * level)];
	}

	if (yuv.channels() == 2) {
		convert cvt(yuv.data, intencity_pyr[0]);
		tbb::parallel_for(tbb::blocked_range<int>(0, cols * rows), cvt);

	} else if (yuv.channels() == 1) {
		memcpy(intencity_pyr[0], yuv.data, cols * rows * sizeof(uint8_t));
	}
	memcpy(depth_pyr[0], depth.data, cols * rows * sizeof(uint16_t));

	for (int level = 1; level < max_level; level++) {

		subsample sub(intencity_pyr[level - 1], depth_pyr[level - 1],
				cols >> level, rows >> level, intencity_pyr[level],
				depth_pyr[level]);
		tbb::parallel_for(
				tbb::blocked_range<int>(0, (cols * rows) >> (2 * level)), sub);

	}

	/*
	 cv::imshow("get_i(0)", get_i(0));
	 cv::imshow("get_d(0)", get_d(0));
	 cv::imshow("get_i(1)", get_i(1));
	 cv::imshow("get_d(1)", get_d(1));
	 cv::imshow("get_i(2)", get_i(2));
	 cv::imshow("get_d(2)", get_d(2));
	 cv::waitKey();
	 */

}

void frame::warp(
		const Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> & cloud,
		const Sophus::SE3f & relative_position, int level,
		cv::Mat & intencity_warped, cv::Mat & depth_warped) {

	Eigen::Matrix<float, 4, 4, Eigen::ColMajor> transform(
			relative_position.matrix());

	int c = cols >> level;
	int r = rows >> level;

	Eigen::Vector3f intrinsics = get_intrinsics(level);

	parallel_warp w(intencity_pyr[level], depth_pyr[level], transform, cloud,
			intrinsics, c, r, (float *) intencity_warped.data,
			(float *) depth_warped.data);

	tbb::parallel_for(tbb::blocked_range<int>(0, c * r), w);

}

frame::~frame() {

	for (int level = 0; level < max_level; level++) {
		delete[] intencity_pyr[level];
		delete[] depth_pyr[level];
	}

	delete[] intencity_pyr;
	delete[] depth_pyr;

}

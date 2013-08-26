#include <keyframe.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

keyframe::keyframe(const cv::Mat & yuv, const cv::Mat & depth,
		const Sophus::SE3f & position, const Eigen::Vector3f & intrinsics,
		int max_level) :
		frame(yuv, depth, position, max_level) {

	this->intrinsics = intrinsics;

	intencity_pyr_dx = new int16_t *[max_level];
	intencity_pyr_dy = new int16_t *[max_level];

	for (int level = 0; level < max_level; level++) {
		intencity_pyr_dx[level] = new int16_t[cols * rows / (1 << 2 * level)];
		intencity_pyr_dy[level] = new int16_t[cols * rows / (1 << 2 * level)];
	}

	clouds.resize(max_level);
	for (int level = 0; level < max_level; level++) {

		int c = cols >> level;
		int r = rows >> level;

		Eigen::Vector3f intrinsics = get_intrinsics(level);
		clouds[level].setZero(4, c * r);

		convert_depth_to_pointcloud sub(intencity_pyr[level], depth_pyr[level],
				intrinsics, c, r, clouds[level], intencity_pyr_dx[level],
				intencity_pyr_dy[level]);
		tbb::parallel_for(tbb::blocked_range<int>(0, c * r), sub);

	}

	/*
	 cv::imshow("intencity_pyr", intencity_pyr);
	 cv::imshow("depth_pyr", depth_pyr);
	 cv::imshow("intencity_pyr_dx", intencity_pyr_dx);
	 cv::imshow("intencity_pyr_dy", intencity_pyr_dy);
	 cv::waitKey();
	 */

}

keyframe::~keyframe() {

	for (int level = 0; level < max_level; level++) {
		delete[] intencity_pyr_dx[level];
		delete[] intencity_pyr_dy[level];
	}

	delete[] intencity_pyr_dx;
	delete[] intencity_pyr_dy;

}

void keyframe::estimate_position(frame & f) {

	int level_iterations[] = { 1, 2, 3 };

	for (int level = 2; level >= 0; level--) {
		for (int iteration = 0; iteration < level_iterations[level];
				iteration++) {

			Sophus::SE3f Mrc = position.inverse() * f.position;

			cv::Mat intencity = get_i(level), depth = get_d(level),
					intencity_dx = get_i_dx(level), intencity_dy = get_i_dy(
							level);

			Eigen::Vector3f intrinsics = get_intrinsics(level);
			cv::Mat intencity_warped(intencity.rows, intencity.cols,
					intencity.type()), depth_warped(depth.rows, depth.cols,
					depth.type());

			int c = cols >> level;
			int r = rows >> level;

			f.warp(clouds[level], intrinsics, position, level, intencity_warped,
					depth_warped);

			reduce_jacobian rj(intencity_pyr[level], intencity_pyr_dx[level],
					intencity_pyr_dy[level], (uint8_t *) intencity_warped.data,
					(uint16_t *) depth_warped.data, intrinsics, clouds[level],
					c, r);

			tbb::parallel_reduce(
					tbb::blocked_range<int>(0, intencity.cols * intencity.rows),
					rj);

			//rj(tbb::blocked_range<int>(0, intencity.cols * intencity.rows));

			Sophus::Vector6f update = -rj.JtJ.llt().solve(rj.Jte);

			//std::cerr << "update " << std::endl << update << std::endl;

			f.position = position * Sophus::SE3f::exp(update) * Mrc;

			//std::cerr << "Transform " << std::endl << f.position.matrix()
			//		<< std::endl;

			//if(level == 0) {
			//	cv::imshow("intencity_warped", intencity_warped);
			//	cv::imshow("intencity", intencity);
			//	cv::waitKey(3);
			//}

		}
	}

}

void keyframe::update_intrinsics(const Eigen::Vector3f & intrinsics) {
	this->intrinsics = intrinsics;

	for (int level = 0; level < max_level; level++) {

		int c = cols >> level;
		int r = rows >> level;

		Eigen::Vector3f intrinsics = get_intrinsics(level);
		clouds[level].setZero(4, c * r);

		convert_depth_to_pointcloud sub(intencity_pyr[level], depth_pyr[level],
				intrinsics, c, r, clouds[level], intencity_pyr_dx[level],
				intencity_pyr_dy[level]);
		tbb::parallel_for(tbb::blocked_range<int>(0, c * r), sub);

	}

}

rm_localization::Keyframe::Ptr keyframe::to_msg(
		const cv_bridge::CvImageConstPtr & yuv2, int idx) {
	rm_localization::Keyframe::Ptr k(new rm_localization::Keyframe);

	cv::Mat rgb;
	cv::cvtColor(yuv2->image, rgb, CV_YUV2BGR_UYVY);

	cv::imencode(".png", rgb, k->rgb_png_data);
	cv::imencode(".png", get_d(0), k->depth_png_data);

	k->header.frame_id = yuv2->header.frame_id;
	k->header.stamp = yuv2->header.stamp;

	memcpy(k->intrinsics.data(), intrinsics.data(), 3 * sizeof(float));
	memcpy(k->transform.unit_quaternion.data(),
			position.unit_quaternion().coeffs().data(), 4 * sizeof(float));
	memcpy(k->transform.position.data(), position.translation().data(),
			3 * sizeof(float));

	k->idx = idx;

	return k;
}

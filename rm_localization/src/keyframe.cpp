#include <keyframe.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tbb/parallel_for.h>

keyframe::keyframe(const cv::Mat & yuv, const cv::Mat & depth,
		const Sophus::SE3f & position, const Eigen::Vector3f & intrinsics,
		int max_level) :
		frame(yuv, depth, position, intrinsics, max_level) {

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

void keyframe::set_timestamp(ros::Time stamp) {
	timestamp = stamp;
}

bool keyframe::estimate_position(frame & f) {
	Sophus::SE3f Mrc;
	bool res = estimate_relative_position(f, Mrc);
	if (res) {
		f.position = position * Mrc;
	}
	return res;

}

bool keyframe::estimate_relative_position(frame & f, Sophus::SE3f & Mrc) {

	int level_iterations[] = { 2, 4, 6 };

	Mrc = position.inverse() * f.position;

	for (int level = 2; level >= 0; level--) {
		for (int iteration = 0; iteration < level_iterations[level];
				iteration++) {

			cv::Mat intencity = get_i(level), depth = get_d(level),
					intencity_dx = get_i_dx(level), intencity_dy = get_i_dy(
							level);

			cv::Mat intencity_warped(intencity.rows, intencity.cols, CV_32F),
					depth_warped(depth.rows, depth.cols, CV_32F);

			int c = cols >> level;
			int r = rows >> level;

			f.warp(clouds[level], Mrc.inverse(), level,
					intencity_warped, depth_warped);

			reduce_jacobian rj(intencity_pyr[level], intencity_pyr_dx[level],
					intencity_pyr_dy[level], (float *) intencity_warped.data,
					(float *) depth_warped.data, intrinsics, clouds[level], c,
					r);

			tbb::parallel_reduce(
					tbb::blocked_range<int>(0, intencity.cols * intencity.rows),
					rj);

			//rj(tbb::blocked_range<int>(0, intencity.cols * intencity.rows));

			if (level == 0 && iteration == level_iterations[level]-1 && (float) rj.num_points / (c * r) < 0.1) {
				return false;
			}

			//ROS_INFO("Mean error %f with %f\% valid points", std::sqrt(rj.error_sum)/rj.num_points, (float)rj.num_points / (c*r));

			Sophus::Vector6f update = -rj.JtJ.ldlt().solve(rj.Jte);

			//std::cerr << "update " << std::endl << update << std::endl;

			Mrc = Sophus::SE3f::exp(update) * Mrc;

			//std::cerr << "Transform " << std::endl << f.position.matrix()
			//		<< std::endl;

			//if(level == 0) {
			//	cv::imshow("intencity_warped", intencity_warped);
			//	cv::imshow("intencity", intencity);
			//	cv::waitKey(3);
			//}

		}
	}

	return true;

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
	if (yuv2->image.channels() == 3) {
		rgb = yuv2->image;
	} else {
		cv::cvtColor(yuv2->image, rgb, CV_YUV2RGB_UYVY);
	}

	cv::imencode(".png", rgb, k->rgb_png_data);
	cv::imencode(".png", get_d(0), k->depth_png_data);

	k->header.frame_id = yuv2->header.frame_id;
	k->header.stamp = yuv2->header.stamp;

	k->intrinsics[0] = intrinsics[0];
	k->intrinsics[1] = intrinsics[1];
	k->intrinsics[2] = intrinsics[2];

	k->transform.unit_quaternion[0] = position.unit_quaternion().coeffs()[0];
	k->transform.unit_quaternion[1] = position.unit_quaternion().coeffs()[1];
	k->transform.unit_quaternion[2] = position.unit_quaternion().coeffs()[2];
	k->transform.unit_quaternion[3] = position.unit_quaternion().coeffs()[3];

	k->transform.position[0] = position.translation()[0];
	k->transform.position[1] = position.translation()[1];
	k->transform.position[2] = position.translation()[2];

	k->idx = idx;

	return k;
}

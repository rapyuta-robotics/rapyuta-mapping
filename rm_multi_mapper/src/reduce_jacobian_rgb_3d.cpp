#include <reduce_jacobian_rgb_3d.h>
#include <opencv2/imgproc/imgproc.hpp>

reduce_jacobian_rgb_3d::reduce_jacobian_rgb_3d(
		tbb::concurrent_vector<color_keyframe::Ptr> & frames, int size,
		int subsample_level) :
		size(size), subsample_level(subsample_level), frames(frames) {

	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);

}

reduce_jacobian_rgb_3d::reduce_jacobian_rgb_3d(reduce_jacobian_rgb_3d& rb,
		tbb::split) :
		size(rb.size), subsample_level(rb.subsample_level), frames(rb.frames) {
	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);
}

void reduce_jacobian_rgb_3d::compute_frame_jacobian(const Eigen::Vector3f & i,
		const Eigen::Vector4f & p,
		const Eigen::Matrix<float, 4, 4, Eigen::ColMajor> & Miw,
		Eigen::Matrix<float, 2, 6> & J) {

	J(0,0) = (Miw(0,0)*i(0) + Miw(2,0)*i(1) - Miw(2,0)*(Miw(0,3)*i(0) + Miw(2,3)*i(1) + p(0)*(Miw(0,0)*i(0) + Miw(2,0)*i(1)) + p(1)*(Miw(0,1)*i(0) + Miw(2,1)*i(1)) + p(2)*(Miw(0,2)*i(0) + Miw(2,2)*i(1)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(0,1) = (Miw(0,1)*i(0) + Miw(2,1)*i(1) - Miw(2,1)*(Miw(0,3)*i(0) + Miw(2,3)*i(1) + p(0)*(Miw(0,0)*i(0) + Miw(2,0)*i(1)) + p(1)*(Miw(0,1)*i(0) + Miw(2,1)*i(1)) + p(2)*(Miw(0,2)*i(0) + Miw(2,2)*i(1)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(0,2) = (Miw(0,2)*i(0) + Miw(2,2)*i(1) - Miw(2,2)*(Miw(0,3)*i(0) + Miw(2,3)*i(1) + p(0)*(Miw(0,0)*i(0) + Miw(2,0)*i(1)) + p(1)*(Miw(0,1)*i(0) + Miw(2,1)*i(1)) + p(2)*(Miw(0,2)*i(0) + Miw(2,2)*i(1)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(0,3) = (p(1)*(Miw(0,2)*i(0) + Miw(2,2)*i(1) - Miw(2,2)*(Miw(0,3)*i(0) + Miw(2,3)*i(1) + p(0)*(Miw(0,0)*i(0) + Miw(2,0)*i(1)) + p(1)*(Miw(0,1)*i(0) + Miw(2,1)*i(1)) + p(2)*(Miw(0,2)*i(0) + Miw(2,2)*i(1)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))) - p(2)*(Miw(0,1)*i(0) + Miw(2,1)*i(1) - Miw(2,1)*(Miw(0,3)*i(0) + Miw(2,3)*i(1) + p(0)*(Miw(0,0)*i(0) + Miw(2,0)*i(1)) + p(1)*(Miw(0,1)*i(0) + Miw(2,1)*i(1)) + p(2)*(Miw(0,2)*i(0) + Miw(2,2)*i(1)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(0,4) = (-p(0)*(Miw(0,2)*i(0) + Miw(2,2)*i(1) - Miw(2,2)*(Miw(0,3)*i(0) + Miw(2,3)*i(1) + p(0)*(Miw(0,0)*i(0) + Miw(2,0)*i(1)) + p(1)*(Miw(0,1)*i(0) + Miw(2,1)*i(1)) + p(2)*(Miw(0,2)*i(0) + Miw(2,2)*i(1)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))) + p(2)*(Miw(0,0)*i(0) + Miw(2,0)*i(1) - Miw(2,0)*(Miw(0,3)*i(0) + Miw(2,3)*i(1) + p(0)*(Miw(0,0)*i(0) + Miw(2,0)*i(1)) + p(1)*(Miw(0,1)*i(0) + Miw(2,1)*i(1)) + p(2)*(Miw(0,2)*i(0) + Miw(2,2)*i(1)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(0,5) = (p(0)*(Miw(0,1)*i(0) + Miw(2,1)*i(1) - Miw(2,1)*(Miw(0,3)*i(0) + Miw(2,3)*i(1) + p(0)*(Miw(0,0)*i(0) + Miw(2,0)*i(1)) + p(1)*(Miw(0,1)*i(0) + Miw(2,1)*i(1)) + p(2)*(Miw(0,2)*i(0) + Miw(2,2)*i(1)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))) - p(1)*(Miw(0,0)*i(0) + Miw(2,0)*i(1) - Miw(2,0)*(Miw(0,3)*i(0) + Miw(2,3)*i(1) + p(0)*(Miw(0,0)*i(0) + Miw(2,0)*i(1)) + p(1)*(Miw(0,1)*i(0) + Miw(2,1)*i(1)) + p(2)*(Miw(0,2)*i(0) + Miw(2,2)*i(1)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(1,0) = (Miw(1,0)*i(0) + Miw(2,0)*i(2) - Miw(2,0)*(Miw(1,3)*i(0) + Miw(2,3)*i(2) + p(0)*(Miw(1,0)*i(0) + Miw(2,0)*i(2)) + p(1)*(Miw(1,1)*i(0) + Miw(2,1)*i(2)) + p(2)*(Miw(1,2)*i(0) + Miw(2,2)*i(2)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(1,1) = (Miw(1,1)*i(0) + Miw(2,1)*i(2) - Miw(2,1)*(Miw(1,3)*i(0) + Miw(2,3)*i(2) + p(0)*(Miw(1,0)*i(0) + Miw(2,0)*i(2)) + p(1)*(Miw(1,1)*i(0) + Miw(2,1)*i(2)) + p(2)*(Miw(1,2)*i(0) + Miw(2,2)*i(2)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(1,2) = (Miw(1,2)*i(0) + Miw(2,2)*i(2) - Miw(2,2)*(Miw(1,3)*i(0) + Miw(2,3)*i(2) + p(0)*(Miw(1,0)*i(0) + Miw(2,0)*i(2)) + p(1)*(Miw(1,1)*i(0) + Miw(2,1)*i(2)) + p(2)*(Miw(1,2)*i(0) + Miw(2,2)*i(2)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(1,3) = (p(1)*(Miw(1,2)*i(0) + Miw(2,2)*i(2) - Miw(2,2)*(Miw(1,3)*i(0) + Miw(2,3)*i(2) + p(0)*(Miw(1,0)*i(0) + Miw(2,0)*i(2)) + p(1)*(Miw(1,1)*i(0) + Miw(2,1)*i(2)) + p(2)*(Miw(1,2)*i(0) + Miw(2,2)*i(2)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))) - p(2)*(Miw(1,1)*i(0) + Miw(2,1)*i(2) - Miw(2,1)*(Miw(1,3)*i(0) + Miw(2,3)*i(2) + p(0)*(Miw(1,0)*i(0) + Miw(2,0)*i(2)) + p(1)*(Miw(1,1)*i(0) + Miw(2,1)*i(2)) + p(2)*(Miw(1,2)*i(0) + Miw(2,2)*i(2)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(1,4) = (-p(0)*(Miw(1,2)*i(0) + Miw(2,2)*i(2) - Miw(2,2)*(Miw(1,3)*i(0) + Miw(2,3)*i(2) + p(0)*(Miw(1,0)*i(0) + Miw(2,0)*i(2)) + p(1)*(Miw(1,1)*i(0) + Miw(2,1)*i(2)) + p(2)*(Miw(1,2)*i(0) + Miw(2,2)*i(2)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))) + p(2)*(Miw(1,0)*i(0) + Miw(2,0)*i(2) - Miw(2,0)*(Miw(1,3)*i(0) + Miw(2,3)*i(2) + p(0)*(Miw(1,0)*i(0) + Miw(2,0)*i(2)) + p(1)*(Miw(1,1)*i(0) + Miw(2,1)*i(2)) + p(2)*(Miw(1,2)*i(0) + Miw(2,2)*i(2)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));
	J(1,5) = (p(0)*(Miw(1,1)*i(0) + Miw(2,1)*i(2) - Miw(2,1)*(Miw(1,3)*i(0) + Miw(2,3)*i(2) + p(0)*(Miw(1,0)*i(0) + Miw(2,0)*i(2)) + p(1)*(Miw(1,1)*i(0) + Miw(2,1)*i(2)) + p(2)*(Miw(1,2)*i(0) + Miw(2,2)*i(2)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))) - p(1)*(Miw(1,0)*i(0) + Miw(2,0)*i(2) - Miw(2,0)*(Miw(1,3)*i(0) + Miw(2,3)*i(2) + p(0)*(Miw(1,0)*i(0) + Miw(2,0)*i(2)) + p(1)*(Miw(1,1)*i(0) + Miw(2,1)*i(2)) + p(2)*(Miw(1,2)*i(0) + Miw(2,2)*i(2)))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3))))/(Miw(2,0)*p(0) + Miw(2,1)*p(1) + Miw(2,2)*p(2) + Miw(2,3));

}

void reduce_jacobian_rgb_3d::operator()(
		const tbb::blocked_range<
				tbb::concurrent_vector<std::pair<int, int> >::iterator>& r) {
	for (tbb::concurrent_vector<std::pair<int, int> >::iterator it = r.begin();
			it != r.end(); it++) {
		int i = it->first;
		int j = it->second;

		Eigen::Matrix<float, 4, 4, Eigen::ColMajor> Miw(
				frames[i]->get_pos().inverse().matrix());
		Eigen::Matrix<float, 4, 4, Eigen::ColMajor> Mwi(
				frames[i]->get_pos().matrix());

		cv::Mat intencity = frames[i]->get_i(subsample_level), depth =
				frames[i]->get_d(subsample_level), intencity_dx =
				frames[i]->get_i_dx(subsample_level), intencity_dy =
				frames[i]->get_i_dy(subsample_level);

		Eigen::Vector3f intrinsics_i = frames[i]->get_intrinsics(
				subsample_level);
		cv::Mat intencity_warped(intencity.rows, intencity.cols,
				intencity.type()), depth_warped(depth.rows, depth.cols,
				depth.type());

		frames[j]->warp(frames[i]->get_cloud(subsample_level), intrinsics_i,
				frames[i]->get_pos(), subsample_level, intencity_warped,
				depth_warped);




		//cv::imshow("intencity", intencity);
		//cv::imshow("intencity_warped", intencity_warped);
		//cv::waitKey();


		int size = intencity.rows * intencity.cols;
		for (int vec = 0; vec < size; vec++) {
			int u = vec % intencity.cols;
			int v = vec / intencity.cols;

			Eigen::Vector4f p = frames[i]->get_cloud(subsample_level).col(vec);
			if (p(3) > 0.0f && depth_warped.at<uint16_t>(v, u) != 0) {

				p = Mwi * p;

				float error = (float) intencity.at<uint8_t>(v, u)
						- (float) intencity_warped.at<uint8_t>(v, u);

				Eigen::Matrix<float, 1, 2> Jintr;
				Eigen::Matrix<float, 2, 6> Jw;
				Eigen::Matrix<float, 1, 6> Ji, Jj;
				Jintr[0] = intencity_dx.at<int16_t>(v, u);
				Jintr[1] = intencity_dy.at<int16_t>(v, u);

				compute_frame_jacobian(intrinsics_i, p, Miw, Jw);

				Jj = Jintr * Jw;
				Ji = -Jj;

				//
				JtJ.block<6, 6>(i * 6, i * 6) += Ji.transpose() * Ji;
				JtJ.block<6, 6>(j * 6, j * 6) += Jj.transpose() * Jj;
				// i and j
				JtJ.block<6, 6>(i * 6, j * 6) += Ji.transpose() * Jj;
				JtJ.block<6, 6>(j * 6, i * 6) += Jj.transpose() * Ji;

				// errors
				Jte.segment<6>(i * 6) += Ji.transpose() * error;
				Jte.segment<6>(j * 6) += Jj.transpose() * error;

			}

		}

	}
}

void reduce_jacobian_rgb_3d::join(reduce_jacobian_rgb_3d& rb) {
	JtJ += rb.JtJ;
	Jte += rb.Jte;
}

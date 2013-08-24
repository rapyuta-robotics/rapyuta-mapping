#include <reduce_jacobian_rgb.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

reduce_jacobian_rgb::reduce_jacobian_rgb(
		tbb::concurrent_vector<color_keyframe::Ptr> & frames, int size,
		int subsample_level) :
		size(size), subsample_level(subsample_level), frames(frames) {

	JtJ.setZero(size * 3 + 3, size * 3 + 3);
	Jte.setZero(size * 3 + 3);

}

reduce_jacobian_rgb::reduce_jacobian_rgb(reduce_jacobian_rgb& rb, tbb::split) :
		size(rb.size), subsample_level(rb.subsample_level), frames(rb.frames) {
	JtJ.setZero(size * 3 + 3, size * 3 + 3);
	Jte.setZero(size * 3 + 3);
}

void reduce_jacobian_rgb::compute_frame_jacobian(const Eigen::Vector3f & i,
		const Eigen::Matrix3f & Rwi, const Eigen::Matrix3f & Rwj,
		Eigen::Matrix<float, 9, 3> & Ji, Eigen::Matrix<float, 9, 3> & Jj,
		Eigen::Matrix<float, 9, 3> & Jk) {

	Ji.coeffRef(0, 0) = (-Rwj(1, 0) * (Rwi(2, 0) * i(0) + Rwi(2, 2) * i(1))
			+ Rwj(2, 0) * (Rwi(1, 0) * i(0) + Rwi(1, 2) * i(1))) / i(0);
	Ji.coeffRef(0, 1) = (Rwj(0, 0) * (Rwi(2, 0) * i(0) + Rwi(2, 2) * i(1))
			- Rwj(2, 0) * (Rwi(0, 0) * i(0) + Rwi(0, 2) * i(1))) / i(0);
	Ji.coeffRef(0, 2) = (-Rwj(0, 0) * (Rwi(1, 0) * i(0) + Rwi(1, 2) * i(1))
			+ Rwj(1, 0) * (Rwi(0, 0) * i(0) + Rwi(0, 2) * i(1))) / i(0);
	Ji.coeffRef(1, 0) = (-Rwj(1, 1) * (Rwi(2, 0) * i(0) + Rwi(2, 2) * i(1))
			+ Rwj(2, 1) * (Rwi(1, 0) * i(0) + Rwi(1, 2) * i(1))) / i(0);
	Ji.coeffRef(1, 1) = (Rwj(0, 1) * (Rwi(2, 0) * i(0) + Rwi(2, 2) * i(1))
			- Rwj(2, 1) * (Rwi(0, 0) * i(0) + Rwi(0, 2) * i(1))) / i(0);
	Ji.coeffRef(1, 2) = (-Rwj(0, 1) * (Rwi(1, 0) * i(0) + Rwi(1, 2) * i(1))
			+ Rwj(1, 1) * (Rwi(0, 0) * i(0) + Rwi(0, 2) * i(1))) / i(0);
	Ji.coeffRef(2, 0) = -((Rwi(1, 0) * i(0) + Rwi(1, 2) * i(1))
			* (Rwj(2, 0) * i(1) + Rwj(2, 1) * i(2) - Rwj(2, 2) * i(0))
			- (Rwi(2, 0) * i(0) + Rwi(2, 2) * i(1))
					* (Rwj(1, 0) * i(1) + Rwj(1, 1) * i(2) - Rwj(1, 2) * i(0)))
			/ i(0);
	Ji.coeffRef(2, 1) = (Rwi(0, 0) * i(0) + Rwi(0, 2) * i(1))
			* (Rwj(2, 0) * i(1) / i(0) + Rwj(2, 1) * i(2) / i(0) - Rwj(2, 2))
			- (Rwi(2, 0) * i(0) + Rwi(2, 2) * i(1))
					* (Rwj(0, 0) * i(1) / i(0) + Rwj(0, 1) * i(2) / i(0)
							- Rwj(0, 2));
	Ji.coeffRef(2, 2) = -((Rwi(0, 0) * i(0) + Rwi(0, 2) * i(1))
			* (Rwj(1, 0) * i(1) + Rwj(1, 1) * i(2) - Rwj(1, 2) * i(0))
			- (Rwi(1, 0) * i(0) + Rwi(1, 2) * i(1))
					* (Rwj(0, 0) * i(1) + Rwj(0, 1) * i(2) - Rwj(0, 2) * i(0)))
			/ i(0);
	Ji.coeffRef(3, 0) = (-Rwj(1, 0) * (Rwi(2, 1) * i(0) + Rwi(2, 2) * i(2))
			+ Rwj(2, 0) * (Rwi(1, 1) * i(0) + Rwi(1, 2) * i(2))) / i(0);
	Ji.coeffRef(3, 1) = (Rwj(0, 0) * (Rwi(2, 1) * i(0) + Rwi(2, 2) * i(2))
			- Rwj(2, 0) * (Rwi(0, 1) * i(0) + Rwi(0, 2) * i(2))) / i(0);
	Ji.coeffRef(3, 2) = (-Rwj(0, 0) * (Rwi(1, 1) * i(0) + Rwi(1, 2) * i(2))
			+ Rwj(1, 0) * (Rwi(0, 1) * i(0) + Rwi(0, 2) * i(2))) / i(0);
	Ji.coeffRef(4, 0) = (-Rwj(1, 1) * (Rwi(2, 1) * i(0) + Rwi(2, 2) * i(2))
			+ Rwj(2, 1) * (Rwi(1, 1) * i(0) + Rwi(1, 2) * i(2))) / i(0);
	Ji.coeffRef(4, 1) = (Rwj(0, 1) * (Rwi(2, 1) * i(0) + Rwi(2, 2) * i(2))
			- Rwj(2, 1) * (Rwi(0, 1) * i(0) + Rwi(0, 2) * i(2))) / i(0);
	Ji.coeffRef(4, 2) = (-Rwj(0, 1) * (Rwi(1, 1) * i(0) + Rwi(1, 2) * i(2))
			+ Rwj(1, 1) * (Rwi(0, 1) * i(0) + Rwi(0, 2) * i(2))) / i(0);
	Ji.coeffRef(5, 0) = -((Rwi(1, 1) * i(0) + Rwi(1, 2) * i(2))
			* (Rwj(2, 0) * i(1) + Rwj(2, 1) * i(2) - Rwj(2, 2) * i(0))
			- (Rwi(2, 1) * i(0) + Rwi(2, 2) * i(2))
					* (Rwj(1, 0) * i(1) + Rwj(1, 1) * i(2) - Rwj(1, 2) * i(0)))
			/ i(0);
	Ji.coeffRef(5, 1) = (Rwi(0, 1) * i(0) + Rwi(0, 2) * i(2))
			* (Rwj(2, 0) * i(1) / i(0) + Rwj(2, 1) * i(2) / i(0) - Rwj(2, 2))
			- (Rwi(2, 1) * i(0) + Rwi(2, 2) * i(2))
					* (Rwj(0, 0) * i(1) / i(0) + Rwj(0, 1) * i(2) / i(0)
							- Rwj(0, 2));
	Ji.coeffRef(5, 2) = -((Rwi(0, 1) * i(0) + Rwi(0, 2) * i(2))
			* (Rwj(1, 0) * i(1) + Rwj(1, 1) * i(2) - Rwj(1, 2) * i(0))
			- (Rwi(1, 1) * i(0) + Rwi(1, 2) * i(2))
					* (Rwj(0, 0) * i(1) + Rwj(0, 1) * i(2) - Rwj(0, 2) * i(0)))
			/ i(0);
	Ji.coeffRef(6, 0) = (Rwi(1, 2) * Rwj(2, 0) - Rwi(2, 2) * Rwj(1, 0)) / i(0);
	Ji.coeffRef(6, 1) = (-Rwi(0, 2) * Rwj(2, 0) + Rwi(2, 2) * Rwj(0, 0)) / i(0);
	Ji.coeffRef(6, 2) = (Rwi(0, 2) * Rwj(1, 0) - Rwi(1, 2) * Rwj(0, 0)) / i(0);
	Ji.coeffRef(7, 0) = (Rwi(1, 2) * Rwj(2, 1) - Rwi(2, 2) * Rwj(1, 1)) / i(0);
	Ji.coeffRef(7, 1) = (-Rwi(0, 2) * Rwj(2, 1) + Rwi(2, 2) * Rwj(0, 1)) / i(0);
	Ji.coeffRef(7, 2) = (Rwi(0, 2) * Rwj(1, 1) - Rwi(1, 2) * Rwj(0, 1)) / i(0);
	Ji.coeffRef(8, 0) = (-Rwi(1, 2)
			* (Rwj(2, 0) * i(1) + Rwj(2, 1) * i(2) - Rwj(2, 2) * i(0))
			+ Rwi(2, 2)
					* (Rwj(1, 0) * i(1) + Rwj(1, 1) * i(2) - Rwj(1, 2) * i(0)))
			/ i(0);
	Ji.coeffRef(8, 1) = (Rwi(0, 2)
			* (Rwj(2, 0) * i(1) + Rwj(2, 1) * i(2) - Rwj(2, 2) * i(0))
			- Rwi(2, 2)
					* (Rwj(0, 0) * i(1) + Rwj(0, 1) * i(2) - Rwj(0, 2) * i(0)))
			/ i(0);
	Ji.coeffRef(8, 2) = (-Rwi(0, 2)
			* (Rwj(1, 0) * i(1) + Rwj(1, 1) * i(2) - Rwj(1, 2) * i(0))
			+ Rwi(1, 2)
					* (Rwj(0, 0) * i(1) + Rwj(0, 1) * i(2) - Rwj(0, 2) * i(0)))
			/ i(0);

	Jk.coeffRef(0, 0) = i(1)
			* (-Rwi(0, 2) * Rwj(0, 0) - Rwi(1, 2) * Rwj(1, 0)
					- Rwi(2, 2) * Rwj(2, 0)) / i(0);
	Jk.coeffRef(0, 1) = i(1)
			* (Rwi(0, 2) * Rwj(0, 0) + Rwi(1, 2) * Rwj(1, 0)
					+ Rwi(2, 2) * Rwj(2, 0)) / i(0);
	Jk.coeffRef(0, 2) = 0;
	Jk.coeffRef(1, 0) = i(1)
			* (-Rwi(0, 2) * Rwj(0, 1) - Rwi(1, 2) * Rwj(1, 1)
					- Rwi(2, 2) * Rwj(2, 1)) / i(0);
	Jk.coeffRef(1, 1) = i(1)
			* (Rwi(0, 2) * Rwj(0, 1) + Rwi(1, 2) * Rwj(1, 1)
					+ Rwi(2, 2) * Rwj(2, 1)) / i(0);
	Jk.coeffRef(1, 2) = 0;
	Jk.coeffRef(2, 0) = (Rwi(0, 0) * Rwj(0, 2) * i(0) * i(0)
			+ Rwi(0, 2) * Rwj(0, 0) * i(1) * i(1)
			+ Rwi(0, 2) * Rwj(0, 1) * i(1) * i(2)
			+ Rwi(1, 0) * Rwj(1, 2) * i(0) * i(0)
			+ Rwi(1, 2) * Rwj(1, 0) * i(1) * i(1)
			+ Rwi(1, 2) * Rwj(1, 1) * i(1) * i(2)
			+ Rwi(2, 0) * Rwj(2, 2) * i(0) * i(0)
			+ Rwi(2, 2) * Rwj(2, 0) * i(1) * i(1)
			+ Rwi(2, 2) * Rwj(2, 1) * i(1) * i(2)) / i(0);
	Jk.coeffRef(2, 1) = i(1)
			* (-Rwi(0, 0) * Rwj(0, 0) * i(0) - 2 * Rwi(0, 2) * Rwj(0, 0) * i(1)
					- Rwi(0, 2) * Rwj(0, 1) * i(2)
					+ Rwi(0, 2) * Rwj(0, 2) * i(0)
					- Rwi(1, 0) * Rwj(1, 0) * i(0)
					- 2 * Rwi(1, 2) * Rwj(1, 0) * i(1)
					- Rwi(1, 2) * Rwj(1, 1) * i(2)
					+ Rwi(1, 2) * Rwj(1, 2) * i(0)
					- Rwi(2, 0) * Rwj(2, 0) * i(0)
					- 2 * Rwi(2, 2) * Rwj(2, 0) * i(1)
					- Rwi(2, 2) * Rwj(2, 1) * i(2)
					+ Rwi(2, 2) * Rwj(2, 2) * i(0)) / i(0);
	Jk.coeffRef(2, 2) = -i(2)
			* (Rwj(0, 1) * (Rwi(0, 0) * i(0) + Rwi(0, 2) * i(1))
					+ Rwj(1, 1) * (Rwi(1, 0) * i(0) + Rwi(1, 2) * i(1))
					+ Rwj(2, 1) * (Rwi(2, 0) * i(0) + Rwi(2, 2) * i(1))) / i(0);
	Jk.coeffRef(3, 0) = i(2)
			* (-Rwi(0, 2) * Rwj(0, 0) - Rwi(1, 2) * Rwj(1, 0)
					- Rwi(2, 2) * Rwj(2, 0)) / i(0);
	Jk.coeffRef(3, 1) = 0;
	Jk.coeffRef(3, 2) = i(2)
			* (Rwi(0, 2) * Rwj(0, 0) + Rwi(1, 2) * Rwj(1, 0)
					+ Rwi(2, 2) * Rwj(2, 0)) / i(0);
	Jk.coeffRef(4, 0) = i(2)
			* (-Rwi(0, 2) * Rwj(0, 1) - Rwi(1, 2) * Rwj(1, 1)
					- Rwi(2, 2) * Rwj(2, 1)) / i(0);
	Jk.coeffRef(4, 1) = 0;
	Jk.coeffRef(4, 2) = i(2)
			* (Rwi(0, 2) * Rwj(0, 1) + Rwi(1, 2) * Rwj(1, 1)
					+ Rwi(2, 2) * Rwj(2, 1)) / i(0);
	Jk.coeffRef(5, 0) = (Rwi(0, 1) * Rwj(0, 2) * i(0) * i(0)
			+ Rwi(0, 2) * Rwj(0, 0) * i(1) * i(2)
			+ Rwi(0, 2) * Rwj(0, 1) * i(2) * i(2)
			+ Rwi(1, 1) * Rwj(1, 2) * i(0) * i(0)
			+ Rwi(1, 2) * Rwj(1, 0) * i(1) * i(2)
			+ Rwi(1, 2) * Rwj(1, 1) * i(2) * i(2)
			+ Rwi(2, 1) * Rwj(2, 2) * i(0) * i(0)
			+ Rwi(2, 2) * Rwj(2, 0) * i(1) * i(2)
			+ Rwi(2, 2) * Rwj(2, 1) * i(2) * i(2)) / i(0);
	Jk.coeffRef(5, 1) = -i(1)
			* (Rwj(0, 0) * (Rwi(0, 1) * i(0) + Rwi(0, 2) * i(2))
					+ Rwj(1, 0) * (Rwi(1, 1) * i(0) + Rwi(1, 2) * i(2))
					+ Rwj(2, 0) * (Rwi(2, 1) * i(0) + Rwi(2, 2) * i(2))) / i(0);
	Jk.coeffRef(5, 2) = i(2)
			* (-Rwi(0, 1) * Rwj(0, 1) * i(0) - Rwi(0, 2) * Rwj(0, 0) * i(1)
					- 2 * Rwi(0, 2) * Rwj(0, 1) * i(2)
					+ Rwi(0, 2) * Rwj(0, 2) * i(0)
					- Rwi(1, 1) * Rwj(1, 1) * i(0)
					- Rwi(1, 2) * Rwj(1, 0) * i(1)
					- 2 * Rwi(1, 2) * Rwj(1, 1) * i(2)
					+ Rwi(1, 2) * Rwj(1, 2) * i(0)
					- Rwi(2, 1) * Rwj(2, 1) * i(0)
					- Rwi(2, 2) * Rwj(2, 0) * i(1)
					- 2 * Rwi(2, 2) * Rwj(2, 1) * i(2)
					+ Rwi(2, 2) * Rwj(2, 2) * i(0)) / i(0);
	Jk.coeffRef(6, 0) = -(Rwi(0, 2) * Rwj(0, 0) + Rwi(1, 2) * Rwj(1, 0)
			+ Rwi(2, 2) * Rwj(2, 0)) / i(0);
	Jk.coeffRef(6, 1) = 0;
	Jk.coeffRef(6, 2) = 0;
	Jk.coeffRef(7, 0) = -(Rwi(0, 2) * Rwj(0, 1) + Rwi(1, 2) * Rwj(1, 1)
			+ Rwi(2, 2) * Rwj(2, 1)) / i(0);
	Jk.coeffRef(7, 1) = 0;
	Jk.coeffRef(7, 2) = 0;
	Jk.coeffRef(8, 0) = (i(1)
			* (Rwi(0, 2) * Rwj(0, 0) + Rwi(1, 2) * Rwj(1, 0)
					+ Rwi(2, 2) * Rwj(2, 0))
			+ i(2)
					* (Rwi(0, 2) * Rwj(0, 1) + Rwi(1, 2) * Rwj(1, 1)
							+ Rwi(2, 2) * Rwj(2, 1))) / i(0);
	Jk.coeffRef(8, 1) = i(1)
			* (-Rwi(0, 2) * Rwj(0, 0) - Rwi(1, 2) * Rwj(1, 0)
					- Rwi(2, 2) * Rwj(2, 0)) / i(0);
	Jk.coeffRef(8, 2) = i(2)
			* (-Rwi(0, 2) * Rwj(0, 1) - Rwi(1, 2) * Rwj(1, 1)
					- Rwi(2, 2) * Rwj(2, 1)) / i(0);

	Jj = -Ji;

}

void reduce_jacobian_rgb::operator()(
		const tbb::blocked_range<
				tbb::concurrent_vector<std::pair<int, int> >::iterator>& r) {
	for (tbb::concurrent_vector<std::pair<int, int> >::iterator it = r.begin();
			it != r.end(); it++) {
		int i = it->first;
		int j = it->second;

		Eigen::Vector3f intrinsics = frames[i]->get_intrinsics(subsample_level);

		cv::Mat intensity_i, intensity_j;
		//frames[i]->get_i(subsample_level).convertTo(intensity_i, CV_32F);
		//frames[j]->get_i(subsample_level).convertTo(intensity_j, CV_32F);
		intensity_i = frames[i]->get_i(subsample_level);
		intensity_j = frames[j]->get_i(subsample_level);

		Eigen::Quaternionf Qij =
				frames[i]->get_pos().unit_quaternion().inverse()
						* frames[j]->get_pos().unit_quaternion();

		Eigen::Matrix3f K;
		K << intrinsics[0], 0, intrinsics[1], 0, intrinsics[0], intrinsics[2], 0, 0, 1;

		Eigen::Matrix3f H = K * Qij.matrix() * K.inverse();
		cv::Mat cvH(3, 3, CV_32F, H.data());

		cv::Mat intensity_j_warped, intensity_i_dx, intensity_i_dy;
		intensity_j_warped = cv::Mat::zeros(intensity_j.size(),
				intensity_j.type());
		cv::warpPerspective(intensity_j, intensity_j_warped, cvH.t(),
				intensity_j.size());

		intensity_i_dx = frames[i]->get_i_dx(subsample_level);
		intensity_i_dy = frames[i]->get_i_dy(subsample_level);

		//cv::imshow("intensity_i", intensity_i);
		//cv::imshow("intensity_j", intensity_j);
		//cv::imshow("intensity_j_warped", intensity_j_warped);
		//cv::imshow("intensity_i_dx", intensity_i_dx);
		//cv::imshow("intensity_i_dy", intensity_i_dy);
		//cv::waitKey();

		//cv::Sobel(intensity_i, intensity_i_dx, CV_32F, 1, 0);
		//cv::Sobel(intensity_i, intensity_i_dy, CV_32F, 0, 1);

		Eigen::Matrix<float, 9, 3> Jwi, Jwj, Jwk;

		compute_frame_jacobian(intrinsics,
				frames[i]->get_pos().unit_quaternion().matrix(),
				frames[j]->get_pos().unit_quaternion().matrix(), Jwi, Jwj, Jwk);

		for (int v = 0; v < intensity_i.rows; v++) {
			for (int u = 0; u < intensity_i.cols; u++) {
				if (intensity_j_warped.at<uint8_t>(v, u) != 0) {

					float e = (float) intensity_i.at<uint8_t>(v, u)
							- (float) intensity_j_warped.at<uint8_t>(v, u);

					float dx = intensity_i_dx.at<int16_t>(v, u);
					float dy = intensity_i_dy.at<int16_t>(v, u);
					float udx = dx * u;
					float vdx = dx * v;
					float udy = dy * u;
					float vdy = dy * v;

					float mudxmvdy = -udx - vdy;

					Eigen::Matrix<float, 1, 9> Jp;
					Jp << udx, vdx, dx, udy, vdy, dy, u * mudxmvdy, v
							* mudxmvdy, mudxmvdy;

					Eigen::Matrix<float, 1, 3> Ji, Jj, Jk;
					Ji = Jp * Jwi;
					Jj = Jp * Jwj;
					Jk = Jp * Jwk;

					//
					JtJ.block<3, 3>(i * 3, i * 3) += Ji.transpose() * Ji;
					JtJ.block<3, 3>(j * 3, j * 3) += Jj.transpose() * Jj;
					JtJ.block<3, 3>(size * 3, size * 3) += Jk.transpose() * Jk;
					// i and j
					JtJ.block<3, 3>(i * 3, j * 3) += Ji.transpose() * Jj;
					JtJ.block<3, 3>(j * 3, i * 3) += Jj.transpose() * Ji;

					// i and k
					JtJ.block<3, 3>(i * 3, size * 3) += Ji.transpose() * Jk;
					JtJ.block<3, 3>(size * 3, i * 3) += Jk.transpose() * Ji;

					// j and k
					JtJ.block<3, 3>(size * 3, j * 3) += Jk.transpose() * Jj;
					JtJ.block<3, 3>(j * 3, size * 3) += Jj.transpose() * Jk;

					// errors
					Jte.segment<3>(i * 3) += Ji.transpose() * e;
					Jte.segment<3>(j * 3) += Jj.transpose() * e;
					Jte.segment<3>(size * 3) += Jk.transpose() * e;

				}
			}

		}

	}
}

void reduce_jacobian_rgb::join(reduce_jacobian_rgb& rb) {
	JtJ += rb.JtJ;
	Jte += rb.Jte;
}

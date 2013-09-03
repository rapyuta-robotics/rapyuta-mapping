#include <reduce_jacobian_slam_3d.h>
#include <opencv2/imgproc/imgproc.hpp>

reduce_jacobian_slam_3d::reduce_jacobian_slam_3d(
		tbb::concurrent_vector<color_keyframe::Ptr> & frames, int size) :
		size(size), frames(frames) {

	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);

}

reduce_jacobian_slam_3d::reduce_jacobian_slam_3d(reduce_jacobian_slam_3d& rb,
		tbb::split) :
		size(rb.size), frames(rb.frames) {
	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);
}

// Miw = Mij * Mjw
void reduce_jacobian_slam_3d::compute_frame_jacobian(
		const Eigen::Matrix4f & Mwi, const Eigen::Matrix4f & Miw,
		Eigen::Matrix<float, 6, 6> & Ji) {

	Ji(0, 0) = Miw(0, 0);
	Ji(0, 1) = Miw(0, 1);
	Ji(0, 2) = Miw(0, 2);
	Ji(0, 3) = -Miw(0, 1) * Mwi(2, 3) + Miw(0, 2) * Mwi(1, 3);
	Ji(0, 4) = Miw(0, 0) * Mwi(2, 3) - Miw(0, 2) * Mwi(0, 3);
	Ji(0, 5) = -Miw(0, 0) * Mwi(1, 3) + Miw(0, 1) * Mwi(0, 3);
	Ji(1, 0) = Miw(1, 0);
	Ji(1, 1) = Miw(1, 1);
	Ji(1, 2) = Miw(1, 2);
	Ji(1, 3) = -Miw(1, 1) * Mwi(2, 3) + Miw(1, 2) * Mwi(1, 3);
	Ji(1, 4) = Miw(1, 0) * Mwi(2, 3) - Miw(1, 2) * Mwi(0, 3);
	Ji(1, 5) = -Miw(1, 0) * Mwi(1, 3) + Miw(1, 1) * Mwi(0, 3);
	Ji(2, 0) = Miw(2, 0);
	Ji(2, 1) = Miw(2, 1);
	Ji(2, 2) = Miw(2, 2);
	Ji(2, 3) = -Miw(2, 1) * Mwi(2, 3) + Miw(2, 2) * Mwi(1, 3);
	Ji(2, 4) = Miw(2, 0) * Mwi(2, 3) - Miw(2, 2) * Mwi(0, 3);
	Ji(2, 5) = -Miw(2, 0) * Mwi(1, 3) + Miw(2, 1) * Mwi(0, 3);
	Ji(3, 0) = 0;
	Ji(3, 1) = 0;
	Ji(3, 2) = 0;
	Ji(3, 3) = Miw(1, 1) * Mwi(2, 2) - Miw(1, 2) * Mwi(1, 2)
			- Miw(2, 1) * Mwi(2, 1) + Miw(2, 2) * Mwi(1, 1);
	Ji(3, 4) = -Miw(1, 0) * Mwi(2, 2) + Miw(1, 2) * Mwi(0, 2)
			+ Miw(2, 0) * Mwi(2, 1) - Miw(2, 2) * Mwi(0, 1);
	Ji(3, 5) = Miw(1, 0) * Mwi(1, 2) - Miw(1, 1) * Mwi(0, 2)
			- Miw(2, 0) * Mwi(1, 1) + Miw(2, 1) * Mwi(0, 1);
	Ji(4, 0) = 0;
	Ji(4, 1) = 0;
	Ji(4, 2) = 0;
	Ji(4, 3) = -Miw(0, 1) * Mwi(2, 2) + Miw(0, 2) * Mwi(1, 2)
			+ Miw(2, 1) * Mwi(2, 0) - Miw(2, 2) * Mwi(1, 0);
	Ji(4, 4) = Miw(0, 0) * Mwi(2, 2) - Miw(0, 2) * Mwi(0, 2)
			- Miw(2, 0) * Mwi(2, 0) + Miw(2, 2) * Mwi(0, 0);
	Ji(4, 5) = -Miw(0, 0) * Mwi(1, 2) + Miw(0, 1) * Mwi(0, 2)
			+ Miw(2, 0) * Mwi(1, 0) - Miw(2, 1) * Mwi(0, 0);
	Ji(5, 0) = 0;
	Ji(5, 1) = 0;
	Ji(5, 2) = 0;
	Ji(5, 3) = Miw(0, 1) * Mwi(2, 1) - Miw(0, 2) * Mwi(1, 1)
			- Miw(1, 1) * Mwi(2, 0) + Miw(1, 2) * Mwi(1, 0);
	Ji(5, 4) = -Miw(0, 0) * Mwi(2, 1) + Miw(0, 2) * Mwi(0, 1)
			+ Miw(1, 0) * Mwi(2, 0) - Miw(1, 2) * Mwi(0, 0);
	Ji(5, 5) = Miw(0, 0) * Mwi(1, 1) - Miw(0, 1) * Mwi(0, 1)
			- Miw(1, 0) * Mwi(1, 0) + Miw(1, 1) * Mwi(0, 0);

}

void reduce_jacobian_slam_3d::operator()(
		const tbb::blocked_range<
				tbb::concurrent_vector<std::pair<int, int> >::iterator>& r) {
	for (tbb::concurrent_vector<std::pair<int, int> >::iterator it = r.begin();
			it != r.end(); it++) {
		int i = it->first;
		int j = it->second;

		Sophus::SE3f Mij;
		frames[i]->estimate_relative_position(*frames[j], Mij);

		Sophus::Vector6f error = Sophus::SE3f::log(
				Mij * frames[j]->get_pos().inverse() * frames[i]->get_pos());

		Sophus::Matrix6f Ji, Jj;
		compute_frame_jacobian(frames[i]->get_pos().matrix(),
				(Mij * frames[j]->get_pos().inverse()).matrix(), Ji);

		Jj = -Ji;

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

void reduce_jacobian_slam_3d::join(reduce_jacobian_slam_3d& rb) {
	JtJ += rb.JtJ;
	Jte += rb.Jte;
}

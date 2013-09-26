#include <reduce_jacobian.h>

reduce_jacobian::reduce_jacobian(const uint8_t * intencity,
		const int16_t * intencity_dx, const int16_t * intencity_dy,
		const float * intencity_warped, const float * depth_warped,
		const Eigen::Vector3f & intrinsics,
		const Eigen::Matrix<float, 4, Eigen::Dynamic, Eigen::ColMajor> & cloud,
		int cols, int rows) :
		intencity(intencity), intencity_dx(intencity_dx), intencity_dy(
				intencity_dy), intencity_warped(intencity_warped), depth_warped(
				depth_warped), intrinsics(intrinsics), cloud(cloud), cols(cols), rows(
				rows) {

	JtJ.setZero();
	Jte.setZero();
	num_points = 0;
	error_sum = 0;

}

reduce_jacobian::reduce_jacobian(reduce_jacobian & rb, tbb::split) :
intencity(rb.intencity), intencity_dx(rb.intencity_dx), intencity_dy(
		rb.intencity_dy), intencity_warped(rb.intencity_warped), depth_warped(
		rb.depth_warped), intrinsics(rb.intrinsics), cloud(
		rb.cloud), cols(rb.cols), rows(rb.rows) {
	JtJ.setZero();
	Jte.setZero();
	num_points = 0;
	error_sum = 0;
}

void reduce_jacobian::operator()(const tbb::blocked_range<int>& range) {
	for (int i = range.begin(); i != range.end(); i++) {

		Eigen::Vector4f p = cloud.col(i);
		if (p(3) && depth_warped[i] != 0) {

			float error = (float) intencity[i] - (float) intencity_warped[i];

			Eigen::Matrix<float, 1, 2> Ji;
			Eigen::Matrix<float, 2, 6> Jw;
			Eigen::Matrix<float, 1, 6> J;
			Ji[0] = intencity_dx[i] * intrinsics[0];
			Ji[1] = intencity_dy[i] * intrinsics[0];

			compute_jacobian(p, Jw);

			J = Ji * Jw;

			JtJ += J.transpose() * J;
			Jte += J.transpose() * error;

			num_points++;
			error_sum += error * error;

		}

	}

}

void reduce_jacobian::join(reduce_jacobian& rb) {
	JtJ += rb.JtJ;
	Jte += rb.Jte;
	num_points += rb.num_points;
	error_sum += rb.error_sum;
}


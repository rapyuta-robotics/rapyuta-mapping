#include <reduce_jacobian_rgb_3d.h>
#include <opencv2/imgproc/imgproc.hpp>

reduce_jacobian_rgb_3d::reduce_jacobian_rgb_3d(
		tbb::concurrent_vector<keyframe::Ptr> & frames,
		std::vector<Eigen::Vector3f> & intrinsics_vector, int size,
		int intrinsics_size, int subsample_level) :
		size(size), intrinsics_size(intrinsics_size), subsample_level(
				subsample_level), frames(frames), intrinsics_vector(
				intrinsics_vector) {

	JtJ.setZero(size * 6 + 3 * intrinsics_size, size * 6 + 3 * intrinsics_size);
	Jte.setZero(size * 6 + 3 * intrinsics_size);

}

reduce_jacobian_rgb_3d::reduce_jacobian_rgb_3d(reduce_jacobian_rgb_3d& rb,
		tbb::split) :
		size(rb.size), intrinsics_size(rb.intrinsics_size), subsample_level(
				rb.subsample_level), frames(rb.frames), intrinsics_vector(
				rb.intrinsics_vector) {
	JtJ.setZero(size * 6 + 3 * intrinsics_size, size * 6 + 3 * intrinsics_size);
	Jte.setZero(size * 6 + 3 * intrinsics_size);
}

void reduce_jacobian_rgb_3d::compute_frame_jacobian(const Eigen::Vector3f & i,
		const Eigen::Matrix4f & Miw, const Eigen::Matrix4f & Mwj,
		Eigen::Matrix<float, 12, 6> & Ji, Eigen::Matrix<float, 12, 6> & Jj,
		Eigen::Matrix<float, 12, 3> & Jk) {

	Jj(0, 0) = 0;
	Jj(0, 1) = 0;
	Jj(0, 2) = 0;
	Jj(0, 3) = (Mwj(1, 0) * (Miw(0, 2) * i(0) + Miw(2, 2) * i(1))
			- Mwj(2, 0) * (Miw(0, 1) * i(0) + Miw(2, 1) * i(1))) / i(0);
	Jj(0, 4) = (-Mwj(0, 0) * (Miw(0, 2) * i(0) + Miw(2, 2) * i(1))
			+ Mwj(2, 0) * (Miw(0, 0) * i(0) + Miw(2, 0) * i(1))) / i(0);
	Jj(0, 5) = (Mwj(0, 0) * (Miw(0, 1) * i(0) + Miw(2, 1) * i(1))
			- Mwj(1, 0) * (Miw(0, 0) * i(0) + Miw(2, 0) * i(1))) / i(0);
	Jj(1, 0) = 0;
	Jj(1, 1) = 0;
	Jj(1, 2) = 0;
	Jj(1, 3) = (Mwj(1, 1) * (Miw(0, 2) * i(0) + Miw(2, 2) * i(1))
			- Mwj(2, 1) * (Miw(0, 1) * i(0) + Miw(2, 1) * i(1))) / i(0);
	Jj(1, 4) = (-Mwj(0, 1) * (Miw(0, 2) * i(0) + Miw(2, 2) * i(1))
			+ Mwj(2, 1) * (Miw(0, 0) * i(0) + Miw(2, 0) * i(1))) / i(0);
	Jj(1, 5) = (Mwj(0, 1) * (Miw(0, 1) * i(0) + Miw(2, 1) * i(1))
			- Mwj(1, 1) * (Miw(0, 0) * i(0) + Miw(2, 0) * i(1))) / i(0);
	Jj(2, 0) = 0;
	Jj(2, 1) = 0;
	Jj(2, 2) = 0;
	Jj(2, 3) = (Miw(0, 1) * i(0) + Miw(2, 1) * i(1))
			* (Mwj(2, 0) * i(1) / i(0) + Mwj(2, 1) * i(2) / i(0) - Mwj(2, 2))
			- (Miw(0, 2) * i(0) + Miw(2, 2) * i(1))
					* (Mwj(1, 0) * i(1) / i(0) + Mwj(1, 1) * i(2) / i(0)
							- Mwj(1, 2));
	Jj(2, 4) = -((Miw(0, 0) * i(0) + Miw(2, 0) * i(1))
			* (Mwj(2, 0) * i(1) + Mwj(2, 1) * i(2) - Mwj(2, 2) * i(0))
			- (Miw(0, 2) * i(0) + Miw(2, 2) * i(1))
					* (Mwj(0, 0) * i(1) + Mwj(0, 1) * i(2) - Mwj(0, 2) * i(0)))
			/ i(0);
	Jj(2, 5) = (Miw(0, 0) * i(0) + Miw(2, 0) * i(1))
			* (Mwj(1, 0) * i(1) / i(0) + Mwj(1, 1) * i(2) / i(0) - Mwj(1, 2))
			- (Miw(0, 1) * i(0) + Miw(2, 1) * i(1))
					* (Mwj(0, 0) * i(1) / i(0) + Mwj(0, 1) * i(2) / i(0)
							- Mwj(0, 2));
	Jj(3, 0) = Miw(0, 0) * i(0) + Miw(2, 0) * i(1);
	Jj(3, 1) = Miw(0, 1) * i(0) + Miw(2, 1) * i(1);
	Jj(3, 2) = Miw(0, 2) * i(0) + Miw(2, 2) * i(1);
	Jj(3, 3) = Mwj(1, 3) * (Miw(0, 2) * i(0) + Miw(2, 2) * i(1))
			- Mwj(2, 3) * (Miw(0, 1) * i(0) + Miw(2, 1) * i(1));
	Jj(3, 4) = -Mwj(0, 3) * (Miw(0, 2) * i(0) + Miw(2, 2) * i(1))
			+ Mwj(2, 3) * (Miw(0, 0) * i(0) + Miw(2, 0) * i(1));
	Jj(3, 5) = Mwj(0, 3) * (Miw(0, 1) * i(0) + Miw(2, 1) * i(1))
			- Mwj(1, 3) * (Miw(0, 0) * i(0) + Miw(2, 0) * i(1));
	Jj(4, 0) = 0;
	Jj(4, 1) = 0;
	Jj(4, 2) = 0;
	Jj(4, 3) = (Mwj(1, 0) * (Miw(1, 2) * i(0) + Miw(2, 2) * i(2))
			- Mwj(2, 0) * (Miw(1, 1) * i(0) + Miw(2, 1) * i(2))) / i(0);
	Jj(4, 4) = (-Mwj(0, 0) * (Miw(1, 2) * i(0) + Miw(2, 2) * i(2))
			+ Mwj(2, 0) * (Miw(1, 0) * i(0) + Miw(2, 0) * i(2))) / i(0);
	Jj(4, 5) = (Mwj(0, 0) * (Miw(1, 1) * i(0) + Miw(2, 1) * i(2))
			- Mwj(1, 0) * (Miw(1, 0) * i(0) + Miw(2, 0) * i(2))) / i(0);
	Jj(5, 0) = 0;
	Jj(5, 1) = 0;
	Jj(5, 2) = 0;
	Jj(5, 3) = (Mwj(1, 1) * (Miw(1, 2) * i(0) + Miw(2, 2) * i(2))
			- Mwj(2, 1) * (Miw(1, 1) * i(0) + Miw(2, 1) * i(2))) / i(0);
	Jj(5, 4) = (-Mwj(0, 1) * (Miw(1, 2) * i(0) + Miw(2, 2) * i(2))
			+ Mwj(2, 1) * (Miw(1, 0) * i(0) + Miw(2, 0) * i(2))) / i(0);
	Jj(5, 5) = (Mwj(0, 1) * (Miw(1, 1) * i(0) + Miw(2, 1) * i(2))
			- Mwj(1, 1) * (Miw(1, 0) * i(0) + Miw(2, 0) * i(2))) / i(0);
	Jj(6, 0) = 0;
	Jj(6, 1) = 0;
	Jj(6, 2) = 0;
	Jj(6, 3) = (Miw(1, 1) * i(0) + Miw(2, 1) * i(2))
			* (Mwj(2, 0) * i(1) / i(0) + Mwj(2, 1) * i(2) / i(0) - Mwj(2, 2))
			- (Miw(1, 2) * i(0) + Miw(2, 2) * i(2))
					* (Mwj(1, 0) * i(1) / i(0) + Mwj(1, 1) * i(2) / i(0)
							- Mwj(1, 2));
	Jj(6, 4) = -((Miw(1, 0) * i(0) + Miw(2, 0) * i(2))
			* (Mwj(2, 0) * i(1) + Mwj(2, 1) * i(2) - Mwj(2, 2) * i(0))
			- (Miw(1, 2) * i(0) + Miw(2, 2) * i(2))
					* (Mwj(0, 0) * i(1) + Mwj(0, 1) * i(2) - Mwj(0, 2) * i(0)))
			/ i(0);
	Jj(6, 5) = (Miw(1, 0) * i(0) + Miw(2, 0) * i(2))
			* (Mwj(1, 0) * i(1) / i(0) + Mwj(1, 1) * i(2) / i(0) - Mwj(1, 2))
			- (Miw(1, 1) * i(0) + Miw(2, 1) * i(2))
					* (Mwj(0, 0) * i(1) / i(0) + Mwj(0, 1) * i(2) / i(0)
							- Mwj(0, 2));
	Jj(7, 0) = Miw(1, 0) * i(0) + Miw(2, 0) * i(2);
	Jj(7, 1) = Miw(1, 1) * i(0) + Miw(2, 1) * i(2);
	Jj(7, 2) = Miw(1, 2) * i(0) + Miw(2, 2) * i(2);
	Jj(7, 3) = Mwj(1, 3) * (Miw(1, 2) * i(0) + Miw(2, 2) * i(2))
			- Mwj(2, 3) * (Miw(1, 1) * i(0) + Miw(2, 1) * i(2));
	Jj(7, 4) = -Mwj(0, 3) * (Miw(1, 2) * i(0) + Miw(2, 2) * i(2))
			+ Mwj(2, 3) * (Miw(1, 0) * i(0) + Miw(2, 0) * i(2));
	Jj(7, 5) = Mwj(0, 3) * (Miw(1, 1) * i(0) + Miw(2, 1) * i(2))
			- Mwj(1, 3) * (Miw(1, 0) * i(0) + Miw(2, 0) * i(2));
	Jj(8, 0) = 0;
	Jj(8, 1) = 0;
	Jj(8, 2) = 0;
	Jj(8, 3) = (-Miw(2, 1) * Mwj(2, 0) + Miw(2, 2) * Mwj(1, 0)) / i(0);
	Jj(8, 4) = (Miw(2, 0) * Mwj(2, 0) - Miw(2, 2) * Mwj(0, 0)) / i(0);
	Jj(8, 5) = (-Miw(2, 0) * Mwj(1, 0) + Miw(2, 1) * Mwj(0, 0)) / i(0);
	Jj(9, 0) = 0;
	Jj(9, 1) = 0;
	Jj(9, 2) = 0;
	Jj(9, 3) = (-Miw(2, 1) * Mwj(2, 1) + Miw(2, 2) * Mwj(1, 1)) / i(0);
	Jj(9, 4) = (Miw(2, 0) * Mwj(2, 1) - Miw(2, 2) * Mwj(0, 1)) / i(0);
	Jj(9, 5) = (-Miw(2, 0) * Mwj(1, 1) + Miw(2, 1) * Mwj(0, 1)) / i(0);
	Jj(10, 0) = 0;
	Jj(10, 1) = 0;
	Jj(10, 2) = 0;
	Jj(10, 3) = (Miw(2, 1)
			* (Mwj(2, 0) * i(1) + Mwj(2, 1) * i(2) - Mwj(2, 2) * i(0))
			- Miw(2, 2)
					* (Mwj(1, 0) * i(1) + Mwj(1, 1) * i(2) - Mwj(1, 2) * i(0)))
			/ i(0);
	Jj(10, 4) = (-Miw(2, 0)
			* (Mwj(2, 0) * i(1) + Mwj(2, 1) * i(2) - Mwj(2, 2) * i(0))
			+ Miw(2, 2)
					* (Mwj(0, 0) * i(1) + Mwj(0, 1) * i(2) - Mwj(0, 2) * i(0)))
			/ i(0);
	Jj(10, 5) = (Miw(2, 0)
			* (Mwj(1, 0) * i(1) + Mwj(1, 1) * i(2) - Mwj(1, 2) * i(0))
			- Miw(2, 1)
					* (Mwj(0, 0) * i(1) + Mwj(0, 1) * i(2) - Mwj(0, 2) * i(0)))
			/ i(0);
	Jj(11, 0) = Miw(2, 0);
	Jj(11, 1) = Miw(2, 1);
	Jj(11, 2) = Miw(2, 2);
	Jj(11, 3) = -Miw(2, 1) * Mwj(2, 3) + Miw(2, 2) * Mwj(1, 3);
	Jj(11, 4) = Miw(2, 0) * Mwj(2, 3) - Miw(2, 2) * Mwj(0, 3);
	Jj(11, 5) = -Miw(2, 0) * Mwj(1, 3) + Miw(2, 1) * Mwj(0, 3);
	Jk(0, 0) = i(1)
			* (-Miw(2, 0) * Mwj(0, 0) - Miw(2, 1) * Mwj(1, 0)
					- Miw(2, 2) * Mwj(2, 0)) / i(0);
	Jk(0, 1) = i(1)
			* (Miw(2, 0) * Mwj(0, 0) + Miw(2, 1) * Mwj(1, 0)
					+ Miw(2, 2) * Mwj(2, 0)) / i(0);
	Jk(0, 2) = 0;
	Jk(1, 0) = i(1)
			* (-Miw(2, 0) * Mwj(0, 1) - Miw(2, 1) * Mwj(1, 1)
					- Miw(2, 2) * Mwj(2, 1)) / i(0);
	Jk(1, 1) = i(1)
			* (Miw(2, 0) * Mwj(0, 1) + Miw(2, 1) * Mwj(1, 1)
					+ Miw(2, 2) * Mwj(2, 1)) / i(0);
	Jk(1, 2) = 0;
	Jk(2, 0) = (Miw(0, 0) * Mwj(0, 2) * i(0) * i(0)
			+ Miw(0, 1) * Mwj(1, 2) * i(0) * i(0)
			+ Miw(0, 2) * Mwj(2, 2) * i(0) * i(0)
			+ Miw(2, 0) * Mwj(0, 0) * i(1) * i(1)
			+ Miw(2, 0) * Mwj(0, 1) * i(1) * i(2)
			+ Miw(2, 1) * Mwj(1, 0) * i(1) * i(1)
			+ Miw(2, 1) * Mwj(1, 1) * i(1) * i(2)
			+ Miw(2, 2) * Mwj(2, 0) * i(1) * i(1)
			+ Miw(2, 2) * Mwj(2, 1) * i(1) * i(2)) / i(0);
	Jk(2, 1) = i(1)
			* (-Miw(0, 0) * Mwj(0, 0) * i(0) - Miw(0, 1) * Mwj(1, 0) * i(0)
					- Miw(0, 2) * Mwj(2, 0) * i(0)
					- 2 * Miw(2, 0) * Mwj(0, 0) * i(1)
					- Miw(2, 0) * Mwj(0, 1) * i(2)
					+ Miw(2, 0) * Mwj(0, 2) * i(0)
					- 2 * Miw(2, 1) * Mwj(1, 0) * i(1)
					- Miw(2, 1) * Mwj(1, 1) * i(2)
					+ Miw(2, 1) * Mwj(1, 2) * i(0)
					- 2 * Miw(2, 2) * Mwj(2, 0) * i(1)
					- Miw(2, 2) * Mwj(2, 1) * i(2)
					+ Miw(2, 2) * Mwj(2, 2) * i(0)) / i(0);
	Jk(2, 2) = -i(2)
			* (Mwj(0, 1) * (Miw(0, 0) * i(0) + Miw(2, 0) * i(1))
					+ Mwj(1, 1) * (Miw(0, 1) * i(0) + Miw(2, 1) * i(1))
					+ Mwj(2, 1) * (Miw(0, 2) * i(0) + Miw(2, 2) * i(1))) / i(0);
	Jk(3, 0) = i(0)
			* (Miw(0, 0) * Mwj(0, 3) + Miw(0, 1) * Mwj(1, 3)
					+ Miw(0, 2) * Mwj(2, 3) + Miw(0, 3));
	Jk(3, 1) = i(1)
			* (Miw(2, 0) * Mwj(0, 3) + Miw(2, 1) * Mwj(1, 3)
					+ Miw(2, 2) * Mwj(2, 3) + Miw(2, 3));
	Jk(3, 2) = 0;
	Jk(4, 0) = i(2)
			* (-Miw(2, 0) * Mwj(0, 0) - Miw(2, 1) * Mwj(1, 0)
					- Miw(2, 2) * Mwj(2, 0)) / i(0);
	Jk(4, 1) = 0;
	Jk(4, 2) = i(2)
			* (Miw(2, 0) * Mwj(0, 0) + Miw(2, 1) * Mwj(1, 0)
					+ Miw(2, 2) * Mwj(2, 0)) / i(0);
	Jk(5, 0) = i(2)
			* (-Miw(2, 0) * Mwj(0, 1) - Miw(2, 1) * Mwj(1, 1)
					- Miw(2, 2) * Mwj(2, 1)) / i(0);
	Jk(5, 1) = 0;
	Jk(5, 2) = i(2)
			* (Miw(2, 0) * Mwj(0, 1) + Miw(2, 1) * Mwj(1, 1)
					+ Miw(2, 2) * Mwj(2, 1)) / i(0);
	Jk(6, 0) = (Miw(1, 0) * Mwj(0, 2) * i(0) * i(0)
			+ Miw(1, 1) * Mwj(1, 2) * i(0) * i(0)
			+ Miw(1, 2) * Mwj(2, 2) * i(0) * i(0)
			+ Miw(2, 0) * Mwj(0, 0) * i(1) * i(2)
			+ Miw(2, 0) * Mwj(0, 1) * i(2) * i(2)
			+ Miw(2, 1) * Mwj(1, 0) * i(1) * i(2)
			+ Miw(2, 1) * Mwj(1, 1) * i(2) * i(2)
			+ Miw(2, 2) * Mwj(2, 0) * i(1) * i(2)
			+ Miw(2, 2) * Mwj(2, 1) * i(2) * i(2)) / i(0);
	Jk(6, 1) = -i(1)
			* (Mwj(0, 0) * (Miw(1, 0) * i(0) + Miw(2, 0) * i(2))
					+ Mwj(1, 0) * (Miw(1, 1) * i(0) + Miw(2, 1) * i(2))
					+ Mwj(2, 0) * (Miw(1, 2) * i(0) + Miw(2, 2) * i(2))) / i(0);
	Jk(6, 2) = i(2)
			* (-Miw(1, 0) * Mwj(0, 1) * i(0) - Miw(1, 1) * Mwj(1, 1) * i(0)
					- Miw(1, 2) * Mwj(2, 1) * i(0)
					- Miw(2, 0) * Mwj(0, 0) * i(1)
					- 2 * Miw(2, 0) * Mwj(0, 1) * i(2)
					+ Miw(2, 0) * Mwj(0, 2) * i(0)
					- Miw(2, 1) * Mwj(1, 0) * i(1)
					- 2 * Miw(2, 1) * Mwj(1, 1) * i(2)
					+ Miw(2, 1) * Mwj(1, 2) * i(0)
					- Miw(2, 2) * Mwj(2, 0) * i(1)
					- 2 * Miw(2, 2) * Mwj(2, 1) * i(2)
					+ Miw(2, 2) * Mwj(2, 2) * i(0)) / i(0);
	Jk(7, 0) = i(0)
			* (Miw(1, 0) * Mwj(0, 3) + Miw(1, 1) * Mwj(1, 3)
					+ Miw(1, 2) * Mwj(2, 3) + Miw(1, 3));
	Jk(7, 1) = 0;
	Jk(7, 2) = i(2)
			* (Miw(2, 0) * Mwj(0, 3) + Miw(2, 1) * Mwj(1, 3)
					+ Miw(2, 2) * Mwj(2, 3) + Miw(2, 3));
	Jk(8, 0) = -(Miw(2, 0) * Mwj(0, 0) + Miw(2, 1) * Mwj(1, 0)
			+ Miw(2, 2) * Mwj(2, 0)) / i(0);
	Jk(8, 1) = 0;
	Jk(8, 2) = 0;
	Jk(9, 0) = -(Miw(2, 0) * Mwj(0, 1) + Miw(2, 1) * Mwj(1, 1)
			+ Miw(2, 2) * Mwj(2, 1)) / i(0);
	Jk(9, 1) = 0;
	Jk(9, 2) = 0;
	Jk(10, 0) = (i(1)
			* (Miw(2, 0) * Mwj(0, 0) + Miw(2, 1) * Mwj(1, 0)
					+ Miw(2, 2) * Mwj(2, 0))
			+ i(2)
					* (Miw(2, 0) * Mwj(0, 1) + Miw(2, 1) * Mwj(1, 1)
							+ Miw(2, 2) * Mwj(2, 1))) / i(0);
	Jk(10, 1) = i(1)
			* (-Miw(2, 0) * Mwj(0, 0) - Miw(2, 1) * Mwj(1, 0)
					- Miw(2, 2) * Mwj(2, 0)) / i(0);
	Jk(10, 2) = i(2)
			* (-Miw(2, 0) * Mwj(0, 1) - Miw(2, 1) * Mwj(1, 1)
					- Miw(2, 2) * Mwj(2, 1)) / i(0);
	Jk(11, 0) = 0;
	Jk(11, 1) = 0;
	Jk(11, 2) = 0;

	Ji = -Jj;

}

void reduce_jacobian_rgb_3d::warpImage(int i, int j, cv::Mat & intensity_i,
		cv::Mat & intensity_j, cv::Mat & intensity_j_warped,
		cv::Mat & idx_j_warped) {

	int subsample_scale = 1 << subsample_level;

	Eigen::Vector3f intrinsics_i = frames[i]->get_subsampled_intrinsics(
			subsample_level);
	Eigen::Vector3f intrinsics_j = frames[j]->get_subsampled_intrinsics(
			subsample_level);
	intensity_i = frames[i]->get_subsampled_intencity(subsample_level);
	intensity_j = frames[j]->get_subsampled_intencity(subsample_level);
	cv::Mat & depth_j = frames[j]->depth;

	Sophus::SE3f Mij = frames[i]->get_position().inverse()
			* frames[j]->get_position();

	cv::Mat depth_j_warped = cv::Mat(intensity_j.size(), depth_j.type());
	intensity_j_warped = cv::Mat(intensity_j.size(), intensity_j.type());
	idx_j_warped = cv::Mat(intensity_j.size(), CV_16SC2);
	depth_j_warped.setTo(0);
	intensity_j_warped.setTo(0);
	idx_j_warped.setTo(cv::Vec2s(-1, -1));

	for (int v = 0; v < intensity_j.rows; v++) {
		for (int u = 0; u < intensity_j.cols; u++) {
			int depth_u = (u + 0.5) * subsample_scale;
			int depth_v = (v + 0.5) * subsample_scale;

			if (depth_j.at<unsigned short>(depth_v, depth_u) != 0) {
				pcl::PointXYZ p;
				p.z = depth_j.at<unsigned short>(depth_v, depth_u) / 1000.0f;
				p.x = (u - intrinsics_j[1]) * p.z / intrinsics_j[0];
				p.y = (v - intrinsics_j[2]) * p.z / intrinsics_j[0];

				p.getVector3fMap() = Mij * p.getVector3fMap();
				int u_i = p.x * intrinsics_i[0] / p.z + intrinsics_i[1];
				int v_i = p.y * intrinsics_i[0] / p.z + intrinsics_i[2];

				if (u_i >= 0 && u_i < intensity_j_warped.cols && v_i >= 0
						&& v_i < intensity_j_warped.rows) {
					unsigned short d = p.z * 1000.0f;

					if (depth_j_warped.at<unsigned short>(v_i, u_i) == 0
							|| depth_j_warped.at<unsigned short>(v_i, u_i)
									> d) {
						//std::cerr << "Writing to depth_j_warped (" << v_i << "," << u_i << ")" << std::endl;
						depth_j_warped.at<unsigned short>(v_i, u_i) = d;

						//std::cerr << "Writing to intensity_j_warped (" << v_i << "," << u_i << ") from intensity_j (" << v << "," << u << ")" << std::endl;
						intensity_j_warped.at<float>(v_i, u_i) = intensity_j.at<
								float>(v, u);

						//std::cerr << "Writing to idx_j_warped (" << v_i << "," << u_i << ") value (" << v << "," << u << ")" << std::endl;
						idx_j_warped.at<cv::Vec2s>(v_i, u_i) = cv::Vec2s(v, u);
					}
				}

			}
		}
	}


}

void reduce_jacobian_rgb_3d::operator()(
		const tbb::blocked_range<
				tbb::concurrent_vector<std::pair<int, int> >::iterator>& r) {
	for (tbb::concurrent_vector<std::pair<int, int> >::iterator it = r.begin();
			it != r.end(); it++) {
		int i = it->first;
		int j = it->second;

		cv::Mat intensity_i, intensity_j, intensity_j_warped, idx_j_warped,
				intensity_i_dx, intensity_i_dy;

		warpImage(i, j, intensity_i, intensity_j, intensity_j_warped,
				idx_j_warped);

		//std::cerr << "intensity_i " << intensity_i.size() << " " << intensity_i.type() << std::endl;
		//std::cerr << "intensity_j " << intensity_j.size() << " " << intensity_j.type() << std::endl;
		//std::cerr << "intensity_j_warped " << intensity_j_warped.size() << " " << intensity_j_warped.type() << std::endl;

		//cv::imshow("intensity_i", intensity_i);
		//cv::imshow("intensity_j", intensity_j);
		//cv::imshow("intensity_j_warped", intensity_j_warped);
		//cv::waitKey();


		intensity_i_dx = cv::Mat(intensity_i.size(), intensity_i.type());
		intensity_i_dy = cv::Mat(intensity_i.size(), intensity_i.type());

		cv::Sobel(intensity_i, intensity_i_dx, CV_32F, 1, 0);
		cv::Sobel(intensity_i, intensity_i_dy, CV_32F, 0, 1);

		cv::Mat error = intensity_i - intensity_j_warped;

		int ki = frames[i]->get_intrinsics_idx();
		int kj = frames[j]->get_intrinsics_idx();

		if (ki == kj) {

			Eigen::Matrix<float, 12, 6> Jwi, Jwj;
			Eigen::Matrix<float, 12, 3> Jwk;

			Eigen::Vector3f intrinsics = frames[i]->get_subsampled_intrinsics(
					subsample_level);

			compute_frame_jacobian(intrinsics,
					frames[i]->get_position().inverse().matrix(),
					frames[j]->get_position().matrix(), Jwi, Jwj, Jwk);

			for (int vi = 0; vi < intensity_i.rows; vi++) {
				for (int ui = 0; ui < intensity_i.cols; ui++) {

					cv::Vec2s uvj = idx_j_warped.at<cv::Vec2s>(vi, ui);
					if (uvj[0] >= 0 && uvj[1] >= 0) {

						float e = error.at<float>(vi, ui);

						float dx_i = intensity_i_dx.at<float>(vi, ui);
						float dy_i = intensity_i_dy.at<float>(vi, ui);
						float vj = uvj[0];
						float uj = uvj[1];

						int subsample_scale = 1 << subsample_level;
						int depth_u = (uj + 0.5) * subsample_scale;
						int depth_v = (vj + 0.5) * subsample_scale;
						float dj = frames[j]->depth.at<unsigned short>(depth_v, depth_u) / 1000.0f;

						Eigen::Matrix<float, 1, 12> Jp;
						Jp << dj * dx_i * uj, dj * dx_i * vj, dj * dx_i, dx_i, dj
								* dy_i * uj, dj * dy_i * vj, dj * dy_i, dy_i, dj
								* uj * (-dx_i * ui - dy_i * vi), dj * vj
								* (-dx_i * ui - dy_i * vi), dj
								* (-dx_i * ui - dy_i * vi), -dx_i * ui
								- dy_i * vi;

						Eigen::Matrix<float, 1, 6> Ji, Jj;
						Eigen::Matrix<float, 1, 3> Jk;
						Ji = Jp * Jwi;
						Jj = Jp * Jwj;
						Jk = Jp * Jwk;

						int i_idx = i * 6;
						int j_idx = j * 6;
						int k_idx = size * 6 + ki*3;

						//
						JtJ.block<6, 6>(i_idx, i_idx) += Ji.transpose() * Ji;
						JtJ.block<6, 6>(j_idx, j_idx) += Jj.transpose() * Jj;
						JtJ.block<3, 3>(k_idx, k_idx) +=
								Jk.transpose() * Jk;
						// i and j
						JtJ.block<6, 6>(i_idx, j_idx) += Ji.transpose() * Jj;
						JtJ.block<6, 6>(j_idx, i_idx) += Jj.transpose() * Ji;

						// i and k
						JtJ.block<6, 3>(i_idx, k_idx) += Ji.transpose()
								* Jk;
						JtJ.block<3, 6>(k_idx, i_idx) += Jk.transpose()
								* Ji;

						// j and k
						JtJ.block<3, 6>(k_idx, j_idx) += Jk.transpose()
								* Jj;
						JtJ.block<6, 3>(j_idx, k_idx) += Jj.transpose()
								* Jk;

						// errors
						Jte.segment<6>(i_idx) += Ji.transpose() * e;
						Jte.segment<6>(j_idx) += Jj.transpose() * e;
						Jte.segment<3>(k_idx) += Jk.transpose() * e;

					}

				}
			}
		}


	}
}

void reduce_jacobian_rgb_3d::join(reduce_jacobian_rgb_3d& rb) {
	JtJ += rb.JtJ;
	Jte += rb.Jte;
}

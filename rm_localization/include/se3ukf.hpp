/*
 * se3ukf.hpp
 *
 *  Created on: Oct 13, 2013
 *      Author: vsu
 */

#ifndef SE3UKF_HPP_
#define SE3UKF_HPP_

#include <sophus/se3.hpp>
#include <iostream>

template<typename _Scalar>
class SE3UKF {
public:

	typedef Sophus::SE3Group<_Scalar> SE3Type;

	typedef Eigen::Matrix<_Scalar, 3, 1> Vector3;
	typedef Eigen::Matrix<_Scalar, 6, 1> Vector6;
	typedef Eigen::Matrix<_Scalar, 12, 1> Vector12;

	typedef Eigen::Matrix<_Scalar, 6, 6> Matrix6;
	typedef Eigen::Matrix<_Scalar, 12, 12> Matrix12;
	typedef Eigen::Matrix<_Scalar, 12, 6> Matrix12_6;

	SE3Type pose;
	Vector6 velocity;
	Matrix12 covariance, model_noise;

	SE3Type sigma_pose[25];
	Vector6 sigma_velocity[25];

	unsigned int iteration;

	void print_if_too_large(const Vector3 & update, const std::string & name) {
		if (update.array().abs().maxCoeff() > M_PI / 4) {
			std::cout << "[" << iteration << "]" << name
					<< " has too large elements " << std::endl
					<< update.transpose() << std::endl;
		}
	}

	void compute_sigma_points(const Vector12 & delta = Vector12::Zero()) {

		print_if_too_large(delta.template segment<3>(3), "Delta");

		Eigen::LLT<Matrix12> llt_of_covariance(covariance);
		assert(llt_of_covariance.info() == Eigen::Success);
		Matrix12 L = llt_of_covariance.matrixL();

		sigma_pose[0] = pose * SE3Type::exp(delta.template head<6>());
		sigma_velocity[0] = velocity + delta.template tail<6>();

		for (int i = 0; i < 12; i++) {
			Vector12 eps = L.col(i);

			sigma_pose[1 + i] = pose
					* SE3Type::exp(
							delta.template head<6>() + eps.template head<6>());
			sigma_velocity[1 + i] = velocity + delta.template tail<6>()
					+ eps.template tail<6>();

			sigma_pose[13 + i] = pose
					* SE3Type::exp(
							delta.template head<6>() - eps.template head<6>());
			sigma_velocity[13 + i] = velocity + delta.template tail<6>()
					- eps.template tail<6>();

			print_if_too_large(eps.template segment<3>(3), "Epsilon");
			print_if_too_large(
					delta.template segment<3>(3) + eps.template segment<3>(3),
					"Delta + Epsilon");
			print_if_too_large(
					delta.template segment<3>(3) - eps.template segment<3>(3),
					"Delta - Epsilon");

		}

	}

	void compute_mean(SE3Type & mean_pose, Vector6 & mean_velocity) {

		mean_velocity.setZero();
		for (int i = 0; i < 25; i++) {
			mean_velocity += sigma_velocity[i];
		}
		mean_velocity /= 25.0f;

		mean_pose = sigma_pose[0];
		Vector6 delta;

		const static int max_iterations = 1000;
		int iterations = 0;

		do {
			delta.setZero();

			for (int i = 0; i < 25; i++) {
				delta += SE3Type::log(mean_pose.inverse() * sigma_pose[i]);
			}
			delta /= 25.0f;

			mean_pose *= SE3Type::exp(delta);
			iterations++;

		} while (delta.array().abs().maxCoeff()
				> Sophus::SophusConstants<_Scalar>::epsilon()
				&& iterations < max_iterations);

	}

	void compute_mean_and_covariance() {
		compute_mean(pose, velocity);
		covariance.setZero();

		for (int i = 0; i < 25; i++) {
			Vector12 eps;

			eps.template head<6>() = SE3Type::log(
					pose.inverse() * sigma_pose[i]);
			eps.template tail<6>() = sigma_velocity[i] - velocity;

			covariance += eps * eps.transpose();
		}
		covariance /= 2;

	}

public:

	SE3UKF(const SE3Type & initial_pose, const Vector6 & initial_velocity,
			const Matrix12 & initial_covariance) {

		iteration = 0;
		pose = initial_pose;
		velocity = initial_velocity;
		covariance = initial_covariance;

		model_noise.setIdentity();
		model_noise *= 0.01;

	}

	void predict(_Scalar dt) {

		compute_sigma_points();

		for (int i = 0; i < 25; i++) {
			sigma_pose[i] *= SE3Type::exp(dt * sigma_velocity[i]);
		}

		compute_mean_and_covariance();
		covariance += model_noise * dt;

	}

	void measure(const SE3Type & measured_pose,
			const Matrix6 & measurement_noise) {

		Matrix6 S = covariance.template topLeftCorner<6, 6>();
		Matrix12_6 Sigma = covariance.template topLeftCorner<12, 6>();

		S += measurement_noise;

		Matrix12_6 K;
		K = Sigma * S.inverse();

		Vector12 delta;
		delta = K * SE3Type::log(pose.inverse() * measured_pose);

		covariance -= K * S * K.transpose();

		//temp. insure covariance is symmetric.
		covariance = (covariance + covariance.transpose()) / 2;

		compute_sigma_points(delta);
		compute_mean_and_covariance();

		iteration++;

	}

	SE3Type get_pose() const {
		return pose;
	}

	Vector6 get_velocity() const {
		return velocity;
	}

	Matrix12 get_covariance() const {
		return covariance;
	}

	void test_sigma_points() {
		SE3Type pose1 = pose;
		Vector6 velocity1 = velocity;
		Matrix12 covariance1 = covariance;

		compute_sigma_points();
		compute_mean_and_covariance();

		SE3Type pose2 = pose;
		Vector6 velocity2 = velocity;
		Matrix12 covariance2 = covariance;

		if ((covariance1 - covariance2).array().abs().maxCoeff() > 1e-5) {
			std::cerr << "[" << iteration << "]" << "Covariance miscalculated\n"
					<< covariance1 << "\n\n" << covariance2;
			assert(false);
		}

		if ((velocity1 - velocity2).array().abs().maxCoeff() > 1e-5) {
			std::cerr << "[" << iteration << "]" << "Velocity miscalculated\n"
					<< velocity1 << "\n\n" << velocity2;
			assert(false);
		}

		//if (pose1 != pose2) {
		//	std::cerr << "[" << iteration << "]" << "Pose miscalculated\n";
		//	assert(false);
		//}

	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef SE3UKF<float> SE3UKFf;
typedef SE3UKF<double> SE3UKFd;

#endif /* SE3UKF_HPP_ */

#include <se3ukf.hpp>
#include <gtest/gtest.h>

TEST(SigmaPointsTest, floatTest) {

	Sophus::SE3f init_pose, pose;
	Sophus::SE3f::Tangent init_velocity, velocity;
	Eigen::Matrix<float, 12, 12> init_cov, cov;
	Eigen::Quaternionf q;
	q.coeffs().setRandom();
	q.normalize();

	init_pose.setQuaternion(q);
	init_pose.translation().setRandom();
	init_velocity.setRandom();

	// Get random symmetric matrix
	init_cov.setRandom();
	init_cov *= init_cov.transpose();
	init_cov /= 10 * init_cov.array().abs().maxCoeff();

	SE3UKFf f(init_pose, init_velocity, init_cov);
	f.compute_sigma_points();
	f.compute_mean_and_covariance();

	pose = f.get_pose();
	velocity = f.get_velocity();
	cov = f.get_covariance();

	EXPECT_LE(
			(pose.translation() - init_pose.translation()).array().abs().maxCoeff(),
			Sophus::SophusConstants<float>::epsilon());

	EXPECT_LE(
			(pose.unit_quaternion().coeffs() - init_pose.unit_quaternion().coeffs()).array().abs().maxCoeff(),
			Sophus::SophusConstants<float>::epsilon());

	EXPECT_LE( (init_velocity - velocity).array().abs().maxCoeff(),
			Sophus::SophusConstants<float>::epsilon());

	EXPECT_LE( (init_cov - cov).array().abs().maxCoeff(),
			Sophus::SophusConstants<float>::epsilon());
}

TEST(SigmaPointsTest, doubleTest) {

	Sophus::SE3d init_pose, pose;
	Sophus::SE3d::Tangent init_velocity, velocity;
	Eigen::Matrix<double, 12, 12> init_cov, cov;
	Eigen::Quaterniond q;
	q.coeffs().setRandom();
	q.normalize();

	init_pose.setQuaternion(q);
	init_pose.translation().setRandom();
	init_velocity.setRandom();

	// Get random symmetric matrix
	init_cov.setRandom();
	init_cov *= init_cov.transpose();
	init_cov /= 10 * init_cov.array().abs().maxCoeff();

	SE3UKFd f(init_pose, init_velocity, init_cov);
	f.compute_sigma_points();
	f.compute_mean_and_covariance();

	pose = f.get_pose();
	velocity = f.get_velocity();
	cov = f.get_covariance();

	EXPECT_LE(
			(pose.translation() - init_pose.translation()).array().abs().maxCoeff(),
			Sophus::SophusConstants<double>::epsilon());

	EXPECT_LE(
			(pose.unit_quaternion().coeffs() - init_pose.unit_quaternion().coeffs()).array().abs().maxCoeff(),
			Sophus::SophusConstants<double>::epsilon());

	EXPECT_LE( (init_velocity - velocity).array().abs().maxCoeff(),
			Sophus::SophusConstants<double>::epsilon());

	EXPECT_LE( (init_cov - cov).array().abs().maxCoeff(),
			Sophus::SophusConstants<double>::epsilon());
}

TEST(VelocityEstimationTest, floatTest) {

	Sophus::SE3f true_pose, pose;
	Sophus::SE3f::Tangent true_velocity, velocity;
	Eigen::Matrix<float, 12, 12> init_cov;
	Sophus::Matrix6f measurement_noise;
	Eigen::Quaternionf q;
	q.coeffs().setRandom();
	q.normalize();

	true_pose.setQuaternion(q);
	true_pose.translation().setRandom();
	true_velocity.setRandom();

	true_velocity /= 10 * true_velocity.array().abs().maxCoeff();

	std::cerr << "Veclocity " << true_velocity.transpose() << std::endl;

	init_cov.setIdentity();
	init_cov *= 0.01;

	measurement_noise.setIdentity();
	measurement_noise *= 0.01;

	SE3UKFf f(true_pose, Sophus::SE3f::Tangent::Zero(), init_cov);

	float dt = 1.0 / 100;

	for (int i = 0; i < 10000; i++) {
		true_pose *= Sophus::SE3f::exp(true_velocity * dt);

		f.predict(dt);
		f.measure(true_pose, measurement_noise);
	}

	velocity = f.get_velocity();
	pose = f.get_pose();

	EXPECT_LE( (true_velocity - velocity).array().abs().maxCoeff(), 1e-4);

	EXPECT_LE(
			(pose.translation() - true_pose.translation()).array().abs().maxCoeff(),
			Sophus::SophusConstants<float>::epsilon());

	EXPECT_LE(
			(pose.unit_quaternion().coeffs() - true_pose.unit_quaternion().coeffs()).array().abs().maxCoeff(),
			Sophus::SophusConstants<float>::epsilon());

}

TEST(VelocityEstimationTest, doubleTest) {

	Sophus::SE3d true_pose, pose;
	Sophus::SE3d::Tangent true_velocity, velocity;
	Eigen::Matrix<double, 12, 12> init_cov;
	Sophus::Matrix6d measurement_noise;
	Eigen::Quaterniond q;
	q.coeffs().setRandom();
	q.normalize();

	true_pose.setQuaternion(q);
	true_pose.translation().setRandom();
	true_velocity.setRandom();

	true_velocity /= 10 * true_velocity.array().abs().maxCoeff();

	std::cerr << "Veclocity " << true_velocity.transpose() << std::endl;

	init_cov.setIdentity();
	init_cov *= 0.01;

	measurement_noise.setIdentity();
	measurement_noise *= 0.01;

	SE3UKFd f(true_pose, Sophus::SE3d::Tangent::Zero(), init_cov);

	float dt = 1.0 / 100;

	for (int i = 0; i < 10000; i++) {
		true_pose *= Sophus::SE3d::exp(true_velocity * dt);

		f.predict(dt);
		f.measure(true_pose, measurement_noise);
	}

	velocity = f.get_velocity();
	pose = f.get_pose();

	EXPECT_LE( (true_velocity - velocity).array().abs().maxCoeff(), 1e-6);

	EXPECT_LE(
			(pose.translation() - true_pose.translation()).array().abs().maxCoeff(),
			Sophus::SophusConstants<double>::epsilon());

	EXPECT_LE(
			(pose.unit_quaternion().coeffs() - true_pose.unit_quaternion().coeffs()).array().abs().maxCoeff(),
			Sophus::SophusConstants<double>::epsilon());

}

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

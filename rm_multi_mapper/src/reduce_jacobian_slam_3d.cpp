#include <reduce_jacobian_slam_3d.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <keyframe_map.h>

reduce_jacobian_slam_3d::reduce_jacobian_slam_3d(
		tbb::concurrent_vector<color_keyframe::Ptr> & frames, int size) :
		size(size), frames(frames) {

	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);

	icp.setMaxCorrespondenceDistance(0.5);
	boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
	icp.setTransformationEstimation(point_to_plane);

}

reduce_jacobian_slam_3d::reduce_jacobian_slam_3d(reduce_jacobian_slam_3d& rb,
		tbb::split) :
		size(rb.size), frames(rb.frames) {
	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);

	icp.setMaxCorrespondenceDistance(0.5);
	boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
	icp.setTransformationEstimation(point_to_plane);
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

void reduce_jacobian_slam_3d::add_icp_measurement(int i, int j) {

	Sophus::SE3f Mij = frames[i]->get_pos().inverse() * frames[j]->get_pos();

	pcl::PointCloud<pcl::PointNormal>::Ptr Final(
			new pcl::PointCloud<pcl::PointNormal>);

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_j =
			frames[j]->get_pointcloud_with_normals(8, false);
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_i =
			frames[i]->get_pointcloud_with_normals(8, false);

	pcl::transformPointCloudWithNormals(*cloud_j, *cloud_j, Mij.matrix());

	/*static pcl::visualization::PCLVisualizer vis;
	 {
	 vis.removeAllPointClouds();
	 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> color_i(
	 cloud_i, 255, 0, 0);
	 vis.addPointCloud<pcl::PointNormal>(cloud_i, color_i, "cloud_i");
	 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> color_j(
	 cloud_j, 0, 255, 0);
	 vis.addPointCloud<pcl::PointNormal>(cloud_j, color_j, "cloud_j");
	 vis.spin();
	 }*/

	icp.setInputCloud(cloud_j);
	icp.setInputTarget(cloud_i);
	icp.align(*Final);
	if (icp.hasConverged()) {

		//std::cout << "has converged:" << icp.hasConverged() << " score: "
		//		<< icp.getFitnessScore() << std::endl;
		//std::cout << icp.getFinalTransformation() << std::endl;

		Eigen::Affine3f tm(icp.getFinalTransformation());
		Mij = Sophus::SE3f(tm.rotation(), tm.translation()) * Mij;

		/*{
		 Final->clear();
		 pcl::transformPointCloudWithNormals(*frames[j]->get_pointcloud_with_normals(4, false), *Final, Mij.matrix());

		 vis.removeAllPointClouds();
		 pcl::visualization::PointCloudColorHandlerCustom<
		 pcl::PointNormal> color_i(cloud_i, 255, 0, 0);
		 vis.addPointCloud<pcl::PointNormal>(cloud_i, color_i,
		 "cloud_i");
		 pcl::visualization::PointCloudColorHandlerCustom<
		 pcl::PointNormal> color_j(Final, 0, 0, 255);
		 vis.addPointCloud<pcl::PointNormal>(Final, color_j, "cloud_j");
		 vis.spin();
		 }*/

		Sophus::SE3f error_transform = Mij * frames[j]->get_pos().inverse()
				* frames[i]->get_pos();

		Eigen::Matrix4f e = error_transform.matrix();
		Sophus::Vector6f error;
		error << e(0, 3), e(1, 3), e(2, 3), -e(1, 2) + e(2, 1), e(0, 2)
				- e(2, 0), -e(0, 1) + e(1, 0);

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

void reduce_jacobian_slam_3d::add_rgbd_measurement(int i, int j) {

	Sophus::SE3f Mij;

	if (frames[i]->estimate_relative_position(*frames[j], Mij)) {

		Sophus::SE3f error_transform = Mij * frames[j]->get_pos().inverse()
				* frames[i]->get_pos();

		Eigen::Matrix4f e = error_transform.matrix();
		Sophus::Vector6f error;
		error << e(0, 3), e(1, 3), e(2, 3), -e(1, 2) + e(2, 1), e(0, 2)
				- e(2, 0), -e(0, 1) + e(1, 0);

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

void compute_floor_jacobian(float nx, float ny, float nz, float y, float x,
		Eigen::Matrix<float, 3, 6> & Ji) {
	Ji(0, 0) = 0;
	Ji(0, 1) = 0;
	Ji(0, 2) = 0;
	Ji(0, 3) = 0;
	Ji(0, 4) = nz;
	Ji(0, 5) = -ny;
	Ji(1, 0) = 0;
	Ji(1, 1) = 0;
	Ji(1, 2) = 0;
	Ji(1, 3) = -nz;
	Ji(1, 4) = 0;
	Ji(1, 5) = nx;
	Ji(2, 0) = 0;
	Ji(2, 1) = 0;
	Ji(2, 2) = 1;
	Ji(2, 3) = y;
	Ji(2, 4) = -x;
	Ji(2, 5) = 0;
}

void reduce_jacobian_slam_3d::add_floor_measurement(int i) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = frames[i]->get_pointcloud(8,
			true, -0.2, 0.2);

	if (cloud->size() < 30)
		return;

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.05);
	seg.setProbability(0.99);
	seg.setMaxIterations(5000);

	seg.setInputCloud(cloud);
	seg.segment(*inliers, *coefficients);

	if (inliers->indices.size() < 100)
		return;

	//std::cerr << "Model coefficients: " << coefficients->values[0] << " "
	//		<< coefficients->values[1] << " " << coefficients->values[2] << " "
	//		<< coefficients->values[3] << " Num inliers "
	//		<< inliers->indices.size() << std::endl;

	if (coefficients->values[2] < 0.9)
		return;

	Eigen::Matrix<float, 3, 6> Ji;

	Eigen::Vector3f pos = frames[i]->get_pos().translation();

	compute_floor_jacobian(coefficients->values[0], coefficients->values[1],
			coefficients->values[2], pos(0), pos(1), Ji);

	Sophus::Vector3f error;
	error << coefficients->values[0], coefficients->values[1], -coefficients->values[3];

	JtJ.block<6, 6>(i * 6, i * 6) += Ji.transpose() * Ji;
	Jte.segment<6>(i * 6) += Ji.transpose() * error;

}

void reduce_jacobian_slam_3d::operator()(
		const tbb::blocked_range<
				tbb::concurrent_vector<std::pair<int, int> >::iterator>& r) {
	for (tbb::concurrent_vector<std::pair<int, int> >::iterator it = r.begin();
			it != r.end(); it++) {
		int i = it->first;
		int j = it->second;

		if (j >= 0) {
			//add_icp_measurement(i, j);
			add_rgbd_measurement(i, j);
			//add_ransac_measurement(i,j);
		} else {
			//add_floor_measurement(i);
		}

		//ROS_INFO_STREAM(
		//		"Computing transformation between frames " << i << " and " << j);
	}

}

void reduce_jacobian_slam_3d::join(reduce_jacobian_slam_3d& rb) {
	JtJ += rb.JtJ;
	Jte += rb.Jte;
}

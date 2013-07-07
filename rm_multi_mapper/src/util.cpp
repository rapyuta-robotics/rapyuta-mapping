/*
 * util.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <util.h>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/flann_search.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/pcd_io.h>

keypoint_map::keypoint_map(cv::Mat & rgb, cv::Mat & depth) {
	de = new cv::SurfDescriptorExtractor;
	dm = new cv::FlannBasedMatcher;
	fd = new cv::SurfFeatureDetector;
	fd->setInt("hessianThreshold", 400);
	fd->setBool("extended", true);
	fd->setBool("upright", true);

	intrinsics << 525.0, 525.0, 319.5, 239.5;

	std::vector<cv::KeyPoint> keypoints;

	compute_features(rgb, depth, intrinsics, fd, de, keypoints, keypoints3d,
			descriptors);

	for (int keypoint_id = 0; keypoint_id < keypoints.size(); keypoint_id++) {
		observation o;
		o.cam_id = 0;
		o.point_id = keypoint_id;
		o.coord << keypoints[keypoint_id].pt.x, keypoints[keypoint_id].pt.y;

		observations.push_back(o);
		weights.push_back(1.0f);
	}

	camera_positions.push_back(Eigen::Affine3f::Identity());
	rgb_imgs.push_back(rgb);
	depth_imgs.push_back(depth);

}

bool keypoint_map::merge_keypoint_map(const keypoint_map & other) {

	std::vector<cv::DMatch> matches;
	dm->match(other.descriptors, descriptors, matches);

	Eigen::Affine3f transform;
	std::vector<bool> inliers;

	bool res = estimate_transform_ransac(other.keypoints3d, keypoints3d,
			matches, 3000, 0.03 * 0.03, 20, transform, inliers);

	if (!res)
		return false;

	size_t current_camera_positions_size = camera_positions.size();

	for (size_t i = 0; i < other.camera_positions.size(); i++) {
		camera_positions.push_back(transform * other.camera_positions[i]);
		rgb_imgs.push_back(other.rgb_imgs[i]);
		depth_imgs.push_back(other.depth_imgs[i]);
	}

	for (size_t i = 0; i < matches.size(); i++) {

		if (inliers[i]) {
			descriptors.row(matches[i].trainIdx) = (descriptors.row(
					matches[i].trainIdx) * weights[matches[i].trainIdx]
					+ other.descriptors.row(matches[i].queryIdx)
							* other.weights[matches[i].queryIdx])
					/ (weights[matches[i].trainIdx]
							+ other.weights[matches[i].queryIdx]);

			weights[matches[i].trainIdx] += other.weights[matches[i].queryIdx];

			for (size_t j = 0; j < other.observations.size(); j++) {
				if (other.observations[j].point_id == matches[i].queryIdx) {

					observation o = other.observations[j];
					o.point_id = matches[i].trainIdx;
					o.cam_id += current_camera_positions_size;
					observations.push_back(o);
				}
			}

		} else {
			pcl::PointXYZ p;
			p.getVector4fMap() = transform
					* other.keypoints3d[matches[i].queryIdx].getVector4fMap();
			keypoints3d.push_back(p);

			cv::vconcat(descriptors, other.descriptors.row(matches[i].queryIdx),
					descriptors);

			weights.push_back(other.weights[matches[i].queryIdx]);

			for (size_t j = 0; j < other.observations.size(); j++) {
				if (other.observations[j].point_id == matches[i].queryIdx) {

					observation o = other.observations[j];
					o.point_id = keypoints3d.size() - 1;
					o.cam_id += current_camera_positions_size;
					observations.push_back(o);
				}
			}

		}
	}

	return true;

}

void keypoint_map::remove_bad_points() {

	pcl::PointCloud<pcl::PointXYZ> new_keypoints3d;
	cv::Mat new_descriptors;
	std::vector<float> new_weights;

	std::vector<observation> new_observations;

	for (size_t i = 0; i < keypoints3d.size(); i++) {

		if (weights[i] > 1) {
			new_keypoints3d.push_back(keypoints3d[i]);
			if (new_descriptors.rows == 0) {
				descriptors.row(i).copyTo(new_descriptors);
			} else {
				cv::vconcat(new_descriptors, descriptors.row(i),
						new_descriptors);
			}
			new_weights.push_back(weights[i]);

			for (size_t j = 0; j < observations.size(); j++) {
				if (observations[j].point_id == i) {
					observation o = observations[j];
					o.point_id = new_keypoints3d.size() - 1;
					new_observations.push_back(o);
				}
			}

		}

	}

	keypoints3d = new_keypoints3d;
	descriptors = new_descriptors;
	weights = new_weights;
	observations = new_observations;

}

void keypoint_map::optimize() {

	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(true);
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverCholmod<
			g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver =
			new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	double focal_length = intrinsics[0];
	Eigen::Vector2d principal_point(intrinsics[2], intrinsics[3]);

	g2o::CameraParameters * cam_params = new g2o::CameraParameters(focal_length,
			principal_point, 0.);
	cam_params->setId(0);

	if (!optimizer.addParameter(cam_params)) {
		assert(false);
	}

	std::cerr << camera_positions.size() << " " << keypoints3d.size() << " "
			<< observations.size() << std::endl;

	int vertex_id = 0, point_id = 0;

	for (size_t i = 0; i < camera_positions.size(); i++) {
		Eigen::Vector3d trans(camera_positions[i].translation().cast<double>());
		Eigen::Quaterniond q(camera_positions[i].rotation().cast<double>());

		g2o::SE3Quat pose(q, trans);
		g2o::VertexSE3Expmap * v_se3 = new g2o::VertexSE3Expmap();
		v_se3->setId(vertex_id);
		if (i < 1) {
			v_se3->setFixed(true);
		}
		v_se3->setEstimate(pose);
		optimizer.addVertex(v_se3);
		vertex_id++;
	}

	for (size_t i = 0; i < keypoints3d.size(); i++) {
		g2o::VertexSBAPointXYZ * v_p = new g2o::VertexSBAPointXYZ();
		v_p->setId(vertex_id + point_id);
		v_p->setMarginalized(true);
		v_p->setEstimate(keypoints3d[i].getVector3fMap().cast<double>());
		optimizer.addVertex(v_p);
		point_id++;
	}

	for (size_t i = 0; i < observations.size(); i++) {
		g2o::EdgeProjectXYZ2UV * e = new g2o::EdgeProjectXYZ2UV();
		e->setVertex(0,
				dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(
						vertex_id + observations[i].point_id)->second));
		e->setVertex(1,
				dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(
						observations[i].cam_id)->second));
		e->setMeasurement(observations[i].coord.cast<double>());
		e->information() = Eigen::Matrix2d::Identity();

		//g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		//e->setRobustKernel(rk);

		e->setParameterId(0, 0);
		optimizer.addEdge(e);

	}

	optimizer.save("debug.txt");

	optimizer.initializeOptimization();
	optimizer.setVerbose(true);

	std::cout << std::endl;
	std::cout << "Performing full BA:" << std::endl;
	optimizer.optimize(10);
	std::cout << std::endl;

	for (int i = 0; i < vertex_id; i++) {
		g2o::HyperGraph::VertexIDMap::iterator v_it = optimizer.vertices().find(
				i);
		if (v_it == optimizer.vertices().end()) {
			std::cerr << "Vertex " << i << " not in graph!" << std::endl;
			exit(-1);
		}

		g2o::VertexSE3Expmap * v_c =
				dynamic_cast<g2o::VertexSE3Expmap *>(v_it->second);
		if (v_c == 0) {
			std::cerr << "Vertex " << i << "is not a VertexSE3Expmap!"
					<< std::endl;
			exit(-1);
		}

		Eigen::Affine3f pos;
		pos.fromPositionOrientationScale(
				v_c->estimate().translation().cast<float>(),
				v_c->estimate().rotation().cast<float>(),
				Eigen::Vector3f(1, 1, 1));
		camera_positions[i] = pos;
	}

	for (int i = 0; i < point_id; i++) {
		g2o::HyperGraph::VertexIDMap::iterator v_it = optimizer.vertices().find(
				vertex_id + i);
		if (v_it == optimizer.vertices().end()) {
			std::cerr << "Vertex " << vertex_id + i << " not in graph!"
					<< std::endl;
			exit(-1);
		}

		g2o::VertexSBAPointXYZ * v_p =
				dynamic_cast<g2o::VertexSBAPointXYZ *>(v_it->second);
		if (v_p == 0) {
			std::cerr << "Vertex " << vertex_id + i
					<< "is not a VertexSE3Expmap!" << std::endl;
			exit(-1);
		}

		keypoints3d[i].getVector3fMap() = v_p->estimate().cast<float>();
	}

}

pcl::PolygonMesh::Ptr keypoint_map::extract_surface() {

	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
			new pcl::PointCloud<pcl::PointXYZ>);

	for (size_t i = 0; i < depth_imgs.size(); i++) {

		std::cerr << camera_positions[i].matrix() << std::endl;

		for (int v = 0; v < depth_imgs[i].rows; v++) {
			for (int u = 0; u < depth_imgs[i].cols; u++) {
				if (depth_imgs[i].at<unsigned short>(v, u) != 0) {
					pcl::PointXYZ p;
					p.z = depth_imgs[i].at<unsigned short>(v, u) / 1000.0f;
					p.x = (u - intrinsics[2]) * p.z / intrinsics[0];
					p.y = (v - intrinsics[3]) * p.z / intrinsics[1];

					Eigen::Vector4f tmp = camera_positions[i]
							* p.getVector4fMap();

					p.getVector4fMap() = tmp;

					//ROS_INFO("Point %f %f %f from  %f %f ", p.x, p.y, p.z, keypoints[i].pt.x, keypoints[i].pt.y);

					point_cloud->push_back(p);
				}
			}
		}
	}

	pcl::io::savePCDFileASCII("room.pcd", *point_cloud);

	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_filtered(
				new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(point_cloud);
	sor.setLeafSize(0.05f, 0.05f, 0.05f);
	sor.filter(*point_cloud_filtered);

	pcl::io::savePCDFileASCII("room_sub.pcd", *point_cloud_filtered);

	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> ne;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(
			new pcl::search::KdTree<pcl::PointXYZ>);
	tree1->setInputCloud(point_cloud);
	ne.setInputCloud(point_cloud);
	ne.setSearchMethod(tree1);
	ne.setKSearch(20);
	pcl::PointCloud<pcl::PointNormal>::Ptr normals(
			new pcl::PointCloud<pcl::PointNormal>);
	ne.compute(*normals);

	point_cloud->clear();

	pcl::MarchingCubesHoppe<pcl::PointNormal> mc;

	pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh);
	mc.setInputCloud(normals);

	pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(
			new pcl::search::KdTree<pcl::PointNormal>);
	tree2->setInputCloud(normals);
	mc.setSearchMethod(tree2);
	mc.reconstruct(*triangles);

	cout << triangles->polygons.size() << " triangles created" << endl;
	pcl::io::saveVTKFile("mesh.vtk", *triangles);

	return triangles;

}

void compute_features(const cv::Mat & rgb, const cv::Mat & depth,
		const Eigen::Vector4f & intrinsics, cv::Ptr<cv::FeatureDetector> & fd,
		cv::Ptr<cv::DescriptorExtractor> & de,
		std::vector<cv::KeyPoint> & filtered_keypoints,
		pcl::PointCloud<pcl::PointXYZ> & keypoints3d, cv::Mat & descriptors) {
	cv::Mat gray;
	cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

	int threshold = 400;
	fd->setInt("hessianThreshold", threshold);
	std::vector<cv::KeyPoint> keypoints;

	cv::Mat mask(depth.size(), CV_8UC1);
	depth.convertTo(mask, CV_8U);

	fd->detect(gray, keypoints, mask);

	for (int i = 0; i < 5; i++) {
		if (keypoints.size() < 300) {
			threshold = threshold / 2;
			fd->setInt("hessianThreshold", threshold);
			keypoints.clear();
			fd->detect(gray, keypoints, mask);
		} else {
			break;
		}
	}

	if (keypoints.size() > 400)
		keypoints.resize(400);

	filtered_keypoints.clear();
	keypoints3d.clear();

	for (size_t i = 0; i < keypoints.size(); i++) {
		if (depth.at<unsigned short>(keypoints[i].pt) != 0) {
			filtered_keypoints.push_back(keypoints[i]);

			pcl::PointXYZ p;
			p.z = depth.at<unsigned short>(keypoints[i].pt) / 1000.0f;
			p.x = (keypoints[i].pt.x - intrinsics[2]) * p.z / intrinsics[0];
			p.y = (keypoints[i].pt.y - intrinsics[3]) * p.z / intrinsics[1];

			//ROS_INFO("Point %f %f %f from  %f %f ", p.x, p.y, p.z, keypoints[i].pt.x, keypoints[i].pt.y);

			keypoints3d.push_back(p);

		}
	}

	de->compute(gray, filtered_keypoints, descriptors);
}

bool estimate_transform_ransac(const pcl::PointCloud<pcl::PointXYZ> & src,
		const pcl::PointCloud<pcl::PointXYZ> & dst,
		const std::vector<cv::DMatch> matches, int num_iter,
		float distance2_threshold, int min_num_inliers, Eigen::Affine3f & trans,
		std::vector<bool> & inliers) {

	int max_inliers = 0;

	for (int iter = 0; iter < num_iter; iter++) {

		int rand_idx[3];
		// Select 3 random points
		for (int i = 0; i < 3; i++) {
			rand_idx[i] = rand() % matches.size();
		}

		while (rand_idx[0] == rand_idx[1] || rand_idx[0] == rand_idx[2]
				|| rand_idx[1] == rand_idx[2]) {
			for (int i = 0; i < 3; i++) {
				rand_idx[i] = rand() % matches.size();
			}
		}

		//std::cerr << "Random idx " << rand_idx[0] << " " << rand_idx[1] << " "
		//		<< rand_idx[2] << " " << matches.size() << std::endl;

		Eigen::Matrix3f src_rand, dst_rand;

		for (int i = 0; i < 3; i++) {
			src_rand.col(i) =
					src[matches[rand_idx[i]].queryIdx].getVector3fMap();
			dst_rand.col(i) =
					dst[matches[rand_idx[i]].trainIdx].getVector3fMap();

		}

		Eigen::Affine3f transformation;
		transformation = Eigen::umeyama(src_rand, dst_rand, false);

		//std::cerr << "src_rand " << std::endl << src_rand << std::endl;
		//std::cerr << "dst_rand " << std::endl << dst_rand << std::endl;
		//std::cerr << "src_rand_trans " << std::endl << transformation * src_rand
		//		<< std::endl;
		//std::cerr << "trans " << std::endl << transformation.matrix()
		//		<< std::endl;

		int current_num_inliers = 0;
		std::vector<bool> current_inliers;
		current_inliers.resize(matches.size());
		for (size_t i = 0; i < matches.size(); i++) {

			Eigen::Vector4f distance_vector = transformation
					* src[matches[i].queryIdx].getVector4fMap()
					- dst[matches[i].trainIdx].getVector4fMap();

			current_inliers[i] = distance_vector.squaredNorm()
					< distance2_threshold;
			if (current_inliers[i])
				current_num_inliers++;

		}

		if (current_num_inliers > max_inliers) {
			max_inliers = current_num_inliers;
			inliers = current_inliers;
		}
	}

	if (max_inliers < min_num_inliers) {
		return false;
	}

	Eigen::Matrix3Xf src_rand(3, max_inliers), dst_rand(3, max_inliers);

	int col_idx = 0;
	for (size_t i = 0; i < inliers.size(); i++) {
		if (inliers[i]) {
			src_rand.col(col_idx) = src[matches[i].queryIdx].getVector3fMap();
			dst_rand.col(col_idx) = dst[matches[i].trainIdx].getVector3fMap();
			col_idx++;
		}

	}

	trans = Eigen::umeyama(src_rand, dst_rand, false);

	//std::cerr << trans.matrix() << std::endl;
	std::cerr << max_inliers << std::endl;

	return true;

}

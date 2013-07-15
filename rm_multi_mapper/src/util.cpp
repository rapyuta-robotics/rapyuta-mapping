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
#include <pcl/common/transforms.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>


#include <octomap/ColorOcTree.h>

//These write and read functions must be defined for the serialization in FileStorage to work
static void write(cv::FileStorage & fs, const std::string&,
		const observation& x) {
	fs << "{" << "cam_id" << x.cam_id << "point_id" << x.point_id << "coord0"
			<< x.coord[0] << "coord1" << x.coord[1] << "}";
}
static void read(const cv::FileNode& node, observation& x,
		const observation& default_value = observation()) {
	if (node.empty()) {
		x = default_value;
	} else {

		x.cam_id = (int) node["cam_id"];
		x.point_id = (int) node["point_id"];
		x.coord[0] = (float) node["coord0"];
		x.coord[1] = (float) node["coord1"];
	}
}

keypoint_map::keypoint_map(cv::Mat & rgb, cv::Mat & depth,
		Eigen::Affine3f & transform) {
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

		keypoints3d[keypoint_id].getVector4fMap() = transform
				* keypoints3d[keypoint_id].getVector4fMap();

	}

	camera_positions.push_back(transform);
	rgb_imgs.push_back(rgb);
	depth_imgs.push_back(depth);

}

keypoint_map::keypoint_map(const std::string & dir_name) {

	de = new cv::SurfDescriptorExtractor;
	dm = new cv::FlannBasedMatcher;
	fd = new cv::SurfFeatureDetector;
	fd->setInt("hessianThreshold", 400);
	fd->setBool("extended", true);
	fd->setBool("upright", true);

	intrinsics << 525.0, 525.0, 319.5, 239.5;

	pcl::io::loadPCDFile(dir_name + "/keypoints3d.pcd", keypoints3d);

	cv::FileStorage fs(dir_name + "/descriptors.yml", cv::FileStorage::READ);

	fs["descriptors"] >> descriptors;
	fs["weights"] >> weights;

	cv::FileNode obs = fs["observations"];
	for (cv::FileNodeIterator it = obs.begin(); it != obs.end(); ++it) {
		observation o;
		*it >> o;
		observations.push_back(o);
	}

	cv::FileNode cam_pos = fs["camera_positions"];
	for (cv::FileNodeIterator it = cam_pos.begin(); it != cam_pos.end(); ++it) {
		Eigen::Affine3f pos;

		int i = 0;
		for (cv::FileNodeIterator it2 = (*it).begin(); it2 != (*it).end();
				++it2) {
			int u = i / 4;
			int v = i % 4;

			*it2 >> pos.matrix().coeffRef(u, v);
			i++;

		}
		camera_positions.push_back(pos);
	}

	fs.release();

	for (size_t i = 0; i < camera_positions.size(); i++) {
		cv::Mat rgb = cv::imread(
				dir_name + "/rgb/" + boost::lexical_cast<std::string>(i)
						+ ".png", CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat depth = cv::imread(
				dir_name + "/depth/" + boost::lexical_cast<std::string>(i)
						+ ".png", CV_LOAD_IMAGE_UNCHANGED);

		rgb_imgs.push_back(rgb);
		depth_imgs.push_back(depth);
	}

}

bool keypoint_map::merge_keypoint_map(const keypoint_map & other) {

	/*
	 std::vector<std::vector<cv::DMatch> > all_matches2;
	 std::vector<cv::DMatch> matches;


	 dm->knnMatch(
	 other.descriptors, descriptors, all_matches2, 2);

	 for (size_t i = 0; i < all_matches2.size(); ++i) {
	 double ratio = all_matches2[i][0].distance
	 / all_matches2[i][1].distance;
	 if (ratio < 0.6) {
	 matches.push_back(all_matches2[i][0]);
	 }
	 }

	 */

	std::vector<cv::DMatch> matches;
	dm->match(other.descriptors, descriptors, matches);

	Eigen::Affine3f transform;
	std::vector<bool> inliers;

	bool res = estimate_transform_ransac(other.keypoints3d, keypoints3d,
			matches, 3000, 0.03 * 0.03, 20, transform, inliers);

	if (!res)
		return false;

	std::map<int, int> unique_matches;
	for (size_t i = 0; i < matches.size(); i++) {

		if (unique_matches.find(matches[i].trainIdx) == unique_matches.end()) {
			unique_matches[matches[i].trainIdx] = i;

		} else {
			if (matches[unique_matches[matches[i].trainIdx]].distance
					> matches[i].distance) {
				inliers[unique_matches[matches[i].trainIdx]] = false;
				unique_matches[matches[i].trainIdx] = i;
			} else {
				inliers[i] = false;
			}
		}
	}

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

void keypoint_map::remove_bad_points(int min_num_observations) {

	pcl::PointCloud<pcl::PointXYZ> new_keypoints3d;
	cv::Mat new_descriptors;
	std::vector<float> new_weights;

	std::vector<observation> new_observations;

	for (size_t i = 0; i < keypoints3d.size(); i++) {

		if (weights[i] > min_num_observations) {
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
		Eigen::Affine3f cam_world = camera_positions[i].inverse();
		Eigen::Vector3d trans(cam_world.translation().cast<double>());
		Eigen::Quaterniond q(cam_world.rotation().cast<double>());

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

		g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
		e->setRobustKernel(rk);

		e->setParameterId(0, 0);
		optimizer.addEdge(e);

	}

	//optimizer.save("debug.txt");

	optimizer.initializeOptimization();
	optimizer.setVerbose(true);

	std::cout << std::endl;
	std::cout << "Performing full BA:" << std::endl;
	optimizer.optimize(1);
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
		camera_positions[i] = pos.inverse();
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

void keypoint_map::get_octree(octomap::OcTree & tree) {

	tree.clear();

	for (size_t i = 0; i < depth_imgs.size(); i++) {

		octomap::Pointcloud octomap_cloud;
		octomap::point3d sensor_origin(0.0, 0.0, 0.0);

		Eigen::Vector3f trans(camera_positions[i].translation());
		Eigen::Quaternionf rot(camera_positions[i].rotation());

		octomap::pose6d frame_origin(
				octomath::Vector3(trans.x(), trans.y(), trans.z()),
				octomath::Quaternion(rot.w(), rot.x(), rot.y(), rot.z()));

		for (int v = 0; v < depth_imgs[i].rows; v++) {
			for (int u = 0; u < depth_imgs[i].cols; u++) {
				if (depth_imgs[i].at<unsigned short>(v, u) != 0) {
					pcl::PointXYZ p;
					p.z = depth_imgs[i].at<unsigned short>(v, u) / 1000.0f;
					p.x = (u - intrinsics[2]) * p.z / intrinsics[0];
					p.y = (v - intrinsics[3]) * p.z / intrinsics[1];

					octomap_cloud.push_back(p.x, p.y, p.z);

				}
			}
		}

		tree.insertScan(octomap_cloud, sensor_origin, frame_origin);
	}

	tree.updateInnerOccupancy();

}

void keypoint_map::extract_surface() {

	octomap::ColorOcTree tree(0.05f);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud(
			new pcl::PointCloud<pcl::PointXYZRGBA>);

	for (size_t i = 0; i < depth_imgs.size(); i++) {

		cv::imwrite("rgb/" + boost::lexical_cast<std::string>(i) + ".png",
				rgb_imgs[i]);
		cv::imwrite("depth/" + boost::lexical_cast<std::string>(i) + ".png",
				depth_imgs[i]);

		octomap::Pointcloud octomap_cloud;
		octomap::point3d sensor_origin(0.0, 0.0, 0.0);

		Eigen::Vector3f trans(camera_positions[i].translation());
		Eigen::Quaternionf rot(camera_positions[i].rotation());

		octomap::pose6d frame_origin(
				octomath::Vector3(trans.x(), trans.y(), trans.z()),
				octomath::Quaternion(rot.w(), rot.x(), rot.y(), rot.z()));
		//std::cerr << camera_positions[i].matrix() << std::endl;

		for (int v = 0; v < depth_imgs[i].rows; v++) {
			for (int u = 0; u < depth_imgs[i].cols; u++) {
				if (depth_imgs[i].at<unsigned short>(v, u) != 0) {
					pcl::PointXYZRGBA p;
					p.z = depth_imgs[i].at<unsigned short>(v, u) / 1000.0f;
					p.x = (u - intrinsics[2]) * p.z / intrinsics[0];
					p.y = (v - intrinsics[3]) * p.z / intrinsics[1];
					cv::Vec3b brg = rgb_imgs[i].at<cv::Vec3b>(v, u);
					p.r = brg[2];
					p.g = brg[1];
					p.b = brg[0];
					p.a = 255;

					Eigen::Vector4f tmp = camera_positions[i]
							* p.getVector4fMap();

					if (tmp[2] < 2.0) {

						octomap_cloud.push_back(p.x, p.y, p.z);
						p.getVector4fMap() = tmp;

						octomap::point3d endpoint(p.x, p.y, p.z);
						octomap::ColorOcTreeNode* n = tree.search(endpoint);
						if (n) {
							n->setColor(p.r, p.g, p.b);
						}

						//ROS_INFO("Point %f %f %f from  %f %f ", p.x, p.y, p.z, keypoints[i].pt.x, keypoints[i].pt.y);

						point_cloud->push_back(p);

					}

				}
			}
		}
		tree.insertScan(octomap_cloud, sensor_origin, frame_origin);
		tree.updateInnerOccupancy();
	}

	tree.write("room.ot");

	pcl::io::savePCDFileASCII("room.pcd", *point_cloud);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud_filtered(
			new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
	sor.setInputCloud(point_cloud);
	sor.setLeafSize(0.05f, 0.05f, 0.05f);
	sor.filter(*point_cloud_filtered);

	pcl::io::savePCDFileASCII("room_sub.pcd", *point_cloud_filtered);

}

float keypoint_map::compute_error() {

	float error = 0;

	for (size_t i = 0; i < observations.size(); i++) {

		observation o = observations[i];

		Eigen::Vector4f point_transformed = camera_positions[o.cam_id].inverse()
				* keypoints3d[o.point_id].getVector4fMap();
		point_transformed /= point_transformed[3];

		Eigen::Vector2f pixel_pos;

		pixel_pos[0] = point_transformed[0] * intrinsics[0]
				/ point_transformed[2] + intrinsics[2];
		pixel_pos[1] = point_transformed[1] * intrinsics[1]
				/ point_transformed[2] + intrinsics[3];

		/*
		 std::cerr << "Observation " << i << " Prediction " << pixel_pos[0]
		 << " " << pixel_pos[1] << " measurement " << o.coord[0] << " "
		 << o.coord[1] << std::endl;
		 std::cerr << "Camera id" << o.cam_id << " point id " << o.point_id << std::endl;
		 */

		error += (o.coord - pixel_pos).squaredNorm();

	}

	return error;

}

void keypoint_map::align_z_axis() {

	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
			new pcl::PointCloud<pcl::PointXYZ>);

	for (int v = 0; v < depth_imgs[0].rows; v++) {
		for (int u = 0; u < depth_imgs[0].cols; u++) {
			if (depth_imgs[0].at<unsigned short>(v, u) != 0) {
				pcl::PointXYZ p;

				p.z = depth_imgs[0].at<unsigned short>(v, u) / 1000.0f;
				p.x = (u - intrinsics[2]) * p.z / intrinsics[0];
				p.y = (v - intrinsics[3]) * p.z / intrinsics[1];

				p.getVector4fMap() = camera_positions[0] * p.getVector4fMap();

				point_cloud->push_back(p);

			}
		}
	}

	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients(true);
	// Mandatory
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(0.005);

	seg.setInputCloud(point_cloud);
	seg.segment(*inliers, *coefficients);

	std::cerr << "Model coefficients: " << coefficients->values[0] << " "
			<< coefficients->values[1] << " " << coefficients->values[2] << " "
			<< coefficients->values[3] << " Num inliers "
			<< inliers->indices.size() << std::endl;

	Eigen::Affine3f transform = Eigen::Affine3f::Identity();
	if (coefficients->values[2] > 0) {
		transform.matrix().coeffRef(0, 2) = coefficients->values[0];
		transform.matrix().coeffRef(1, 2) = coefficients->values[1];
		transform.matrix().coeffRef(2, 2) = coefficients->values[2];
	} else {
		transform.matrix().coeffRef(0, 2) = -coefficients->values[0];
		transform.matrix().coeffRef(1, 2) = -coefficients->values[1];
		transform.matrix().coeffRef(2, 2) = -coefficients->values[2];
	}

	transform.matrix().col(0).head<3>() =
			transform.matrix().col(1).head<3>().cross(
					transform.matrix().col(2).head<3>());
	transform.matrix().col(1).head<3>() =
			transform.matrix().col(2).head<3>().cross(
					transform.matrix().col(0).head<3>());

	transform = transform.inverse();

	transform.matrix().coeffRef(2, 3) = coefficients->values[3];

	pcl::transformPointCloud(keypoints3d, keypoints3d, transform);

	for (size_t i = 0; i < camera_positions.size(); i++) {
		camera_positions[i] = transform * camera_positions[i];
	}

}

void keypoint_map::save(const std::string & dir_name) {

	if (boost::filesystem::exists(dir_name)) {
		boost::filesystem::remove_all(dir_name);
	}

	boost::filesystem::create_directory(dir_name);
	boost::filesystem::create_directory(dir_name + "/rgb");
	boost::filesystem::create_directory(dir_name + "/depth");

	pcl::io::savePCDFile(dir_name + "/keypoints3d.pcd", keypoints3d);
	cv::FileStorage fs(dir_name + "/descriptors.yml", cv::FileStorage::WRITE);
	fs << "descriptors" << descriptors << "weights" << weights;
	fs << "observations" << "[";

	for (size_t i = 0; i < observations.size(); i++) {
		fs << observations[i];
	}

	fs << "]";

	fs << "camera_positions" << "[";

	for (size_t i = 0; i < camera_positions.size(); i++) {
		fs << "[";

		for (int j = 0; j < 16; j++) {
			int u = j / 4;
			int v = j % 4;

			fs << camera_positions[i].matrix().coeff(u, v);

		}

		fs << "]";
	}

	fs << "]";

	fs.release();

	for (size_t i = 0; i < depth_imgs.size(); i++) {
		cv::imwrite(
				dir_name + "/rgb/" + boost::lexical_cast<std::string>(i)
						+ ".png", rgb_imgs[i]);
		cv::imwrite(
				dir_name + "/depth/" + boost::lexical_cast<std::string>(i)
						+ ".png", depth_imgs[i]);
	}

}


void keypoint_map::compute_2d_map(const octomap::OcTree & tree,
		nav_msgs::OccupancyGrid & grid) {

	double min_x, min_y, min_z, max_x, max_y, max_z;
	tree.getMetricMin(min_x, min_y, min_z);
	tree.getMetricMax(max_x, max_y, max_z);

	double res = tree.getResolution();

	int map_min_x = floor(min_x) / res;
	int map_min_y = floor(min_y) / res;
	int map_max_x = ceil(max_x) / res;
	int map_max_y = ceil(max_y) / res;

	std::cerr << min_x << " " << min_y << " " << min_z << " " << max_x << " "
			<< max_y << " " << max_z << std::endl;

	int map_width = map_max_x - map_min_x;
	int map_height = map_max_y - map_min_y;

	grid.info.width = map_width;
	grid.info.height = map_height;
	grid.info.resolution = res;
	grid.header.frame_id = "/cloudbot1/odom_combined";
	grid.header.stamp = ros::Time::now();
	grid.info.map_load_time = ros::Time::now();

	grid.data.resize(grid.info.width * grid.info.height, -1);

	for (octomap::OcTree::leaf_iterator it = tree.begin_leafs(), end =
			tree.end_leafs(); it != end; ++it) {

		if ((it.getZ() - it.getSize() / 2) < 0.6
				|| (it.getZ() + it.getSize() / 2) > 0.1) {

			if (it.getDepth() == tree.getTreeDepth()) {

				int x = it.getX() / res + map_min_x;
				int y = it.getY() / res + map_min_y;

				int idx = y * grid.info.width + x;

				if (tree.isNodeOccupied(*it)) {
					grid.data[idx] = 100;
				} else if (grid.data[idx] == -1) {
					grid.data[idx] = 0;
				}

			} else {
				int intSize = 1 << (tree.getTreeDepth() - it.getDepth());
				int x = it.getX() / res + map_min_x;
				int y = it.getY() / res + map_min_y;
				for (int dx = 0; dx < intSize; dx++) {
					for (int dy = 0; dy < intSize; dy++) {

					}
				}
			}

		}

	}

}


void keypoint_map::publish_keypoints(ros::Publisher & pub) {

	for (size_t i = 0; i < depth_imgs.size(); i++) {

		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

			for (int v = 0; v < depth_imgs[i].rows; v++) {
				for (int u = 0; u < depth_imgs[i].cols; u++) {
					if (depth_imgs[i].at<unsigned short>(v, u) != 0) {
						pcl::PointXYZRGBA p;
						p.z = depth_imgs[i].at<unsigned short>(v, u) / 1000.0f;
						p.x = (u - intrinsics[2]) * p.z / intrinsics[0];
						p.y = (v - intrinsics[3]) * p.z / intrinsics[1];
						cv::Vec3b brg = rgb_imgs[i].at<cv::Vec3b>(v, u);
						p.r = brg[2];
						p.g = brg[1];
						p.b = brg[0];
						p.a = 255;

						Eigen::Vector4f tmp = camera_positions[i]
								* p.getVector4fMap();

						if (tmp[2] < 0.6 && tmp[2] > 0.2) {
							p.getVector4fMap() = tmp;
							point_cloud->push_back(p);

						}

					}
				}
			}


			point_cloud->sensor_orientation_ = camera_positions[i].rotation();
			point_cloud->sensor_origin_ = camera_positions[i].translation().homogeneous();
			point_cloud->header.frame_id = "/map";
			point_cloud->header.seq = i;
			point_cloud->header.stamp = ros::Time::now();

			pub.publish(point_cloud);
			usleep(10000);
		}


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

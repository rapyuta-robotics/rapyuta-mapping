#include <keyframe_map.h>
#include <boost/filesystem.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <reduce_jacobian_rgb.h>
#include <reduce_jacobian_rgb_3d.h>

keyframe_map::keyframe_map() {
}

void keyframe_map::add_frame(const rm_localization::Keyframe::ConstPtr & k) {
	frames.push_back(color_keyframe::from_msg(k));

}

float keyframe_map::optimize_panorama(int level) {

	float iteration_max_update;
	int size = frames.size();

	tbb::concurrent_vector<std::pair<int, int> > overlaping_keyframes;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (i != j) {
				float angle =
						frames[i]->get_pos().unit_quaternion().angularDistance(
								frames[j]->get_pos().unit_quaternion());

				if (angle < M_PI / 6) {
					overlaping_keyframes.push_back(std::make_pair(i, j));
					//ROS_INFO("Images %d and %d intersect with angular distance %f", i, j, angle*180/M_PI);
				}
			}
		}
	}

	reduce_jacobian_rgb rj(frames, size, level);

	tbb::parallel_reduce(
			tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>(
					overlaping_keyframes.begin(), overlaping_keyframes.end()),
			rj);

	/*
	 rj(
	 tbb::blocked_range<
	 tbb::concurrent_vector<std::pair<int, int> >::iterator>(
	 overlaping_keyframes.begin(), overlaping_keyframes.end()));
	 */

	Eigen::VectorXf update = -rj.JtJ.ldlt().solve(rj.Jte);
	//Eigen::VectorXf update =
	//			-rj.JtJ.block(0, 0, size * 3, size* 3).ldlt().solve(
	//					rj.Jte.segment(0, size * 3));

	iteration_max_update = std::max(std::abs(update.maxCoeff()),
			std::abs(update.minCoeff()));

	ROS_INFO("Max update %f", iteration_max_update);

	for (int i = 0; i < size; i++) {

		frames[i]->get_pos().so3() = Sophus::SO3f::exp(update.segment<3>(i * 3))
				* frames[i]->get_pos().so3();

		frames[i]->get_pos().translation() = frames[0]->get_pos().translation();

		frames[i]->get_intrinsics().array() =
				update.segment<3>(size * 3).array().exp()
						* frames[i]->get_intrinsics().array();

		if (i == 0) {
			Eigen::Vector3f intrinsics = frames[i]->get_intrinsics();
			ROS_INFO("New intrinsics %f, %f, %f", intrinsics(0), intrinsics(1),
					intrinsics(2));
		}

	}

	return iteration_max_update;

}

float keyframe_map::optimize(int level) {

	float iteration_max_update;
	int size = frames.size();

	tbb::concurrent_vector<std::pair<int, int> > overlaping_keyframes;

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (i != j) {
				float angle =
						frames[i]->get_pos().unit_quaternion().angularDistance(
								frames[j]->get_pos().unit_quaternion());

				if (angle < M_PI / 6) {
					overlaping_keyframes.push_back(std::make_pair(i, j));
					//ROS_INFO("Images %d and %d intersect with angular distance %f", i, j, angle*180/M_PI);
				}
			}
		}
	}

	reduce_jacobian_rgb_3d rj(frames, size, level);

	/*
	tbb::parallel_reduce(
			tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>(
					overlaping_keyframes.begin(), overlaping_keyframes.end()),
			rj);
	*/

	rj(
		 tbb::blocked_range<
		 tbb::concurrent_vector<std::pair<int, int> >::iterator>(
		 overlaping_keyframes.begin(), overlaping_keyframes.end()));

	Eigen::VectorXf update =
					-rj.JtJ.block(6, 6, (size-1) * 6, (size-1) * 6).ldlt().solve(
							rj.Jte.segment(6, (size-1) * 6));


	iteration_max_update = std::max(std::abs(update.maxCoeff()),
			std::abs(update.minCoeff()));

	ROS_INFO("Max update %f", iteration_max_update);

	for (int i = 0; i < size-1; i++) {
		frames[i+1]->get_pos() = Sophus::SE3f::exp(update.segment<6>(i * 6))
				* frames[i+1]->get_pos();
	}

	return iteration_max_update;

}

cv::Mat keyframe_map::get_panorama_image() {
	cv::Mat res = cv::Mat::zeros(512, 1024, CV_32F);
	cv::Mat w = cv::Mat::zeros(res.size(), res.type());

	cv::Mat map_x(res.size(), res.type()), map_y(res.size(), res.type()),
			img_projected(res.size(), res.type()), mask(res.size(), res.type());

	cv::Mat image_weight(frames[0]->get_rgb().size(), res.type()),
			intencity_weighted(image_weight.size(), image_weight.type());

	float cx = image_weight.cols / 2.0;
	float cy = image_weight.rows / 2.0;
	float max_r = cy * cy;
	for (int v = 0; v < image_weight.rows; v++) {
		for (int u = 0; u < image_weight.cols; u++) {
			image_weight.at<float>(v, u) = std::max(
					1.0 - ((u - cx) * (u - cx) + (v - cy) * (v - cy)) / max_r,
					0.0);
		}
	}

//cv::imshow("image_weight", image_weight);
//cv::waitKey(0);

	cx = res.cols / 2.0;
	cy = res.rows / 2.0;

	float scale_x = 2 * M_PI / res.cols;
	float scale_y = M_PI / res.rows;

	for (size_t i = 0; i < frames.size(); i++) {

		Eigen::Vector3f intrinsics = frames[i]->get_intrinsics(0);
		Eigen::Quaternionf Qiw =
				frames[i]->get_pos().unit_quaternion().inverse();

		Eigen::Matrix3f K;
		K << intrinsics[0], 0, intrinsics[1], 0, intrinsics[0], intrinsics[2], 0, 0, 1;
		Eigen::Matrix3f H = K * Qiw.matrix();

		for (int v = 0; v < map_x.rows; v++) {
			for (int u = 0; u < map_x.cols; u++) {

				float phi = (u - cx) * scale_x;
				float theta = (v - cy) * scale_y;

				Eigen::Vector3f vec(cos(theta) * cos(phi),
						-cos(theta) * sin(phi), -sin(theta));
				//Eigen::Vector3f vec(cos(theta) * sin(phi), sin(theta),
				//		cos(theta) * cos(phi));

				vec = H * vec;

				if (vec[2] > 0.01) {
					map_x.at<float>(v, u) = vec[0] / vec[2];
					map_y.at<float>(v, u) = vec[1] / vec[2];
				} else {
					map_x.at<float>(v, u) = -1;
					map_y.at<float>(v, u) = -1;
				}
			}
		}

		img_projected = 0.0f;
		mask = 0.0f;
		cv::Mat intencity;
		frames[i]->get_i(0).convertTo(intencity, CV_32F, 1.0 / 255);
		cv::multiply(intencity, image_weight, intencity_weighted);
		cv::remap(intencity_weighted, img_projected, map_x, map_y,
				CV_INTER_LINEAR, cv::BORDER_TRANSPARENT, 0);
		cv::remap(image_weight, mask, map_x, map_y, CV_INTER_LINEAR,
				cv::BORDER_TRANSPARENT, 0);

		res += img_projected;
		w += mask;

		//cv::imshow("intencity", intencity);
		//cv::imshow("intencity_weighted", intencity_weighted);
		//cv::imshow("img_projected", img_projected);
		//cv::imshow("mask", mask);
		//cv::waitKey(0);

	}

	return res / w;
}

void keyframe_map::save(const std::string & dir_name) {

	if (boost::filesystem::exists(dir_name)) {
		boost::filesystem::remove_all(dir_name);
	}

	boost::filesystem::create_directory(dir_name);
	boost::filesystem::create_directory(dir_name + "/rgb");
	boost::filesystem::create_directory(dir_name + "/depth");

	for (size_t i = 0; i < frames.size(); i++) {
		cv::imwrite(
				dir_name + "/rgb/" + boost::lexical_cast<std::string>(i)
						+ ".png", frames[i]->get_rgb());
		cv::imwrite(
				dir_name + "/depth/" + boost::lexical_cast<std::string>(i)
						+ ".png", frames[i]->get_d(0));

	}

	std::ofstream f((dir_name + "/positions.txt").c_str(),
			std::ios_base::binary);
	for (size_t i = 0; i < frames.size(); i++) {
		Eigen::Quaternionf q = frames[i]->get_pos().unit_quaternion();
		Eigen::Vector3f t = frames[i]->get_pos().translation();
		Eigen::Vector3f intrinsics = frames[i]->get_intrinsics(0);

		f.write((char *) q.coeffs().data(), sizeof(float) * 4);
		f.write((char *) t.data(), sizeof(float) * 3);
		f.write((char *) intrinsics.data(), sizeof(float) * 3);

	}
	f.close();

}

void keyframe_map::load(const std::string & dir_name) {

	std::vector<std::pair<Sophus::SE3f, Eigen::Vector3f> > positions;

	std::ifstream f((dir_name + "/positions.txt").c_str(),
			std::ios_base::binary);
	while (f) {
		Eigen::Quaternionf q;
		Eigen::Vector3f t;
		Eigen::Vector3f intrinsics;

		f.read((char *) q.coeffs().data(), sizeof(float) * 4);
		f.read((char *) t.data(), sizeof(float) * 3);
		f.read((char *) intrinsics.data(), sizeof(float) * 3);

		positions.push_back(std::make_pair(Sophus::SE3f(q, t), intrinsics));
	}

	positions.pop_back();
	std::cerr << "Loaded " << positions.size() << " positions" << std::endl;

	for (size_t i = 0; i < positions.size(); i++) {
		cv::Mat rgb = cv::imread(
				dir_name + "/rgb/" + boost::lexical_cast<std::string>(i)
						+ ".png", CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat depth = cv::imread(
				dir_name + "/depth/" + boost::lexical_cast<std::string>(i)
						+ ".png", CV_LOAD_IMAGE_UNCHANGED);

		cv::Mat gray;
		cv::cvtColor(rgb, gray, CV_RGB2GRAY);

		color_keyframe::Ptr k(
				new color_keyframe(rgb, gray, depth, positions[i].first,
						positions[i].second));
		frames.push_back(k);
	}

}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyframe_map::get_map_pointcloud() {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr res(
			new pcl::PointCloud<pcl::PointXYZRGB>);

	for (size_t i = 0; i < frames.size(); i++) {

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
				frames[i]->get_colored_pointcloud(4);

		*res += *cloud;

	}

	return res;

}


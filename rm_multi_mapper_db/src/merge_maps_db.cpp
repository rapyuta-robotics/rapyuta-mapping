#include <util.h>
#include <util_mysql.h>
#include <util_mongo.h>

int main(int argc, char** argv) {
	ros::init(argc, argv, "map_merger_db");
	util::Ptr U(new util_mongo);

	int map_id1 = boost::lexical_cast<int>(argv[1]);
	int map_id2 = boost::lexical_cast<int>(argv[2]);

	while (ros::ok()) {
		long idx1 = U->get_random_keyframe_idx(map_id1);
		long idx2 = U->get_random_keyframe_idx(map_id2);

		pcl::PointCloud<pcl::PointXYZ> keypoints3d1, keypoints3d2;
		cv::Mat descriptors1, descriptors2;

		//std::cerr << "Trying to match " << goal->Overlap[i].first << " "
		//		<< goal->Overlap[i].second << std::endl;

		U->get_keypoints(idx1, keypoints3d1, descriptors1);
		U->get_keypoints(idx2, keypoints3d2, descriptors2);

		Sophus::SE3f t;
		if (U->find_transform(keypoints3d1, keypoints3d2, descriptors1,
				descriptors2, t)) {
			std::cerr << "Found transformation " << std::endl;

			color_keyframe::Ptr k1 = U->get_keyframe(idx1);
			color_keyframe::Ptr k2 = U->get_keyframe(idx2);

			t = k1->get_pos() * t * k2->get_pos().inverse();

			std::vector<util::position> p;
			U->load_positions(map_id2, p);

			for (size_t i = 0; i < p.size(); i++) {
				p[i].transform = t * p[i].transform;
				U->update_position(p[i]);
			}

			U->merge_map(map_id2, map_id1);

			U->add_measurement(idx1, idx2, t, "RANSAC");
			std::cerr << "Merged maps " << std::endl;
			break;
		}

		ros::spinOnce();
	}

	return 0;
}

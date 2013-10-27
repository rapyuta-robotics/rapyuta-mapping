#include <util.h>
#include <gtest/gtest.h>

template<typename T>
void check_equal(const cv::Mat & m1, const cv::Mat & m2) {
	EXPECT_EQ(m1.cols, m2.cols);
	EXPECT_EQ(m1.rows, m2.rows);
	EXPECT_EQ(m1.type(), m2.type());

	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {
			if (m1.at<T>(i, j) != m2.at<T>(i, j)) {
				ADD_FAILURE()<< "Pixel value mismatch at (" << i << "," <<
				j << ") values " << (double) m1.at<T>(i, j) <<
				" != " << (double) m2.at<T>(i, j);
			}
		}

	}
}

TEST(UtilTest, keyframeSaveTest) {
	util U;
	int robot_id = U.get_new_robot_id();

	keyframe_map map;
	map.load("test/data/map");
	map.frames.resize(1);

	long shift = robot_id * (1l << 32);

	map.frames[0]->set_id(shift);
	U.add_keyframe(robot_id, map.frames[0]);

	keyframe::Ptr k1 = map.frames[0];
	keyframe::Ptr k2 = U.get_keyframe(shift);

	EXPECT_FLOAT_EQ(k1->get_pos().translation().x(),
			k2->get_pos().translation().x());
	EXPECT_FLOAT_EQ(k1->get_pos().translation().y(),
			k2->get_pos().translation().y());
	EXPECT_FLOAT_EQ(k1->get_pos().translation().z(),
			k2->get_pos().translation().z());

	EXPECT_FLOAT_EQ(k1->get_pos().unit_quaternion().x(),
			k2->get_pos().unit_quaternion().x());
	EXPECT_FLOAT_EQ(k1->get_pos().unit_quaternion().y(),
			k2->get_pos().unit_quaternion().y());
	EXPECT_FLOAT_EQ(k1->get_pos().unit_quaternion().z(),
			k2->get_pos().unit_quaternion().z());
	EXPECT_FLOAT_EQ(k1->get_pos().unit_quaternion().w(),
			k2->get_pos().unit_quaternion().w());

	EXPECT_FLOAT_EQ(k1->get_intrinsics().x(), k2->get_intrinsics().x());
	EXPECT_FLOAT_EQ(k1->get_intrinsics().y(), k2->get_intrinsics().y());
	EXPECT_FLOAT_EQ(k1->get_intrinsics().z(), k2->get_intrinsics().z());

	check_equal<uint8_t>(k1->get_i(0), k2->get_i(0));
	check_equal<uint16_t>(k1->get_d(0), k2->get_d(0));

}

/*
TEST(UtilTest, keypointsSaveTest) {
	util U;
	int robot_id = U.get_new_robot_id();

	keyframe_map map;
	map.load("test/data/map");
	map.frames.resize(1);

	long shift = robot_id * (1l << 32);

	map.frames[0]->set_id(shift);
	U.add_keyframe(robot_id, map.frames[0]);

	keyframe::Ptr k = map.frames[0];

	std::vector<cv::KeyPoint> keypoints1;
	pcl::PointCloud<pcl::PointXYZ> keypoints3d1, keypoints3d2;
	cv::Mat descriptors1, descriptors2;
	U.compute_features(k->get_i(0), k->get_d(0), k->get_intrinsics(0), keypoints1,
			keypoints3d1, descriptors1);

	U.get_keypoints(shift, keypoints3d2, descriptors2);
	//EXPECT_EQ(descriptors1.cols, descriptors2.cols);
	//EXPECT_EQ(descriptors1.rows, descriptors2.rows);
	//EXPECT_EQ(descriptors1.type(), descriptors2.type());
	//check_equal<float>(descriptors1, descriptors2);

}
*/

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

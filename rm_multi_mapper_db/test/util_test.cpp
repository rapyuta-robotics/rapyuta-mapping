#include <util.h>
#include <gtest/gtest.h>

void check_equal(const cv::Mat & m1, const cv::Mat & m2) {
	EXPECT_EQ(m1.cols, m2.cols);
	EXPECT_EQ(m1.rows, m2.rows);
	EXPECT_EQ(m1.type(), m2.type());

	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {
			if (m1.at<uint8_t>(i, j) != m2.at<uint8_t>(i, j)) {
				ADD_FAILURE()<< "Pixel value mismatch at (" << i << "," <<
				j << ") values " << (int) m1.at<uint8_t>(i, j) <<
				" != " << (int) m2.at<uint8_t>(i, j);
			}
		}

	}
}

TEST(UtilTest, keyframeSaveTest) {
	util U;
	int robot_id = U.get_new_robot_id();
	std::cerr << "New robot id " << robot_id << std::endl;

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

	check_equal(k1->get_i(0), k2->get_i(0));
}

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

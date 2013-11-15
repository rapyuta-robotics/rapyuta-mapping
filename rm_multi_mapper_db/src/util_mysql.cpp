#include <util_mysql.h>

using namespace std;

util_mysql::util_mysql() {
	// TODO make arguments
	server = "localhost";
	user = "mapping";
	password = "mapping";
	database = "mapping";

	driver = get_driver_instance();
	con.reset(driver->connect("tcp://" + server + ":3306", user, password));
	con->setSchema(database);

	get_map_id_from_robot_id.reset(
			con->prepareStatement("SELECT map_id FROM robot WHERE id = ?"));

	insert_keyframe.reset(con->prepareStatement("INSERT INTO keyframe "
			"(`id`, `q0`, `q1`, `q2`, `q3`,"
			" `t0`, `t1`, `t2`, `int0`, `int1`,"
			" `int2`, `rgb`, `depth`, `map_id`) "
			"VALUES "
			"(?,?,?,?,?"
			",?,?,?,?,?"
			",?,?,?,?)"));

	insert_keypoints.reset(con->prepareStatement("UPDATE keyframe SET"
			" `num_keypoints`=? , `descriptor_size`= ?, `descriptor_type`=?,"
			" `keypoints`= ?, `descriptors`=? "
			" WHERE `id` = ? "));

	insert_measurement.reset(
			con->prepareStatement(
					"INSERT INTO measurement"
							" (`id`, `one`, `two`, `q0`, `q1`, `q2`,"
							" `q3`, `t0`, `t1`, `t2`, `type`)"
							" VALUES"
							" (NULL,?,?,?,?,?,?,?,?,?,?)"));

	insert_new_robot.reset(
			con->prepareStatement("INSERT INTO robot (id, map_id) "
					"VALUES(NULL, NULL)"));

	insert_map_id.reset(
				con->prepareStatement(
						"UPDATE robot SET map_id = LAST_INSERT_ID() "
						"WHERE id = LAST_INSERT_ID()"));

	select_map_id.reset(
				con->prepareStatement(
						"SELECT LAST_INSERT_ID() as id"));

	select_keyframe.reset(
			con->prepareStatement("SELECT `q0`, `q1`, `q2`, `q3`, `t0`, `t1`, `t2`, "
				"`int0`, `int1`, `int2`, `rgb`, `depth`, `id` "
				"FROM keyframe WHERE `id` = ?"));


	select_keypoints.reset(
				con->prepareStatement("SELECT `keypoints`, `descriptors`, "
						"`descriptor_size`, `num_keypoints`, `descriptor_type` "
						"FROM keyframe WHERE `id` = ?"));

	select_map.reset(
				con->prepareStatement("SELECT `q0`, `q1`, `q2`, `q3`, `t0`, `t1`, `t2`, "
					"`int0`, `int1`, `int2`, `rgb`, `depth`, `id` "
					"FROM keyframe WHERE `map_id` = "
					"(SELECT `map_id` FROM robot WHERE `id` = ?)"));

	select_positions.reset(
				con->prepareStatement("SELECT `q0`, `q1`, `q2`, `q3`, `t0`, `t1`, `t2`, `id` "
					"FROM keyframe WHERE `map_id` = ?"));

	select_measurements.reset(
					con->prepareStatement("SELECT * FROM measurement "
							"WHERE measurement.one = ?"));

	select_overlapping_keyframes.reset(
			con->prepareStatement("SELECT f1.id as id1, f2.id as id2 "
					"FROM keyframe f1, keyframe f2 "
					"WHERE f1.map_id = ? "
					"AND f2.map_id = ? "
					"AND (abs(f1.q0*f2.q0 + f1.q1*f2.q1 + f1.q2*f2.q2"
					" + f1.q3*f2.q3) >= 1.0 OR 2*acos(abs(f1.q0*f2.q0 + f1.q1*f2.q1 +"
					" f1.q2*f2.q2 + f1.q3*f2.q3)) < pi()/4) "
					"AND f1.id < f2.id "
					"AND SQRT(POWER((f1.t0 - f2.t0), 2) + "
					"POWER((f1.t1 - f2.t1), 2) + "
					"POWER((f1.t2 - f2.t2), 2)) < 3 "
					"AND NOT EXISTS "
					"(SELECT id FROM measurement "
					"WHERE measurement.one = f1.id AND two = f2.id )"));


	select_random_idx.reset(
			con->prepareStatement("SELECT id FROM keyframe "
					"WHERE map_id=? ORDER BY RAND() LIMIT 1"));

	update_keyframe.reset(
				con->prepareStatement(
						"UPDATE keyframe SET "
						"`q0`= ?, `q1`= ?, `q2`= ?, `q3`= ?, "
						"`t0`= ?, `t1`= ?, `t2`= ? WHERE id = ?"));

	update_robot_map_id.reset(
					con->prepareStatement(
							"UPDATE robot SET "
							"`map_id`= ? WHERE `map_id` = ?"));

	update_keyframe_map_id.reset(
						con->prepareStatement(
								"UPDATE keyframe SET "
								"`map_id`= ? WHERE `map_id` = ?"));


	// Classes for feature extraction
	de = new cv::SurfDescriptorExtractor;
	fd = new cv::SurfFeatureDetector;

	fd->setInt("hessianThreshold", 400);
	fd->setInt("extended", 1);
	fd->setInt("upright", 1);

	de->setInt("hessianThreshold", 400);
	de->setInt("extended", 1);
	de->setInt("upright", 1);

}

util_mysql::~util_mysql() {
}

int util_mysql::get_new_robot_id() {

	insert_new_robot->executeUpdate();
	insert_map_id->executeUpdate();

	boost::shared_ptr<sql::ResultSet> res(select_map_id->executeQuery());
	res->next();
	int robot_id = res->getInt("id");
	return robot_id;

}

void util_mysql::add_keyframe(int robot_id, const color_keyframe::Ptr & k) {

	try {

		get_map_id_from_robot_id->setInt(1, robot_id);
		boost::shared_ptr<sql::ResultSet> res(
				get_map_id_from_robot_id->executeQuery());
		res->next();
		int map_id = res->getInt("map_id");

		insert_keyframe->setInt64(1, k->get_id());
		insert_keyframe->setDouble(2, k->get_pos().unit_quaternion().x());
		insert_keyframe->setDouble(3, k->get_pos().unit_quaternion().y());
		insert_keyframe->setDouble(4, k->get_pos().unit_quaternion().z());
		insert_keyframe->setDouble(5, k->get_pos().unit_quaternion().w());

		insert_keyframe->setDouble(6, k->get_pos().translation().x());
		insert_keyframe->setDouble(7, k->get_pos().translation().y());
		insert_keyframe->setDouble(8, k->get_pos().translation().z());

		insert_keyframe->setDouble(9, k->get_intrinsics()[0]);
		insert_keyframe->setDouble(10, k->get_intrinsics()[1]);
		insert_keyframe->setDouble(11, k->get_intrinsics()[2]);

		std::vector<uint8_t> rgb_data, depth_data;

		cv::imencode(".png", k->get_rgb(), rgb_data);
		DataBuf rgb_buffer((char*) rgb_data.data(), rgb_data.size());
		std::istream rgb_stream(&rgb_buffer);

		cv::imencode(".png", k->get_d(0), depth_data);
		DataBuf depth_buffer((char*) depth_data.data(), depth_data.size());
		std::istream depth_stream(&depth_buffer);

		insert_keyframe->setBlob(12, &rgb_stream);
		insert_keyframe->setBlob(13, &depth_stream);
		insert_keyframe->setInt(14, map_id);

		insert_keyframe->executeUpdate();

	} catch (sql::SQLException &e) {
		std::cout << "# ERR: SQLException in " << __FILE__;
		std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__
				<< std::endl;
		std::cout << "# ERR: " << e.what();
		std::cout << " (MySQL error code: " << e.getErrorCode();
		std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
	}

}

void util_mysql::add_keypoints(const color_keyframe::Ptr & k) {
	try {

		std::vector<cv::KeyPoint> keypoints;
		pcl::PointCloud<pcl::PointXYZ> keypoints3d;
		cv::Mat descriptors;
		compute_features(k->get_i(0), k->get_d(0), k->get_intrinsics(0),
				keypoints, keypoints3d, descriptors);

		insert_keypoints->setInt(1, keypoints3d.size());
		insert_keypoints->setInt(2, descriptors.cols);
		insert_keypoints->setInt(3, descriptors.type());

		assert(descriptors.type() == CV_32F);
		std::cerr << "Keypoints size " << keypoints3d.size() << " "
				<< descriptors.size() << std::endl;

		DataBuf keypoints_buffer((char*) keypoints3d.points.data(),
				keypoints3d.points.size() * sizeof(pcl::PointXYZ));
		std::istream keypoints_stream(&keypoints_buffer);

		DataBuf descriptors_buffer((char*) descriptors.data,
				descriptors.cols * descriptors.rows * sizeof(float));
		std::istream descriptors_stream(&descriptors_buffer);

		insert_keypoints->setBlob(4, &keypoints_stream);
		insert_keypoints->setBlob(5, &descriptors_stream);
		insert_keypoints->setInt64(6, k->get_id());

		insert_keypoints->executeUpdate();

	} catch (sql::SQLException &e) {
		std::cout << "# ERR: SQLException in " << __FILE__;
		std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__
				<< std::endl;
		std::cout << "# ERR: " << e.what();
		std::cout << " (MySQL error code: " << e.getErrorCode();
		std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
	}
}

void util_mysql::add_measurement(long int first, long int second,
		const Sophus::SE3f & transform, const std::string & type) {
	try {

		insert_measurement->setInt64(1, first);
		insert_measurement->setInt64(2, second);

		insert_measurement->setDouble(3, transform.unit_quaternion().x());
		insert_measurement->setDouble(4, transform.unit_quaternion().y());
		insert_measurement->setDouble(5, transform.unit_quaternion().z());
		insert_measurement->setDouble(6, transform.unit_quaternion().w());

		insert_measurement->setDouble(7, transform.translation().x());
		insert_measurement->setDouble(8, transform.translation().y());
		insert_measurement->setDouble(9, transform.translation().z());

		insert_measurement->setString(10, type);

		insert_measurement->executeUpdate();

	} catch (sql::SQLException &e) {
		std::cout << "# ERR: SQLException in " << __FILE__;
		std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__
				<< std::endl;
		std::cout << "# ERR: " << e.what();
		std::cout << " (MySQL error code: " << e.getErrorCode();
		std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
	}
}

color_keyframe::Ptr util_mysql::get_keyframe(long frame_id) {
	select_keyframe->setInt64(1, frame_id);
	boost::shared_ptr<sql::ResultSet> res(select_keyframe->executeQuery());
	res->next();
	return get_keyframe(res);

}

Sophus::SE3f util_mysql::get_pose(boost::shared_ptr<sql::ResultSet> & res) {
	Eigen::Quaternionf q;
	Eigen::Vector3f t;
	q.x() = res->getDouble("q0");
	q.y() = res->getDouble("q1");
	q.z() = res->getDouble("q2");
	q.w() = res->getDouble("q3");
	t[0] = res->getDouble("t0");
	t[1] = res->getDouble("t1");
	t[2] = res->getDouble("t2");

	return Sophus::SE3f(q, t);

}

color_keyframe::Ptr util_mysql::get_keyframe(boost::shared_ptr<sql::ResultSet> & res) {

	Sophus::SE3f pose;
	pose = get_pose(res);

	Eigen::Vector3f intrinsics;

	intrinsics[0] = res->getDouble("int0");
	intrinsics[1] = res->getDouble("int1");
	intrinsics[2] = res->getDouble("int2");

	std::vector<uint8_t> rgb_data, depth_data;
	boost::shared_ptr<std::istream> rgb_in(res->getBlob("rgb"));
	while (*rgb_in) {
		uint8_t tmp;
		rgb_in->read((char*) &tmp, sizeof(tmp));
		rgb_data.push_back(tmp);
	}

	//std::cerr << "Read rgb data size " << rgb_data.size() << std::endl;

	boost::shared_ptr<std::istream> depth_in(res->getBlob("depth"));
	while (*depth_in) {
		uint8_t tmp;
		depth_in->read((char*) &tmp, sizeof(tmp));
		depth_data.push_back(tmp);
	}

	//std::cerr << "Read depth data size " << depth_data.size() << std::endl;

	cv::Mat rgb, depth;
	rgb = cv::imdecode(rgb_data, CV_LOAD_IMAGE_UNCHANGED);
	depth = cv::imdecode(depth_data, CV_LOAD_IMAGE_UNCHANGED);

	cv::Mat gray;
	cv::cvtColor(rgb, gray, CV_RGB2GRAY);

	color_keyframe::Ptr k(
			new color_keyframe(rgb, gray, depth, pose, intrinsics));

	k->set_id(res->getDouble("id"));

	return k;
}

void util_mysql::get_keypoints(long frame_id,
		pcl::PointCloud<pcl::PointXYZ> & keypoints3d, cv::Mat & descriptors) {

	select_keypoints->setInt64(1, frame_id);
	boost::shared_ptr<sql::ResultSet> res(select_keypoints->executeQuery());
	res->next();

	keypoints3d.clear();
	boost::shared_ptr<std::istream> keypoints_in(res->getBlob("keypoints"));
	while (*keypoints_in) {
		pcl::PointXYZ tmp;
		keypoints_in->read((char*) &tmp, sizeof(tmp));
		keypoints3d.push_back(tmp);
	}
	keypoints3d.resize(keypoints3d.size() - 1);

	boost::shared_ptr<std::istream> descriptors_in(res->getBlob("descriptors"));
	std::vector<uint8_t> descriptors_data;

	while (*descriptors_in) {
		uint8_t tmp;
		descriptors_in->read((char*) &tmp, sizeof(tmp));
		descriptors_data.push_back(tmp);
	}
	descriptors_data.resize(descriptors_data.size() - 1);

	int cols = res->getDouble("descriptor_size");
	int rows = res->getDouble("num_keypoints");
	int type = res->getDouble("descriptor_type");

	cv::Mat tmp_mat = cv::Mat(rows, cols, type,
			(void *) descriptors_data.data());


	tmp_mat.copyTo(descriptors);

}

boost::shared_ptr<keyframe_map> util_mysql::get_robot_map(int robot_id) {
	select_map->setInt(1, robot_id);
	boost::shared_ptr<sql::ResultSet> res(select_map->executeQuery());
	boost::shared_ptr<keyframe_map> map(new keyframe_map);

	while (res->next()) {
		map->frames.push_back(get_keyframe(res));
	}

	return map;
}

void util_mysql::get_overlapping_pairs(int map_id,
		std::vector<std::pair<long, long> > & overlapping_keyframes) {

	select_overlapping_keyframes->setInt(1, map_id);
	select_overlapping_keyframes->setInt(2, map_id);

	boost::shared_ptr<sql::ResultSet> res(
			select_overlapping_keyframes->executeQuery());

	while (res->next()) {
		overlapping_keyframes.push_back(
				std::make_pair(res->getInt64("id1"), res->getInt64("id2")));
	}
}

void util_mysql::load_measurements(long keyframe_id,
		std::vector<measurement> & m) {

	select_measurements->setInt64(1, keyframe_id);
	boost::shared_ptr<sql::ResultSet> res(select_measurements->executeQuery());

	while (res->next()) {
		measurement mes;
		mes.first = res->getInt64("one");
		mes.second = res->getInt64("two");
		mes.transform = get_pose(res);
		m.push_back(mes);
	}

}

void util_mysql::load_positions(int map_id, std::vector<position> & p) {
	select_positions->setInt(1, map_id);
	boost::shared_ptr<sql::ResultSet> res(select_positions->executeQuery());


	while (res->next()) {
		position pos;
		pos.idx = res->getInt64("id");
		pos.transform = get_pose(res);
		p.push_back(pos);
	}

}

void util_mysql::update_position(const position & p) {
	update_keyframe->setDouble(1, p.transform.unit_quaternion().x());
	update_keyframe->setDouble(2, p.transform.unit_quaternion().y());
	update_keyframe->setDouble(3, p.transform.unit_quaternion().z());
	update_keyframe->setDouble(4, p.transform.unit_quaternion().w());

	update_keyframe->setDouble(5, p.transform.translation().x());
	update_keyframe->setDouble(6, p.transform.translation().y());
	update_keyframe->setDouble(7, p.transform.translation().z());

	update_keyframe->setInt64(8, p.idx);

	update_keyframe->executeUpdate();

}

long util_mysql::get_random_keyframe_idx(int map_id) {
	select_random_idx->setInt(1, map_id);

	boost::shared_ptr<sql::ResultSet> res(
			select_random_idx->executeQuery());
	res->next();
	return res->getInt64("id");

}

void util_mysql::merge_map(int old_map_id, int new_map_id){
	update_robot_map_id->setInt(1, new_map_id);
	update_robot_map_id->setInt(2, old_map_id);
	update_robot_map_id->executeUpdate();

	update_keyframe_map_id->setInt(1, new_map_id);
	update_keyframe_map_id->setInt(2, old_map_id);
	update_keyframe_map_id->executeUpdate();

}


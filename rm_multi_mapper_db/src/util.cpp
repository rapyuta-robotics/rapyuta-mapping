#include <util.h>

using namespace std;

util::util() {
	// TODO make arguments
	server = "localhost";
	user = "mapping";
	password = "mapping";
	database = "mapping";

	driver = get_driver_instance();
	con = driver->connect("tcp://" + server + ":3306", user, password);
	con->setSchema(database);

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

util::~util() {
	delete con;
}

sql::ResultSet* util::sql_query(std::string query) {
	try {
		sql::PreparedStatement *pstmt;
		sql::ResultSet *res;

		/* Select in ascending order */
		pstmt = con->prepareStatement(query);
		res = pstmt->executeQuery();

		delete pstmt;
		return res;

	} catch (sql::SQLException &e) {
		std::cout << "# ERR: SQLException in " << __FILE__;
		std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__
				<< std::endl;
		std::cout << "# ERR: " << e.what();
		std::cout << " (MySQL error code: " << e.getErrorCode();
		std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
		return NULL;
	}
}

int util::get_new_robot_id() {
	sql::ResultSet *res;
	res = sql_query("INSERT INTO robot (id, map_id) VALUES(NULL, NULL);");
	res = sql_query("SELECT LAST_INSERT_ID() as id");
	res->next();
	int robot_id = res->getInt("id");
	res = sql_query(
			"UPDATE robot SET map_id = "
					+ boost::lexical_cast<std::string>(robot_id)
					+ " WHERE id = "
					+ boost::lexical_cast<std::string>(robot_id));
	return robot_id;

}

void util::add_keyframe(int robot_id, const color_keyframe::Ptr & k) {

	try {

		sql::ResultSet *res;
		res = sql_query(
				"SELECT map_id FROM robot WHERE id = "
						+ boost::lexical_cast<std::string>(robot_id));

		res->next();
		int map_id = res->getInt("map_id");

		sql::PreparedStatement *pstmt = con->prepareStatement(
				"INSERT INTO keyframe "
						"(`id`, `q0`, `q1`, `q2`, `q3`,"
						" `t0`, `t1`, `t2`, `int0`, `int1`,"
						" `int2`, `rgb`, `depth`, `map_id`) "
						"VALUES "
						"(?,?,?,?,?"
						",?,?,?,?,?"
						",?,?,?,?)");

		pstmt->setInt64(1, k->get_id());
		pstmt->setDouble(2, k->get_pos().unit_quaternion().x());
		pstmt->setDouble(3, k->get_pos().unit_quaternion().y());
		pstmt->setDouble(4, k->get_pos().unit_quaternion().z());
		pstmt->setDouble(5, k->get_pos().unit_quaternion().w());

		pstmt->setDouble(6, k->get_pos().translation().x());
		pstmt->setDouble(7, k->get_pos().translation().y());
		pstmt->setDouble(8, k->get_pos().translation().z());

		pstmt->setDouble(9, k->get_intrinsics()[0]);
		pstmt->setDouble(10, k->get_intrinsics()[1]);
		pstmt->setDouble(11, k->get_intrinsics()[2]);

		std::vector<uint8_t> rgb_data, depth_data;

		cv::imencode(".png", k->get_rgb(), rgb_data);
		DataBuf rgb_buffer((char*) rgb_data.data(), rgb_data.size());
		std::istream rgb_stream(&rgb_buffer);
		//std::cerr << "Write rgb data size " << rgb_data.size() << std::endl;

		cv::imencode(".png", k->get_d(0), depth_data);
		DataBuf depth_buffer((char*) depth_data.data(), depth_data.size());
		std::istream depth_stream(&depth_buffer);
		//std::cerr << "Write depth data size " << depth_data.size() << std::endl;

		pstmt->setBlob(12, &rgb_stream);
		pstmt->setBlob(13, &depth_stream);
		pstmt->setInt(14, map_id);

		pstmt->executeUpdate();

		delete pstmt;

	} catch (sql::SQLException &e) {
		std::cout << "# ERR: SQLException in " << __FILE__;
		std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__
				<< std::endl;
		std::cout << "# ERR: " << e.what();
		std::cout << " (MySQL error code: " << e.getErrorCode();
		std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
	}

}

void util::add_keypoints(const color_keyframe::Ptr & k) {
	try {

		sql::PreparedStatement *pstmt =
				con->prepareStatement(
						"UPDATE keyframe SET"
								" `num_keypoints`=? , `descriptor_size`= ?, `descriptor_type`=?,"
								" `keypoints`= ?, `descriptors`=? "
								" WHERE `id` = ? ");

		std::vector<cv::KeyPoint> keypoints;
		pcl::PointCloud<pcl::PointXYZ> keypoints3d;
		cv::Mat descriptors;
		compute_features(k->get_i(0), k->get_d(0), k->get_intrinsics(0),
				keypoints, keypoints3d, descriptors);

		pstmt->setInt(1, keypoints3d.size());
		pstmt->setInt(2, descriptors.cols);
		pstmt->setInt(3, descriptors.type());

		assert(descriptors.type() == CV_32F);
		std::cerr << "Keypoints size " << keypoints3d.size() << " "
				<< descriptors.size() << std::endl;

		DataBuf keypoints_buffer((char*) keypoints3d.points.data(),
				keypoints3d.points.size() * sizeof(pcl::PointXYZ));
		std::istream keypoints_stream(&keypoints_buffer);

		DataBuf descriptors_buffer((char*) descriptors.data,
				descriptors.cols * descriptors.rows * sizeof(float));
		std::istream descriptors_stream(&descriptors_buffer);

		//std::cerr << "Decriptors size " << descriptors.cols * descriptors.rows * sizeof(float) << std::endl;

		pstmt->setBlob(4, &keypoints_stream);
		pstmt->setBlob(5, &descriptors_stream);
		pstmt->setInt64(6, k->get_id());

		pstmt->executeUpdate();

		delete pstmt;

	} catch (sql::SQLException &e) {
		std::cout << "# ERR: SQLException in " << __FILE__;
		std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__
				<< std::endl;
		std::cout << "# ERR: " << e.what();
		std::cout << " (MySQL error code: " << e.getErrorCode();
		std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
	}
}

void util::add_measurement(long int first, long int second,
		const Sophus::SE3f & transform, const std::string & type) {
	try {

		sql::PreparedStatement *pstmt =
				con->prepareStatement(
						"INSERT INTO measurement"
								" (`id`, `one`, `two`, `q0`, `q1`, `q2`, `q3`, `t0`, `t1`, `t2`, `type`)"
								" VALUES"
								" (NULL,?,?,?,?,?,?,?,?,?,?)");

		pstmt->setInt64(1, first);
		pstmt->setInt64(2, second);

		pstmt->setDouble(3, transform.unit_quaternion().x());
		pstmt->setDouble(4, transform.unit_quaternion().y());
		pstmt->setDouble(5, transform.unit_quaternion().z());
		pstmt->setDouble(6, transform.unit_quaternion().w());

		pstmt->setDouble(7, transform.translation().x());
		pstmt->setDouble(8, transform.translation().y());
		pstmt->setDouble(9, transform.translation().z());

		pstmt->setString(10, type);

		pstmt->executeUpdate();

		delete pstmt;

	} catch (sql::SQLException &e) {
		std::cout << "# ERR: SQLException in " << __FILE__;
		std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__
				<< std::endl;
		std::cout << "# ERR: " << e.what();
		std::cout << " (MySQL error code: " << e.getErrorCode();
		std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
	}
}

color_keyframe::Ptr util::get_keyframe(long frame_id) {
	sql::ResultSet *res;
	res = sql_query(
			"SELECT * FROM keyframe WHERE id = "
					+ boost::lexical_cast<std::string>(frame_id));
	res->next();
	return get_keyframe(res);

}

Sophus::SE3f util::get_pose(sql::ResultSet * res) {
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

color_keyframe::Ptr util::get_keyframe(sql::ResultSet * res) {

	Sophus::SE3f pose;
	pose = get_pose(res);

	Eigen::Vector3f intrinsics;

	intrinsics[0] = res->getDouble("int0");
	intrinsics[1] = res->getDouble("int1");
	intrinsics[2] = res->getDouble("int2");

	std::vector<uint8_t> rgb_data, depth_data;
	std::istream * rgb_in = res->getBlob("rgb");
	while (*rgb_in) {
		uint8_t tmp;
		rgb_in->read((char*) &tmp, sizeof(tmp));
		rgb_data.push_back(tmp);
	}
	delete rgb_in;

	//std::cerr << "Read rgb data size " << rgb_data.size() << std::endl;

	std::istream * depth_in = res->getBlob("depth");
	while (*depth_in) {
		uint8_t tmp;
		depth_in->read((char*) &tmp, sizeof(tmp));
		depth_data.push_back(tmp);
	}
	delete depth_in;

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

void util::get_keypoints(long frame_id,
		pcl::PointCloud<pcl::PointXYZ> & keypoints3d, cv::Mat & descriptors) {
	sql::ResultSet *res;
	res = sql_query(
			"SELECT * FROM keyframe WHERE id = "
					+ boost::lexical_cast<std::string>(frame_id));
	res->next();

	keypoints3d.clear();
	std::istream * keypoints_in = res->getBlob("keypoints");
	while (*keypoints_in) {
		pcl::PointXYZ tmp;
		keypoints_in->read((char*) &tmp, sizeof(tmp));
		keypoints3d.push_back(tmp);
	}
	delete keypoints_in;
	keypoints3d.resize(keypoints3d.size() - 1);

	std::istream * descriptors_in = res->getBlob("descriptors");
	std::vector<uint8_t> descriptors_data;

	while (*descriptors_in) {
		uint8_t tmp;
		descriptors_in->read((char*) &tmp, sizeof(tmp));
		descriptors_data.push_back(tmp);
	}
	delete descriptors_in;
	descriptors_data.resize(descriptors_data.size() - 1);

	int cols = res->getDouble("descriptor_size");
	int rows = res->getDouble("num_keypoints");
	int type = res->getDouble("descriptor_type");

	//std::cerr << "Creating matrix " << cols << " " << rows << " " << type << " "
	//		<< descriptors_data.size() << std::endl;

	cv::Mat tmp_mat = cv::Mat(rows, cols, type,
			(void *) descriptors_data.data());

	//std::cerr << "Matrix size " << tmp_mat.size() << std::endl;

	tmp_mat.copyTo(descriptors);
	delete res;

}

boost::shared_ptr<keyframe_map> util::get_robot_map(int robot_id) {
	sql::ResultSet *res;
	res =
			sql_query(
					"SELECT * FROM keyframe WHERE map_id = ( SELECT map_id FROM robot WHERE id = "
							+ boost::lexical_cast<std::string>(robot_id)
							+ " )");

	boost::shared_ptr<keyframe_map> map(new keyframe_map);

	while (res->next()) {
		map->frames.push_back(get_keyframe(res));
	}

	return map;
}

void util::compute_features(const cv::Mat & rgb, const cv::Mat & depth,
		const Eigen::Vector3f & intrinsics,
		std::vector<cv::KeyPoint> & filtered_keypoints,
		pcl::PointCloud<pcl::PointXYZ> & keypoints3d, cv::Mat & descriptors) {
	cv::Mat gray;

	if (rgb.channels() != 1) {
		cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);
	} else {
		gray = rgb.clone();
	}

	cv::GaussianBlur(gray, gray, cv::Size(3, 3), 3);

	int threshold = 400;
	fd->setInt("hessianThreshold", threshold);

	//int threshold = 100;
	//fd->setInt("thres", threshold);

	std::vector<cv::KeyPoint> keypoints;

	cv::Mat mask(depth.size(), CV_8UC1);
	depth.convertTo(mask, CV_8U);

	fd->detect(gray, keypoints, mask);

	for (int i = 0; i < 5; i++) {
		if (keypoints.size() < 300) {
			threshold = threshold / 2;
			fd->setInt("hessianThreshold", threshold);
			//fd->setInt("thres", threshold);
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
			p.x = (keypoints[i].pt.x - intrinsics[1]) * p.z / intrinsics[0];
			p.y = (keypoints[i].pt.y - intrinsics[2]) * p.z / intrinsics[0];

			//ROS_INFO("Point %f %f %f from  %f %f ", p.x, p.y, p.z, keypoints[i].pt.x, keypoints[i].pt.y);

			keypoints3d.push_back(p);

		}
	}

	de->compute(gray, filtered_keypoints, descriptors);
}

void util::get_overlapping_pairs(int map_id,
		std::vector<std::pair<long, long> > & overlapping_keyframes) {
	sql::ResultSet *res;

	std::string map_id_s = boost::lexical_cast<std::string>(map_id);

	res = sql_query(""
			"SELECT f1.id as id1, f2.id as id2 "
			"FROM keyframe f1, keyframe f2 "
			"WHERE f1.map_id = " + map_id_s + " "
			"AND f2.map_id = " + map_id_s + " "
			"AND (abs(f1.q0*f2.q0 + f1.q1*f2.q1 + f1.q2*f2.q2"
			" + f1.q3*f2.q3) >= 1.0 OR 2*acos(abs(f1.q0*f2.q0 + f1.q1*f2.q1 +"
			" f1.q2*f2.q2 + f1.q3*f2.q3)) < pi()/4) "
			"AND f1.id < f2.id "
			"AND SQRT(POWER((f1.t0 - f2.t0), 2) + "
			"POWER((f1.t1 - f2.t1), 2) + "
			"POWER((f1.t2 - f2.t2), 2)) < 3;");

	while (res->next()) {
		overlapping_keyframes.push_back(
				std::make_pair(res->getInt64("id1"), res->getInt64("id2")));
	}

	delete res;

}

void util::load_measurements(long keyframe_id, std::vector<measurement> & m) {

	sql::ResultSet *res;
	res = sql_query(
			"SELECT * FROM measurement WHERE measurement.one = "
					+ boost::lexical_cast<std::string>(keyframe_id));

	while (res->next()) {
		measurement mes;
		mes.first = res->getInt64("one");
		mes.second = res->getInt64("two");
		mes.transform = get_pose(res);
		m.push_back(mes);
	}

}

void util::load_positions(int map_id, std::vector<position> & p) {
	sql::ResultSet *res;
	res = sql_query(
			"SELECT * FROM keyframe WHERE map_id = "
					+ boost::lexical_cast<std::string>(map_id));

	boost::shared_ptr<keyframe_map> map(new keyframe_map);

	while (res->next()) {
		position pos;
		pos.idx = res->getInt64("id");
		pos.transform = get_pose(res);
		p.push_back(pos);
	}

}

void util::update_position(const position & p) {
	sql::PreparedStatement *pstmt =
			con->prepareStatement(
					"UPDATE keyframe SET `q0`= ?, `q1`= ?, `q2`= ?, `q3`= ?, `t0`= ?, `t1`= ?, `t2`= ? WHERE id = ?");

	pstmt->setDouble(1, p.transform.unit_quaternion().x());
	pstmt->setDouble(2, p.transform.unit_quaternion().y());
	pstmt->setDouble(3, p.transform.unit_quaternion().z());
	pstmt->setDouble(4, p.transform.unit_quaternion().w());

	pstmt->setDouble(5, p.transform.translation().x());
	pstmt->setDouble(6, p.transform.translation().y());
	pstmt->setDouble(7, p.transform.translation().z());

	pstmt->setInt64(8, p.idx);

	pstmt->executeUpdate();

}


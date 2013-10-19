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

		sql::PreparedStatement *pstmt =
				con->prepareStatement(
						"INSERT INTO keyframe (`id`, `q0`, `q1`, `q2`, `q3`, `t0`, `t1`, `t2`, `int0`, `int1`, `int2`, `rgb`, `depth`, `map_id`) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)");

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

color_keyframe::Ptr util::get_keyframe(long frame_id) {
	sql::ResultSet *res;
	res = sql_query(
			"SELECT * FROM keyframe WHERE id = "
					+ boost::lexical_cast<std::string>(frame_id));
	res->next();
	return get_keyframe(res);

}

color_keyframe::Ptr util::get_keyframe(sql::ResultSet * res) {
	Eigen::Quaternionf q;
	Eigen::Vector3f t;
	Eigen::Vector3f intrinsics;
	q.x() = res->getDouble("q0");
	q.y() = res->getDouble("q1");
	q.z() = res->getDouble("q2");
	q.w() = res->getDouble("q3");
	t[0] = res->getDouble("t0");
	t[1] = res->getDouble("t1");
	t[2] = res->getDouble("t2");
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

	//std::cerr << "Read rgb data size " << rgb_data.size() << std::endl;

	std::istream * depth_in = res->getBlob("depth");
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
			new color_keyframe(rgb, gray, depth, Sophus::SE3f(q, t),
					intrinsics));

	k->set_id(res->getDouble("id"));

	return k;
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

void util::load_measurements(std::vector<measurement> &m) {

	sql::ResultSet *res;
	res = sql_query("SELECT * FROM measurement");
	int i = 0;
	while (res->next()) {
		m.push_back(measurement());
		Eigen::Quaternionf q;
		Eigen::Vector3f t;

		m[i].i = res->getInt64("one");
		m[i].j = res->getInt64("two");

		q.coeffs()[0] = res->getDouble("q0");
		q.coeffs()[1] = res->getDouble("q1");
		q.coeffs()[2] = res->getDouble("q2");
		q.coeffs()[3] = res->getDouble("q3");
		t[0] = res->getDouble("t0");
		t[1] = res->getDouble("t1");
		t[2] = res->getDouble("t2");

		m[i].transform = Sophus::SE3f(q, t);
		m[i].mt = (measurement_type) res->getInt64("type");
		i++;

	}
}
void util::save_measurements(const std::vector<measurement> &m) {
	for (int i = 0; i < m.size(); i++) {
		try {
			sql::PreparedStatement *pstmt = con->prepareStatement(
					"INSERT INTO measurement (`one`, `two`, `q0`, `q1`, "
							"`q2`, `q3`, `t0`, `t1`, `t2`, `type`) VALUES "
							"(?,?,?,?,?,?,?,?,?,?)");

			pstmt->setInt64(1, m[i].i);
			pstmt->setDouble(2, m[i].j);
			pstmt->setDouble(3, m[i].transform.so3().data()[0]);
			pstmt->setDouble(4, m[i].transform.so3().data()[1]);
			pstmt->setDouble(5, m[i].transform.so3().data()[2]);
			pstmt->setDouble(6, m[i].transform.so3().data()[3]);
			pstmt->setDouble(7, m[i].transform.translation()[0]);
			pstmt->setDouble(8, m[i].transform.translation()[1]);

			pstmt->setDouble(9, m[i].transform.translation()[2]);
			pstmt->setInt64(10, m[i].mt);

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
}



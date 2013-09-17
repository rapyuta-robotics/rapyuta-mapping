#include <util.h>

using Poco::URIStreamOpener;
using Poco::StreamCopier;
using Poco::Path;
using Poco::URI;
using Poco::Exception;
using Poco::Net::HTTPStreamFactory;
using namespace std;

util::util() {
	driver = get_driver_instance();
	con = driver->connect("tcp://localhost:3306", "mapping", "123456");
	con->setSchema("mapping");
}

util::~util() {
	delete con;

}

cv::Mat util::loadFromURL(string url) {
	//Don't register the factory more than once
	if (!factoryLoaded) {
		HTTPStreamFactory::registerFactory();
		factoryLoaded = true;
	}

	//Specify URL and open input stream
	URI uri(url);
	std::auto_ptr<std::istream> pStr(
			URIStreamOpener::defaultOpener().open(uri));

	//Copy image to our string and convert to cv::Mat
	string str;
	StreamCopier::copyToString(*pStr.get(), str);
	vector<char> data(str.begin(), str.end());
	cv::Mat data_mat(data);
	cv::Mat image(cv::imdecode(data_mat, 1));
	return image;
}

cv::Mat util::stringtoMat(string file) {
	cv::Mat image;

	if (file.compare(file.size() - 4, 4, ".gif") == 0) {
		cerr << "UNSUPPORTED_IMAGE_FORMAT";
		return image;
	}

	else if (file.compare(0, 7, "http://") == 0) // Valid URL only if it starts with "http://"
			{
		image = loadFromURL(file);
		if (!image.data)
			cerr << "INVALID_IMAGE_URL";
		return image;
	} else {
		image = cv::imread(file, 1); // Try if the image path is in the local machine
		if (!image.data)
			cerr << "IMAGE_DOESNT_EXIST";
		return image;
	}
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

void util::load_mysql(
		std::vector<std::pair<Sophus::SE3f, Eigen::Vector3f> > & positions) {

	sql::ResultSet *res;
	res = sql_query("SELECT * FROM positions");

	while (res->next()) {
		Eigen::Quaternionf q;
		Eigen::Vector3f t;
		Eigen::Vector3f intrinsics;
		q.coeffs()[0] = res->getDouble("q0");
		q.coeffs()[1] = res->getDouble("q1");
		q.coeffs()[2] = res->getDouble("q2");
		q.coeffs()[3] = res->getDouble("q3");
		t[0] = res->getDouble("t0");
		t[1] = res->getDouble("t1");
		t[2] = res->getDouble("t2");
		intrinsics[0] = res->getDouble("int0");
		intrinsics[1] = res->getDouble("int1");
		intrinsics[2] = res->getDouble("int2");
		positions.push_back(std::make_pair(Sophus::SE3f(q, t), intrinsics));
	}

	delete res;

}

void util::load(const std::string & dir_name,
		std::vector<color_keyframe::Ptr> & frames) {

	std::vector<std::pair<Sophus::SE3f, Eigen::Vector3f> > positions;

	load_mysql(positions);

	std::cerr << "Loaded " << positions.size() << " positions" << std::endl;

	for (size_t i = 0; i < positions.size(); i++) {
		cv::Mat rgb = stringtoMat(
				dir_name + "/rgb/" + boost::lexical_cast<std::string>(i)
						+ ".png");
		cv::Mat depth = stringtoMat(
				dir_name + "/depth/" + boost::lexical_cast<std::string>(i)
						+ ".png");

		cv::Mat gray;
		cv::cvtColor(rgb, gray, CV_RGB2GRAY);

		color_keyframe::Ptr k(
				new color_keyframe(rgb, gray, depth, positions[i].first,
						positions[i].second));
		frames.push_back(k);
	}
	std::cout << "Ready" << std::endl;
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
					new color_keyframe(rgb, gray, depth, Sophus::SE3f(q,t),
							intrinsics));

	return k;

}


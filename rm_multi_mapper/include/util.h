
#include <memory>
#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include "opencv2/opencv.hpp"

/*MySQL includes */
#include "mysql_connection.h"
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/prepared_statement.h>

#include <keyframe_map.h>
#include <reduce_measurement_g2o_dist.h>

class DataBuf: public streambuf {
public:
	DataBuf(char * d, size_t s) {
		setg(d, d, d + s);
	}
};

class util {
public:

	util();
	~util();

	sql::ResultSet* sql_query(std::string query);
	void save_measurements(const std::vector<measurement> &m);
	void load_measurements(std::vector<measurement> &m);

	int get_new_robot_id();
	void add_keyframe(int robot_id, const color_keyframe::Ptr & k);
	void add_measurement(long first, long second,
			const Sophus::SE3f & transform, const std::string & type);

	color_keyframe::Ptr get_keyframe(long frame_id);

	boost::shared_ptr<keyframe_map> get_robot_map(int robot_id);

private:

	std::string server;
	std::string user;
	std::string password;
	std::string database;

	sql::Driver *driver;
	sql::Connection *con;

	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;

	color_keyframe::Ptr get_keyframe(sql::ResultSet * res);
};


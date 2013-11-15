
#ifndef UTIL_MYSQL_H
#define UTIL_MYSQL_H

#include <util.h>

#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <mysql_connection.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/prepared_statement.h>


class util_mysql : public util {
public:

	util_mysql();
	~util_mysql();

	int get_new_robot_id();
	void add_keyframe(int robot_id, const color_keyframe::Ptr & k);
	void add_measurement(long first, long second,
			const Sophus::SE3f & transform, const std::string & type);

	void add_keypoints(const color_keyframe::Ptr & k);
	void get_keypoints(long frame_id,
			pcl::PointCloud<pcl::PointXYZ> & keypoints3d,
			cv::Mat & desctriptors);

	color_keyframe::Ptr get_keyframe(long frame_id);

	boost::shared_ptr<keyframe_map> get_robot_map(int robot_id);

	void get_overlapping_pairs(int map_id,
			std::vector<std::pair<long, long> > & overlapping_keyframes);

	void load_measurements(long keyframe_id, std::vector<measurement> & m);
	void load_positions(int map_id, std::vector<position> & p);
	void update_position(const position & p);

	void compute_features(const cv::Mat & rgb, const cv::Mat & depth,
			const Eigen::Vector3f & intrinsics,
			std::vector<cv::KeyPoint> & filtered_keypoints,
			pcl::PointCloud<pcl::PointXYZ> & keypoints3d,
			cv::Mat & descriptors);


private:

	class DataBuf: public streambuf {
	public:
		DataBuf(char * d, size_t s) {
			setg(d, d, d + s);
		}
	};


	std::string server;
	std::string user;
	std::string password;
	std::string database;

	sql::Driver * driver;
	boost::shared_ptr<sql::Connection> con;

	boost::shared_ptr<sql::PreparedStatement> get_map_id_from_robot_id;
	boost::shared_ptr<sql::PreparedStatement> insert_keyframe;
	boost::shared_ptr<sql::PreparedStatement> insert_keypoints;
	boost::shared_ptr<sql::PreparedStatement> insert_measurement;
	boost::shared_ptr<sql::PreparedStatement> select_keyframe;
	boost::shared_ptr<sql::PreparedStatement> select_keypoints;
	boost::shared_ptr<sql::PreparedStatement> select_map;
	boost::shared_ptr<sql::PreparedStatement> select_positions;
	boost::shared_ptr<sql::PreparedStatement> select_measurements;

	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;

	sql::ResultSet* sql_query(std::string query);
	Sophus::SE3f get_pose(boost::shared_ptr<sql::ResultSet> & res);
	color_keyframe::Ptr get_keyframe(boost::shared_ptr<sql::ResultSet> & res);
};

#endif

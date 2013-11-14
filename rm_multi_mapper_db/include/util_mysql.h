
#ifndef UTIL_MYSQL_H
#define UTIL_MYSQL_H

#include <util.h>
#include <memory>
#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

/*MySQL includes */
#include <mysql_connection.h>
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/prepared_statement.h>

#include <keyframe_map.h>
#include <reduce_measurement_g2o_dist.h>


class util_mysql : public util {
public:

	util_mysql();
	virtual ~util_mysql();

	virtual int get_new_robot_id();
	virtual void add_keyframe(int robot_id, const color_keyframe::Ptr & k);
	virtual void add_measurement(long first, long second,
			const Sophus::SE3f & transform, const std::string & type);

	virtual void add_keypoints(const color_keyframe::Ptr & k);
	virtual void get_keypoints(long frame_id,
			pcl::PointCloud<pcl::PointXYZ> & keypoints3d,
			cv::Mat & desctriptors);

	virtual color_keyframe::Ptr get_keyframe(long frame_id);

	virtual boost::shared_ptr<keyframe_map> get_robot_map(int robot_id);

	virtual void get_overlapping_pairs(int map_id,
			std::vector<std::pair<long, long> > & overlapping_keyframes);

	virtual void load_measurements(long keyframe_id, std::vector<measurement> & m);
	virtual void load_positions(int map_id, std::vector<position> & p);
	virtual void update_position(const position & p);

	virtual void compute_features(const cv::Mat & rgb, const cv::Mat & depth,
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

	sql::Driver *driver;
	sql::Connection *con;

	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;

	sql::ResultSet* sql_query(std::string query);
	Sophus::SE3f get_pose(sql::ResultSet * res);
	color_keyframe::Ptr get_keyframe(sql::ResultSet * res);
};

#endif

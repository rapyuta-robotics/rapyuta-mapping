
#ifndef UTIL_MONGO_H
#define UTIL_MONGO_H

#include <util.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <ctime>
#include <cassert>
#include "mongo/client/dbclient.h"

class util_mongo : public util {
public:

	util_mongo();
	~util_mongo();

	int getNextSequence(std::string name);
	int get_mapid(int robot_id);
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
	long long get_random_keyframe_idx(int map);
	void merge_map(int old_map_id, int new_map_id);


private:

	mongo::DBClientConnection conn;
	color_keyframe::Ptr get_keyframe(mongo::BSONObj res);
	Sophus::SE3f get_pose(mongo::BSONObj res);

};

#endif

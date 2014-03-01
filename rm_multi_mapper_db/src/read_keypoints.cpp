#include <util.h>
#include <util_mysql.h>
#include <util_mongo.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>
#include <pcl/visualization/pcl_visualizer.h>

#include <keyframe_map.h>

int main(int argc, char **argv) {

	int map_id = boost::lexical_cast<int>(argv[1]);

#ifdef MONGO
	util::Ptr U(new util_mongo);
#else
	util::Ptr U(new util_mysql);
#endif
	boost::shared_ptr<keyframe_map> map = U->get_robot_map(map_id);
	pcl::PointCloud<pcl::PointXYZ> keypoints3d1;
	cv::Mat descriptors1;
	for(int i = 0; i < map->frames.size(); i++)
	{
		U->get_keypoints(map->frames[i]->get_id(), keypoints3d1, descriptors1);
	}

	return 0;

}


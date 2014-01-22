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

	ros::init(argc, argv, "show_map");
	ros::NodeHandle nh;

	ros::Publisher pointcloud_pub = nh.advertise<
			pcl::PointCloud<pcl::PointXYZRGB> >("/pointcloud", 1);

#ifdef MONGO
	util::Ptr U(new util_mongo);
#else
	util::Ptr U(new util_mysql);
#endif
	boost::shared_ptr<keyframe_map> map = U->get_robot_map(map_id);

	std::cerr << "Map size " << map->frames.size() << std::endl;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map->get_map_pointcloud();

	cloud->header.frame_id = "/world";
	cloud->header.stamp = ros::Time::now();
	cloud->header.seq = 0;
	pointcloud_pub.publish(cloud);

	return 0;

}

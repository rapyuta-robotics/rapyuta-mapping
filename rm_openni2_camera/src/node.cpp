
#include <openni2_camera.h>

int main(int argc, char** argv) {

	ros::init(argc, argv, "camera");
	ros::NodeHandle nh;
	ros::NodeHandle nh_private("~");

	ROS_INFO("Initializing camera node");
	OpenNI2Camera pc(nh, nh_private);

	ros::spin();

}

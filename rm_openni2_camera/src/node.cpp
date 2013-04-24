
#include <openni2_camera.h>

int main(int argc, char** argv) {

	ros::init(argc, argv, "camera");
	ros::NodeHandle nh;

	OpenNI2Camera pc(nh);

	ros::spin();

}

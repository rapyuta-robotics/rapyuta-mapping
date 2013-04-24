
#include <openni2_camera.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>


class OpenNI2CameraNodelet: public nodelet::Nodelet {
public:
	virtual void onInit() {
		nh = getMTNodeHandle();
		pc.reset(new OpenNI2Camera(nh));

	}

private:
	ros::NodeHandle nh;
	boost::shared_ptr<OpenNI2Camera> pc;

};


// watch the capitalization carefully
PLUGINLIB_DECLARE_CLASS(rm_openni2_camera, OpenNI2CameraNodelet,
		OpenNI2CameraNodelet, nodelet::Nodelet)


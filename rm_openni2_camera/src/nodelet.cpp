
#include <openni2_camera.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>


class OpenNI2CameraNodelet: public nodelet::Nodelet {
public:
	virtual void onInit() {
		nh = getNodeHandle();
		nh_private = getPrivateNodeHandle();
		pc.reset(new OpenNI2Camera(nh, nh_private));

	}

private:
	ros::NodeHandle nh, nh_private;
	boost::shared_ptr<OpenNI2Camera> pc;

};


// watch the capitalization carefully
PLUGINLIB_DECLARE_CLASS(rm_openni2_camera, OpenNI2CameraNodelet,
		OpenNI2CameraNodelet, nodelet::Nodelet)


/*
 * openni2_camera.h
 *
 *  Created on: Apr 24, 2013
 *      Author: vsu
 */

#ifndef OPENNI2_CAMERA_H_
#define OPENNI2_CAMERA_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/distortion_models.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <OpenNI.h>
#include <frame_callback.h>

using namespace openni;

class OpenNI2Camera {
public:

	OpenNI2Camera(ros::NodeHandle & nh, ros::NodeHandle & nh_private);
	virtual ~OpenNI2Camera();

private:
	VideoFrameRef m_frame;
	VideoStream depth, color;
	Device device;

	boost::shared_ptr<FrameCallback> dc, rgbc;

};



#endif /* OPENNI2_CAMERA_H_ */

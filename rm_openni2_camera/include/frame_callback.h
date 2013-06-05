/*
 * frame_callback.h
 *
 *  Created on: Apr 24, 2013
 *      Author: vsu
 */

#ifndef FRAME_CALLBACK_H_
#define FRAME_CALLBACK_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/distortion_models.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include <OpenNI.h>

using namespace openni;

class FrameCallback: public VideoStream::NewFrameListener {
public:

	FrameCallback(ros::NodeHandle & nh, ros::NodeHandle & nh_private, const std::string & camera_name);
	virtual ~FrameCallback();
	void onNewFrame(VideoStream& stream);

private:

	sensor_msgs::CameraInfoPtr getDefaultCameraInfo(int width, int height,
			double f) const;

	ros::NodeHandle cam_nh;
	image_transport::ImageTransport cam_it;
	camera_info_manager::CameraInfoManager cim;
	image_transport::CameraPublisher pub;
	VideoFrameRef m_frame;

	std::string camera_name;
	sensor_msgs::CameraInfoPtr info;
	sensor_msgs::ImagePtr msg;

	std::string frame;

};

#endif /* FRAME_CALLBACK_H_ */

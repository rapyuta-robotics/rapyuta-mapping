#include <frame_callback.h>

FrameCallback::FrameCallback(ros::NodeHandle & nh,
		const std::string & camera_name) :
		cam_nh(nh, camera_name), cam_it(cam_nh), cim(cam_nh, camera_name,
				"file://${ROS_HOME}/camera_info/${NAME}.yaml"), info(
				new sensor_msgs::CameraInfo) {

	ROS_INFO("Creating callback for camera %s", camera_name.c_str());
	pub = cam_it.advertiseCamera("image_raw", 1);
	this->camera_name = camera_name;

	if (cim.isCalibrated()) {
		*info = cim.getCameraInfo();
	} else {
		//if (camera_name == "depth") {
		//	info = getDefaultCameraInfo(640, 480, 570.0);
		//} else {
			info = getDefaultCameraInfo(640/2, 480/2, 525.0/2);
		//}
	}

}

FrameCallback::~FrameCallback() {
	ROS_INFO("Desroying callback for camera %s", this->camera_name.c_str());
}

void FrameCallback::onNewFrame(VideoStream& stream) {
	stream.readFrame(&m_frame);


	msg.reset(new sensor_msgs::Image);

	msg->header.frame_id = "/camera_rgb_optical_frame";
	msg->header.stamp = ros::Time::now();
	msg->header.seq = m_frame.getFrameIndex();
	msg->width = m_frame.getWidth();
	msg->height = m_frame.getHeight();
	msg->step = m_frame.getStrideInBytes();

	switch (m_frame.getVideoMode().getPixelFormat()) {

	case PIXEL_FORMAT_DEPTH_1_MM:
		msg->encoding = sensor_msgs::image_encodings::TYPE_16UC1;
		break;

	case PIXEL_FORMAT_YUV422:
		msg->encoding = sensor_msgs::image_encodings::YUV422;
		break;

	case PIXEL_FORMAT_RGB888:
		msg->encoding = sensor_msgs::image_encodings::RGB8;
		break;

	default:
		ROS_INFO("Unsupported encoding\n");
		break;
	}

	msg->data.resize(m_frame.getDataSize());
	memcpy(msg->data.data(), m_frame.getData(), m_frame.getDataSize());

	info->header = msg->header;
	pub.publish(msg, info);
}

sensor_msgs::CameraInfoPtr FrameCallback::getDefaultCameraInfo(int width,
		int height, double f) const {
	sensor_msgs::CameraInfoPtr info =
			boost::make_shared<sensor_msgs::CameraInfo>();

	info->width = width;
	info->height = height;

	// No distortion
	info->D.resize(5, 0.0);
	info->distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;

	// Simple camera matrix: square pixels (fx = fy), principal point at center
	info->K.assign(0.0);
	info->K[0] = info->K[4] = f;
	info->K[2] = (width / 2) - 0.5;
	// Aspect ratio for the camera center on Kinect (and other devices?) is 4/3
	// This formula keeps the principal point the same in VGA and SXGA modes
	info->K[5] = (width * (3. / 8.)) - 0.5;
	info->K[8] = 1.0;

	// No separate rectified image plane, so R = I
	info->R.assign(0.0);
	info->R[0] = info->R[4] = info->R[8] = 1.0;

	// Then P=K(I|0) = (K|0)
	info->P.assign(0.0);
	info->P[0] = info->P[5] = f; // fx, fy
	info->P[2] = info->K[2]; // cx
	info->P[6] = info->K[5]; // cy
	info->P[10] = 1.0;

	return info;
}


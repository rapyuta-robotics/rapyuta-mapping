#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/distortion_models.h>
#include <image_transport/image_transport.h>
#include <camera_info_manager/camera_info_manager.h>
#include <OpenNI.h>
#include <cstring>

using namespace openni;

class FrameCallback: public VideoStream::NewFrameListener {
public:

	FrameCallback(ros::NodeHandle & nh, image_transport::ImageTransport & it,
			const std::string & camera_name) :
			cim(nh, camera_name, "file://${ROS_HOME}/camera_info/${NAME}.yaml"), info(
					new sensor_msgs::CameraInfo), msg(new sensor_msgs::Image) {
		pub = it.advertiseCamera("/" + camera_name + "/image_raw", 1);
		counter = 0;
		this->camera_name = camera_name;

		if (cim.isCalibrated()) {
			*info = cim.getCameraInfo();
		} else {
			if (camera_name == "depth") {
				info = getDefaultCameraInfo(640, 480, 570.0);
			} else {
				info = getDefaultCameraInfo(640, 480, 525.0);
			}
		}

	}

	virtual ~FrameCallback() {
	}

	void onNewFrame(VideoStream& stream) {
		stream.readFrame(&m_frame);

		msg->header.frame_id = "camera_" + camera_name + "_optical_frame";
		msg->header.stamp = ros::Time::now();
		msg->header.seq = counter++;
		msg->width = m_frame.getWidth();
		msg->height = m_frame.getHeight();
		msg->step = m_frame.getStrideInBytes();

		switch (m_frame.getVideoMode().getPixelFormat()) {

		case PIXEL_FORMAT_DEPTH_1_MM:
			msg->encoding = sensor_msgs::image_encodings::MONO16;
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

	sensor_msgs::CameraInfoPtr getDefaultCameraInfo(int width, int height,
			double f) const {
		sensor_msgs::CameraInfoPtr info = boost::make_shared<
				sensor_msgs::CameraInfo>();

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

private:
	VideoFrameRef m_frame;
	image_transport::CameraPublisher pub;
	camera_info_manager::CameraInfoManager cim;

	unsigned int counter;
	std::string camera_name;
	sensor_msgs::CameraInfoPtr info;
	sensor_msgs::ImagePtr msg;

};

class OpenNI2Camera {
public:

	OpenNI2Camera(ros::NodeHandle & nh) :
			it(nh), dc(nh, it, "depth"), rgbc(nh, it, "rgb") {

		Status rc = OpenNI::initialize();
		if (rc != STATUS_OK) {
			printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
			exit(1);
		}

		rc = device.open(ANY_DEVICE);
		if (rc != STATUS_OK) {
			printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
			exit(2);
		}

		rc = device.setDepthColorSyncEnabled(true);
		if (rc != STATUS_OK) {
			printf(
					"Couldn't enable depth and color images synchronization\n%s\n",
					OpenNI::getExtendedError());
			exit(2);
		}

		if (device.getSensorInfo(SENSOR_DEPTH) != NULL) {
			rc = depth.create(device, SENSOR_DEPTH);
			if (rc != STATUS_OK) {
				printf("Couldn't create depth stream\n%s\n",
						OpenNI::getExtendedError());
			}
		}

		if (device.getSensorInfo(SENSOR_COLOR) != NULL) {
			rc = color.create(device, SENSOR_COLOR);
			if (rc != STATUS_OK) {
				printf("Couldn't create color stream\n%s\n",
						OpenNI::getExtendedError());
			}
		}

		rc = depth.setMirroringEnabled(false);
		if (rc != STATUS_OK) {
			printf("Couldn't disable mirroring for depth stream\n%s\n",
					OpenNI::getExtendedError());
		}

		rc = color.setMirroringEnabled(false);
		if (rc != STATUS_OK) {
			printf("Couldn't disable mirroring for color stream\n%s\n",
					OpenNI::getExtendedError());
		}

		VideoMode depth_video_mode, color_video_mode;

		depth_video_mode.setFps(30);
		depth_video_mode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
		depth_video_mode.setResolution(640, 480);

		color_video_mode.setFps(30);
		color_video_mode.setPixelFormat(PIXEL_FORMAT_RGB888);
		color_video_mode.setResolution(640, 480);

		rc = depth.setVideoMode(depth_video_mode);
		if (rc != STATUS_OK) {
			printf("Couldn't set depth video mode\n%s\n",
					OpenNI::getExtendedError());
		}

		rc = color.setVideoMode(color_video_mode);
		if (rc != STATUS_OK) {
			printf("Couldn't set color video mode\n%s\n",
					OpenNI::getExtendedError());
		}

		rc = depth.start();
		if (rc != STATUS_OK) {
			printf("Couldn't start the depth stream\n%s\n",
					OpenNI::getExtendedError());
		}

		rc = color.start();
		if (rc != STATUS_OK) {
			printf("Couldn't start the color stream\n%s\n",
					OpenNI::getExtendedError());
		}

		// Register to new frame
		depth.addNewFrameListener(&dc);
		color.addNewFrameListener(&rgbc);

	}

	virtual ~OpenNI2Camera() {
		depth.stop();
		color.stop();
		depth.destroy();
		color.destroy();
		device.close();
		OpenNI::shutdown();
	}

private:
	VideoFrameRef m_frame;
	VideoStream depth, color;
	Device device;

	image_transport::ImageTransport it;

	FrameCallback dc;
	FrameCallback rgbc;

};

int main(int argc, char** argv) {

	ros::init(argc, argv, "camera");
	ros::NodeHandle nh;

	OpenNI2Camera pc(nh);

	ros::spin();

}

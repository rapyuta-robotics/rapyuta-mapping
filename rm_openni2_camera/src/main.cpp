#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <OpenNI.h>
#include <cstring>

using namespace openni;

class DepthCallback: public VideoStream::NewFrameListener {
public:

	DepthCallback(image_transport::ImageTransport & it) {
		pub = it.advertise("/camera/depth/image_raw", 1);
	}

	virtual ~DepthCallback() {
	}

	void onNewFrame(VideoStream& stream) {
		stream.readFrame(&m_frame);

		//std::cerr << m_frame.getVideoMode().getPixelFormat() << std::endl;

		sensor_msgs::ImagePtr msg(new sensor_msgs::Image);
		msg->width = m_frame.getWidth();
		msg->height = m_frame.getHeight();
		msg->step = m_frame.getStrideInBytes();
		msg->encoding = sensor_msgs::image_encodings::MONO16;
		msg->data.resize(m_frame.getDataSize());
		memcpy(msg->data.data(), m_frame.getData(), m_frame.getDataSize());

		pub.publish(msg);
	}

private:
	VideoFrameRef m_frame;
	image_transport::Publisher pub;
};

class RGBCallback: public VideoStream::NewFrameListener {
public:

	RGBCallback(image_transport::ImageTransport & it) {
		pub = it.advertise("/camera/rgb/image_color", 1);
	}

	virtual ~RGBCallback() {
	}

	void onNewFrame(VideoStream& stream) {
		stream.readFrame(&m_frame);

		//std::cerr << m_frame.getVideoMode().getPixelFormat() << std::endl;

		sensor_msgs::ImagePtr msg(new sensor_msgs::Image);
		msg->width = m_frame.getWidth();
		msg->height = m_frame.getHeight();
		msg->step = m_frame.getStrideInBytes();
		msg->encoding = sensor_msgs::image_encodings::RGB8;
		msg->data.resize(m_frame.getDataSize());
		memcpy(msg->data.data(), m_frame.getData(), m_frame.getDataSize());

		pub.publish(msg);
	}

private:
	VideoFrameRef m_frame;
	image_transport::Publisher pub;
};

class OpenNI2Camera {
public:

	OpenNI2Camera(ros::NodeHandle & nh) :
			it(nh), dc(it), rgbc(it) {

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

	DepthCallback dc;
	RGBCallback rgbc;

};

int main(int argc, char** argv) {

	ros::init(argc, argv, "openni2_camera");
	ros::NodeHandle nh;

	OpenNI2Camera pc(nh);

	ros::spin();

}

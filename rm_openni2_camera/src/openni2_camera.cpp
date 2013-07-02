/*
 * openni2_camera.cpp
 *
 *  Created on: Apr 24, 2013
 *      Author: vsu
 */

#include <openni2_camera.h>

OpenNI2Camera::OpenNI2Camera(ros::NodeHandle & nh, ros::NodeHandle & nh_private) {

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
		printf("Couldn't enable depth and color images synchronization\n%s\n",
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

        rc = color.getCameraSettings()->setAutoWhiteBalanceEnabled(false);
        if (rc != STATUS_OK) {
                printf("Couldn't disable auto white balance\n%s\n",
                                OpenNI::getExtendedError());
                exit(2);
        }

        rc = color.getCameraSettings()->setAutoExposureEnabled(false);
        if (rc != STATUS_OK) {
                printf("Couldn't disable auto exposure\n%s\n",
                                OpenNI::getExtendedError());
                exit(2);
        }


	VideoMode depth_video_mode, color_video_mode;

	depth_video_mode.setFps(30);
	depth_video_mode.setPixelFormat(PIXEL_FORMAT_DEPTH_1_MM);
	depth_video_mode.setResolution(640, 480);

	color_video_mode.setFps(30);
	color_video_mode.setPixelFormat(PIXEL_FORMAT_YUV422);
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

	rc = device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	if (rc != STATUS_OK) {
		printf("Couldn't enable depth and color images registration\n%s\n",
				OpenNI::getExtendedError());
		exit(2);
	}

        rc = color.getCameraSettings()->setExposure(1000);
        if (rc != STATUS_OK) {
                printf("Couldn't set exposure\n%s\n",
                                OpenNI::getExtendedError());
                exit(2);
        }


	dc.reset(new FrameCallback(nh, nh_private, "depth"));
	rgbc.reset(new FrameCallback(nh, nh_private, "rgb"));

	// Register to new frame
	ROS_INFO("Registering callbacks");
	depth.addNewFrameListener(dc.get());
	color.addNewFrameListener(rgbc.get());
	ROS_INFO("Done registering callbacks");

}

OpenNI2Camera::~OpenNI2Camera() {
	depth.stop();
	color.stop();
	depth.destroy();
	color.destroy();
	device.close();
	OpenNI::shutdown();
}

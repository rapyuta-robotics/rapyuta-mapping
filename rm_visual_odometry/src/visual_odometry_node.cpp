/*****************************************************************************
 *                                                                            *
 *  OpenNI 2.x Alpha                                                          *
 *  Copyright (C) 2012 PrimeSense Ltd.                                        *
 *                                                                            *
 *  This file is part of OpenNI.                                              *
 *                                                                            *
 *  Licensed under the Apache License, Version 2.0 (the "License");           *
 *  you may not use this file except in compliance with the License.          *
 *  You may obtain a copy of the License at                                   *
 *                                                                            *
 *      http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                            *
 *  Unless required by applicable law or agreed to in writing, software       *
 *  distributed under the License is distributed on an "AS IS" BASIS,         *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 *  See the License for the specific language governing permissions and       *
 *  limitations under the License.                                            *
 *                                                                            *
 *****************************************************************************/
#include <stdio.h>
#include <OpenNI.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>

#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms
using namespace openni;

class VisualOdometry {

public:
	VisualOdometry() {

		Status rc = OpenNI::initialize();
		if (rc != STATUS_OK) {
			ROS_ERROR("Initialize failed\n%s\n", OpenNI::getExtendedError());
			exit(1);
		}

		rc = device.open(ANY_DEVICE);
		if (rc != STATUS_OK) {
			ROS_ERROR("Couldn't open device\n%s\n", OpenNI::getExtendedError());
			exit(1);
		}

		if (device.getSensorInfo(SENSOR_DEPTH) != NULL) {
			rc = depth.create(device, SENSOR_DEPTH);
			if (rc == STATUS_OK) {
				rc = depth.start();
				if (rc != STATUS_OK) {
					ROS_ERROR("Couldn't start the color stream\n%s\n",
							OpenNI::getExtendedError());
				}
			} else {
				ROS_ERROR("Couldn't create depth stream\n%s\n",
						OpenNI::getExtendedError());
			}
		}

		if (device.getSensorInfo(SENSOR_COLOR) != NULL) {
			rc = color.create(device, SENSOR_COLOR);
			if (rc == STATUS_OK) {
				rc = color.start();
				if (rc != STATUS_OK) {
					ROS_ERROR("Couldn't start the color stream\n%s\n",
							OpenNI::getExtendedError());
				}
			} else {
				ROS_ERROR("Couldn't create color stream\n%s\n",
						OpenNI::getExtendedError());
			}
		}

		streams[0] = &depth;
		streams[1] = &color;

	}

	~VisualOdometry() {
		depth.stop();
		color.stop();
		depth.destroy();
		color.destroy();
		device.close();
		OpenNI::shutdown();
	}

	void readFrame() {
		int readyStream = -1;
		Status rc = OpenNI::waitForAnyStream(streams, 2, &readyStream,
				SAMPLE_READ_WAIT_TIMEOUT);
		if (rc != STATUS_OK) {
			ROS_ERROR("Wait failed! (timeout is %d ms)\n%s\n",
					SAMPLE_READ_WAIT_TIMEOUT, OpenNI::getExtendedError());
			return;
		}

		switch (readyStream) {
		case 0:
			// Depth
			depth.readFrame(&color_frame);
			break;
		case 1:
			// Color
			color.readFrame(&depth_frame);
			break;
		default:
			ROS_ERROR("Unxpected stream\n");
		}

		if(color_frame.isValid() && depth_frame.isValid()){
			ROS_INFO("Color index: %d Depth index: %d", color_frame.getFrameIndex(), depth_frame.getFrameIndex());
		}

		/*if(color_frame.isValid() && depth_frame.isValid() && color_frame.getFrameIndex() == depth_frame.getFrameIndex()) {
			cv::Mat color_img(color_frame.getHeight(), color_frame.getWidth(), CV_8UC2,(void *) color_frame.getData());
			cv::Mat depth_img(depth_frame.getHeight(), depth_frame.getWidth(), CV_16UC1,(void *) depth_frame.getData());
			cv::Mat gray_img;

			cv::cvtColor(color_img, depth_img, CV_YUV2GRAY_UYVY);
			cv::imshow("Color Image", gray_img);
			cv::imshow("Depth Image", depth_img);
			cv::waitKey(2);
		}*/

	}

private:
	Device device;
	VideoStream depth, color;
	VideoStream * streams[2];
	VideoFrameRef color_frame, depth_frame;

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "camera");

	VisualOdometry vo;

	while (ros::ok()) {
		vo.readFrame();
		ros::spinOnce();
	}

	return 0;
}

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
#include <tbb/concurrent_queue.h>
#include <boost/thread.hpp>
#include <std_msgs/String.h>

#define SAMPLE_READ_WAIT_TIMEOUT 2000 //2000ms
using namespace openni;

class DynamicBRISK: public cv::BRISK {

public:
	DynamicBRISK(int init_threshold, int num_points) :
			BRISK(init_threshold, 0) {
		this->num_points = num_points;
	}

	void operator()(cv::InputArray image, cv::InputArray mask,
			std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
			bool useProvidedKeypoints = false) {
		BRISK::operator()(image, mask, keypoints, descriptors,
				useProvidedKeypoints);
		if (keypoints.size() > num_points) {
			threshold++;
		}

		if (keypoints.size() < num_points) {
			threshold--;
		}
	}

	int getThreshold(){
		return threshold;
	}

private:
	int num_points;

};

class VisualOdometry {

public:
	VisualOdometry(ros::NodeHandle & nh) :
			brisk(30, 100), matcher(cv::NORM_HAMMING, true) {

		pub = nh.advertise<std_msgs::String>("/test", 2);
		msg.reset(new std_msgs::String);

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

		rc = device.setDepthColorSyncEnabled(true);
		if (rc != STATUS_OK) {
			ROS_ERROR("Couldn't set depth color syncronization\n%s\n",
					OpenNI::getExtendedError());
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

		cv::Mat img;
		switch (readyStream) {
		case 0:
			// Depth
			depth.readFrame(&depth_frame);
			//cv::Mat(depth_frame.getHeight(), depth_frame.getWidth(),
			//		CV_16UC1, (void *) depth_frame.getData());

			ROS_INFO("Depth index: %d Time: %ld", depth_frame.getFrameIndex(),
					depth_frame.getTimestamp());
			break;
		case 1:
			// Color
			color.readFrame(&color_frame);
			img = cv::Mat(color_frame.getHeight(), color_frame.getWidth(),
					CV_8UC2, (void *) color_frame.getData());

			processColorFrame(img);
			ROS_INFO("Color index: %d Time: %ld", color_frame.getFrameIndex(),
					color_frame.getTimestamp());
			break;
		default:
			ROS_ERROR("Unxpected stream\n");
		}

	}

	void processColorFrame(const cv::Mat & img) {

		prev_gray_img = gray_img;
		prev_keypoints = keypoints;
		prev_descriptors = descriptors;

		keypoints.reset(new std::vector<cv::KeyPoint>);

		cv::cvtColor(img, gray_img, CV_YUV2GRAY_UYVY);
		cv::GaussianBlur(gray_img, gray_img, cv::Size(3, 3), 0);

		brisk(gray_img, cv::noArray(), *keypoints, descriptors);

		ROS_INFO("Number of keypoints: %d, Threshold %d", keypoints->size(),
				brisk.getThreshold());

		if (prev_keypoints) {

			std::vector<cv::DMatch> matches;
			matcher.match(prev_descriptors, descriptors, matches);

			pub.publish(msg);

			//cv::drawKeypoints(gray_img, keypoints, keypoints_img, cv::Scalar(0, 255, 0),
			//		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			//cv::drawMatches(prev_gray_img, *prev_keypoints, gray_img,
			//		*keypoints, matches, keypoints_img);
			//cv::imshow("Keypoints", keypoints_img);
			//cv::waitKey(2);
		}

	}

private:
	Device device;
	VideoStream depth, color;
	VideoStream * streams[2];
	VideoFrameRef color_frame, depth_frame;

	cv::Mat yuv_img;

	cv::Mat gray_img, prev_gray_img;
	boost::shared_ptr<std::vector<cv::KeyPoint> > keypoints, prev_keypoints;
	cv::Mat descriptors, prev_descriptors;

	cv::Mat keypoints_img;

	DynamicBRISK brisk;
	cv::BFMatcher matcher;

	ros::Publisher pub;
	std_msgs::StringPtr msg;

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "camera");

	ros::NodeHandle nh;

	VisualOdometry vo(nh);

	while (ros::ok()) {
		vo.readFrame();
		cv::waitKey(2);
		ros::spinOnce();
	}

	return 0;
}

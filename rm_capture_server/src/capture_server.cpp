#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <rm_capture_server/Capture.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_listener.h>

class CaptureServer {
protected:

	ros::NodeHandle nh_;

	ros::ServiceServer rgbd_service;

	ros::Subscriber yuv2_sub;
	ros::Subscriber depth_sub;
	ros::Subscriber yuv2_info_sub;
	ros::Subscriber depth_info_sub;

	sensor_msgs::Image::ConstPtr yuv2_msg;
	sensor_msgs::Image::ConstPtr depth_msg;
	sensor_msgs::CameraInfo::ConstPtr yuv2_info_msg;
	sensor_msgs::CameraInfo::ConstPtr depth_info_msg;

	tf::TransformListener listener;

	int queue_size_;

public:

	CaptureServer() {

		ROS_INFO("Creating capture server");

		queue_size_ = 1;

		yuv2_sub = nh_.subscribe<sensor_msgs::Image>("rgb/image_raw",
				queue_size_, &CaptureServer::RGBCallback, this);
		depth_sub = nh_.subscribe<sensor_msgs::Image>("depth/image_raw",
				queue_size_, &CaptureServer::DepthCallback, this);
		yuv2_info_sub = nh_.subscribe<sensor_msgs::CameraInfo>(
				"rgb/camera_info", queue_size_, &CaptureServer::RGBInfoCallback,
				this);
		depth_info_sub = nh_.subscribe<sensor_msgs::CameraInfo>(
				"depth/camera_info", queue_size_,
				&CaptureServer::DepthInfoCallback, this);

		rgbd_service = nh_.advertiseService("capture",
				&CaptureServer::CaptureCallback, this);

	}

	~CaptureServer(void) {
	}

	bool CaptureCallback(rm_capture_server::Capture::Request &req,
			rm_capture_server::Capture::Response &res) {

		cv_bridge::CvImagePtr bgr_image = cv_bridge::toCvCopy(yuv2_msg,
				sensor_msgs::image_encodings::BGR8);
		cv_bridge::CvImageConstPtr depth_image = cv_bridge::toCvShare(
				depth_msg);

		std::vector<int> params;
		params.resize(3, 0);

		params[0] = CV_IMWRITE_PNG_COMPRESSION;
		params[1] = 9;

		res.header = yuv2_msg->header;

		if (!cv::imencode(".png", depth_image->image, res.depth_png_data,
				params)) {
			ROS_ERROR("cv::imencode (png) failed on input image");
			return false;
		}

		if (!cv::imencode(".png", bgr_image->image, res.rgb_png_data, params)) {
			ROS_ERROR("cv::imencode (png) failed on input image");
			return false;
		}

		tf::StampedTransform transform;
		try {
			listener.lookupTransform("/map", yuv2_msg->header.frame_id,
					ros::Time(0), transform);
		} catch (tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
		}

		tf::transformTFToMsg(transform, res.transform);

		return true;
	}

	void RGBCallback(const sensor_msgs::Image::ConstPtr& yuv2_msg) {

		this->yuv2_msg = yuv2_msg;

	}

	void DepthCallback(const sensor_msgs::Image::ConstPtr& depth_msg) {

		this->depth_msg = depth_msg;

	}

	void RGBInfoCallback(
			const sensor_msgs::CameraInfo::ConstPtr& yuv2_info_msg) {

		this->yuv2_info_msg = yuv2_info_msg;

	}

	void DepthInfoCallback(
			const sensor_msgs::CameraInfo::ConstPtr& depth_info_msg) {

		this->depth_info_msg = depth_info_msg;

	}

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "capture_server");

	CaptureServer cs;
	ros::spin();

	return 0;
}

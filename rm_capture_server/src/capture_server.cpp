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

class CaptureServer {
protected:

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
			sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicy;

	typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;

	ros::NodeHandle nh_;

	ros::ServiceServer rgbd_service;

	message_filters::Subscriber<sensor_msgs::Image> rgb_sub;
	message_filters::Subscriber<sensor_msgs::Image> depth_sub;
	message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub;

	sensor_msgs::Image::ConstPtr yuv2_msg;
	sensor_msgs::Image::ConstPtr depth_msg;
	sensor_msgs::CameraInfo::ConstPtr info_msg;

	boost::shared_ptr<Synchronizer> sync;

	int queue_size_;

public:

	CaptureServer() {

		ROS_INFO("Creating capture server");

		queue_size_ = 5;

		rgb_sub.subscribe(nh_, "rgb/image_raw", queue_size_);
		depth_sub.subscribe(nh_, "depth/image_raw", queue_size_);
		info_sub.subscribe(nh_, "rgb/camera_info", queue_size_);

		// Synchronize inputs.
		sync.reset(
				new Synchronizer(SyncPolicy(queue_size_), rgb_sub, depth_sub,
						info_sub));

		sync->registerCallback(
				boost::bind(&CaptureServer::RGBDCallback, this, _1, _2,
						_3));

		rgbd_service = nh_.advertiseService("capture",
				&CaptureServer::CaptureCallback, this);

	}

	~CaptureServer(void) {
	}

	bool CaptureCallback(rm_capture_server::Capture::Request &req,
			rm_capture_server::Capture::Response &res) {

		cv_bridge::CvImagePtr bgr_image = cv_bridge::toCvCopy(yuv2_msg,
				sensor_msgs::image_encodings::BGR8);
		cv_bridge::CvImageConstPtr depth_image = cv_bridge::toCvShare(depth_msg);

		std::vector<int> params;
		params.resize(3, 0);

		params[0] = CV_IMWRITE_PNG_COMPRESSION;
		params[1] = 9;

		res.width = yuv2_msg->width;
		res.height = yuv2_msg->height;
		res.header = yuv2_msg->header;
		res.K = info_msg->K;
		res.D = info_msg->D;

		if (!cv::imencode(".png", depth_image->image, res.depth_png_data,
				params)) {
			ROS_ERROR("cv::imencode (png) failed on input image");
			return false;
		}

		if (!cv::imencode(".png", bgr_image->image, res.rgb_png_data, params)) {
			ROS_ERROR("cv::imencode (png) failed on input image");
			return false;
		}

		return true;
	}

	void RGBDCallback(const sensor_msgs::Image::ConstPtr& yuv2_msg,
			const sensor_msgs::Image::ConstPtr& depth_msg,
			const sensor_msgs::CameraInfo::ConstPtr& info_msg) {

		this->yuv2_msg = yuv2_msg;
		this->depth_msg = depth_msg;
		this->info_msg = info_msg;

	}

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "capture_server");

	CaptureServer cs;
	ros::spin();

	return 0;
}

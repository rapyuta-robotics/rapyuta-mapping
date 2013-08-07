/*
 * test_lkt.cpp
 *
 *  Created on: Jul 27, 2013
 *      Author: vsu
 */




#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <rm_localization/SetMap.h>
#include <rm_localization/SetInitialPose.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>

#include <util.h>

class CaptureServer {
protected:

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
			sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicy;

	typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;

	ros::NodeHandle nh_;
	ros::NodeHandle nh_private;

	ros::ServiceServer set_map_service;
	ros::ServiceServer set_initial_pose;

	message_filters::Subscriber<sensor_msgs::Image> rgb_sub;
	message_filters::Subscriber<sensor_msgs::Image> depth_sub;
	message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub;

	pcl::PointCloud<pcl::PointXYZ> map_keypoints3d;
	cv::Mat map_descriptors;

	boost::shared_ptr<Synchronizer> sync;

	int queue_size_;

	Eigen::Vector4f intrinsics;
	cv::Ptr<cv::FeatureDetector> fd;
	cv::Ptr<cv::DescriptorExtractor> de;
	cv::Ptr<cv::DescriptorMatcher> dm;

	tf::Transform map_to_odom;
	tf::TransformBroadcaster br;
	tf::TransformListener listener;

	std::string tf_prefix_;
	std::string odom_frame;
	std::string map_frame;
	boost::mutex m;

	Eigen::Vector3f offset;

	boost::shared_ptr<std::vector<cv::Vec2f> > tracked_points;

	cv::Mat prev_img, current_img;

public:

	CaptureServer() :
			nh_private("~") {

		ROS_INFO("Creating localization");

		tf_prefix_ = tf::getPrefixParam(nh_private);
		odom_frame = tf::resolve(tf_prefix_, "odom_combined");
		map_frame = tf::resolve(tf_prefix_, "map");
		map_to_odom.setIdentity();
		queue_size_ = 1;

		rgb_sub.subscribe(nh_, "rgb/image_raw", queue_size_);
		depth_sub.subscribe(nh_, "depth/image_raw", queue_size_);
		info_sub.subscribe(nh_, "rgb/camera_info", queue_size_);

		// Synchronize inputs.
		sync.reset(
				new Synchronizer(SyncPolicy(queue_size_), rgb_sub, depth_sub,
						info_sub));

		sync->registerCallback(
				boost::bind(&CaptureServer::RGBDCallback, this, _1, _2, _3));


		tracked_points.reset(new std::vector<cv::Vec2f>);

	}

	~CaptureServer(void) {
	}



	void RGBDCallback(const sensor_msgs::Image::ConstPtr& yuv2_msg,
			const sensor_msgs::Image::ConstPtr& depth_msg,
			const sensor_msgs::CameraInfo::ConstPtr& info_msg) {


		cv_bridge::CvImagePtr gray = cv_bridge::toCvCopy(yuv2_msg,
				sensor_msgs::image_encodings::MONO8);


		if(tracked_points->size() == 0) {
			cv::goodFeaturesToTrack(gray->image, *tracked_points, 400, 0.01, 10);

			prev_img = gray->image;

		} else {


			boost::shared_ptr<std::vector<cv::Vec2f> > current_tracked_points(new std::vector<cv::Vec2f>);
			std::vector<unsigned char> status;
			std::vector<float> error;

			current_img = gray->image;
			cv::calcOpticalFlowPyrLK(prev_img, current_img, *tracked_points, *current_tracked_points, status, error);


			cv::Mat rgb;
			cv::cvtColor(gray->image, rgb, CV_GRAY2BGR);

			for(size_t i=0; i<tracked_points->size(); i++) {
				cv::Point2f p = tracked_points->at(i);
				cv::circle(rgb, cv::Point2i(p.x, p.y), 3, cv::Scalar(255,0,0));
			}


			cv::imshow("Tracked features", rgb);
			cv::waitKey(2);

			prev_img = current_img;
			tracked_points = current_tracked_points;

		}


	}


};

int main(int argc, char** argv) {
	ros::init(argc, argv, "test_localization");

	CaptureServer cs;

	ros::spin();

	return 0;
}

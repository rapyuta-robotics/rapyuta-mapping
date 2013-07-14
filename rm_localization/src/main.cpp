#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <rm_localization/SetMap.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <util.h>

class CaptureServer {
protected:

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
			sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicy;

	typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;

	ros::NodeHandle nh_;
	ros::NodeHandle nh_private;

	ros::ServiceServer set_map_service;

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

public:

	CaptureServer(): nh_private("~") {

		ROS_INFO("Creating capture server");


		tf_prefix_ = tf::getPrefixParam(nh_private);
		odom_frame = tf::resolve(tf_prefix_, "odom_combined");
		map_to_odom.setIdentity();

		de = new cv::SurfDescriptorExtractor;
		dm = new cv::FlannBasedMatcher;
		fd = new cv::SurfFeatureDetector;
		fd->setInt("hessianThreshold", 400);
		fd->setBool("extended", true);
		fd->setBool("upright", true);

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

		set_map_service = nh_.advertiseService("set_map",
				&CaptureServer::SetMapCallback, this);

	}

	~CaptureServer(void) {
	}

	bool SetMapCallback(rm_localization::SetMap::Request &req,
			rm_localization::SetMap::Response &res) {

		cv_bridge::CvImagePtr descriptors = cv_bridge::toCvCopy(
				req.descriptors);
		map_descriptors = descriptors->image;
		pcl::fromROSMsg(req.keypoints3d, map_keypoints3d);

		return true;
	}

	void RGBDCallback(const sensor_msgs::Image::ConstPtr& yuv2_msg,
			const sensor_msgs::Image::ConstPtr& depth_msg,
			const sensor_msgs::CameraInfo::ConstPtr& info_msg) {

		if (map_keypoints3d.size() < 10) {
			return;
		}

		cv_bridge::CvImageConstPtr rgb = cv_bridge::toCvShare(yuv2_msg,
				sensor_msgs::image_encodings::MONO8);
		cv_bridge::CvImageConstPtr depth = cv_bridge::toCvShare(depth_msg);

		intrinsics << 525.0, 525.0, 319.5, 239.5;

		std::vector<cv::KeyPoint> keypoints;

		pcl::PointCloud<pcl::PointXYZ> keypoints3d;
		cv::Mat descriptors;
		compute_features(rgb->image, depth->image, intrinsics, fd, de,
				keypoints, keypoints3d, descriptors);

		std::vector<cv::DMatch> matches;
		dm->match(descriptors, map_descriptors, matches);

		Eigen::Affine3f transform;
		std::vector<bool> inliers;

		bool res = estimate_transform_ransac(keypoints3d, map_keypoints3d,
				matches, 1000, 0.03 * 0.03, 20, transform, inliers);

		if (res) {

			tf::StampedTransform map_to_cam;
			try {
				listener.lookupTransform("/map", yuv2_msg->header.frame_id,
						ros::Time(0), map_to_cam);
			} catch (tf::TransformException ex) {
				ROS_ERROR("%s", ex.what());
			}

			tf::Transform map_to_cam_new;
			tf::transformEigenToTF(transform.cast<double>(), map_to_cam_new);

			map_to_odom = map_to_cam_new * map_to_cam.inverse() * map_to_odom;

		}

	}

	void publishTf() {

		br.sendTransform(
				tf::StampedTransform(map_to_odom, ros::Time::now(), "/map",
						odom_frame));

	}

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "localization");

	CaptureServer cs;

	ros::Rate r(30);
	while (ros::ok()) {
		cs.publishTf();
		r.sleep();
		ros::spinOnce();
	}

	return 0;
}

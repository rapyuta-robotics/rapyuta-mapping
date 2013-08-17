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
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <util.h>
#include <frame.h>
#include <keyframe.h>

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

	keyframe::Ptr k;
	Sophus::SE3f Mwc;

public:

	CaptureServer() :
			nh_private("~"), Mwc(Eigen::Quaternionf::Identity(),
					Eigen::Vector3f::Zero()) {

		ROS_INFO("Creating localization");

		tf_prefix_ = tf::getPrefixParam(nh_private);
		odom_frame = tf::resolve(tf_prefix_, "odom_combined");
		map_frame = tf::resolve(tf_prefix_, "map");
		map_to_odom.setIdentity();

		init_feature_detector(fd, de, dm);

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

		set_initial_pose = nh_.advertiseService("set_initial_pose",
				&CaptureServer::SetInitialPose, this);

	}

	~CaptureServer(void) {
	}

	bool SetInitialPose(rm_localization::SetInitialPose::Request &req,
			rm_localization::SetInitialPose::Response &res) {

	}

	bool SetMapCallback(rm_localization::SetMap::Request &req,
			rm_localization::SetMap::Response &res) {

	}

	void RGBDCallback(const sensor_msgs::Image::ConstPtr& yuv2_msg,
			const sensor_msgs::Image::ConstPtr& depth_msg,
			const sensor_msgs::CameraInfo::ConstPtr& info_msg) {

		cv_bridge::CvImageConstPtr yuv2 = cv_bridge::toCvShare(yuv2_msg);
		cv_bridge::CvImageConstPtr depth = cv_bridge::toCvShare(depth_msg);

		if (k.get()) {

			frame f(yuv2->image, depth->image, Mwc);
			k->estimate_position(f);
			Mwc = f.get_pos();

			tf::Transform cam_to_world;

			Eigen::Affine3d t(Mwc.cast<double>().matrix());
			tf::transformEigenToTF(t, cam_to_world);

			br.sendTransform(
					tf::StampedTransform(cam_to_world, ros::Time::now(),
							"/world",
							"/camera_rgb_optical_frame"));

		} else {
			Eigen::Vector3f intrinsics(525.0, 319.5, 239.5);
			intrinsics /= 2;

			k.reset(new keyframe(yuv2->image, depth->image, Mwc, intrinsics));

		}

		//vis.removeAllPointClouds();
		//vis.addPointCloud<pcl::PointXYZ>(f.get_pointcloud(1));
		//vis.spin();

	}

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "localization");

	CaptureServer cs;

	ros::spin();

	return 0;
}

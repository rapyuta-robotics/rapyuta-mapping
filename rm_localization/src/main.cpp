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

	Eigen::Vector3f intrinsics;
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

	std::vector<keyframe::Ptr> keyframes;
	Sophus::SE3f Mwc;

	ros::Publisher keypoint_pub;

public:

	CaptureServer() :
			nh_private("~"), Mwc(Eigen::Quaternionf::Identity(),
					Eigen::Vector3f::Zero()) {

		ROS_INFO("Creating localization");

		intrinsics << 525.0 / 2, 319.5 / 2, 239.5 / 2;

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

		keypoint_pub = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
				"keypoints", 1);

	}

	~CaptureServer(void) {
	}

	bool SetInitialPose(rm_localization::SetInitialPose::Request &req,
			rm_localization::SetInitialPose::Response &res) {

	}

	bool SetMapCallback(rm_localization::SetMap::Request &req,
			rm_localization::SetMap::Response &res) {

	}

	int get_closest_keyframe() {

		int res = - 1;
		float dist = 10000.0;

		for (size_t i = 0; i < keyframes.size(); i++) {
			Sophus::SE3f t = keyframes[i]->get_pos();

			if ((Mwc.translation() - t.translation()).norm() < 0.5) {

				float current_dist = t.unit_quaternion().angularDistance(
						Mwc.unit_quaternion());

				if (current_dist < dist) {
					res = i;
					dist = current_dist;
				}
			}

		}

		return res;
	}

	void RGBDCallback(const sensor_msgs::Image::ConstPtr& yuv2_msg,
			const sensor_msgs::Image::ConstPtr& depth_msg,
			const sensor_msgs::CameraInfo::ConstPtr& info_msg) {

		cv_bridge::CvImageConstPtr yuv2 = cv_bridge::toCvShare(yuv2_msg);
		cv_bridge::CvImageConstPtr depth = cv_bridge::toCvShare(depth_msg);

		if (keyframes.size() != 0) {

			int closest_keyframe_idx = get_closest_keyframe();

			keyframe::Ptr closest_keyframe = keyframes[closest_keyframe_idx];

			Sophus::SE3f tt = closest_keyframe->get_pos();
			std::cerr << "Closest keyframe " << closest_keyframe_idx
					<< std::endl;

			if ((tt.translation() - Mwc.translation()).norm() > 0.3
					|| tt.unit_quaternion().angularDistance(Mwc.unit_quaternion()) > M_PI/18) {
				keyframe::Ptr k(
						new keyframe(yuv2->image, depth->image, Mwc,
								intrinsics));

				closest_keyframe->estimate_position(*k);
				Mwc = k->get_pos();
				keyframes.push_back(k);
				std::cerr << "Adding new keyframe" << std::endl;

			} else {
				frame f(yuv2->image, depth->image, Mwc);
				closest_keyframe->estimate_position(f);
				Mwc = f.get_pos();
			}

			tf::Transform cam_to_world;

			Eigen::Affine3d t(Mwc.cast<double>().matrix());
			tf::transformEigenToTF(t, cam_to_world);

			br.sendTransform(
					tf::StampedTransform(cam_to_world, ros::Time::now(),
							"/world", "/camera_rgb_optical_frame"));

		} else {

			keyframe::Ptr k(
					new keyframe(yuv2->image, depth->image, Mwc, intrinsics));
			keyframes.push_back(k);
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

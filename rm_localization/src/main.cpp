#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <tf/message_filter.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>

#include <tbb/concurrent_vector.h>

#include <std_srvs/Empty.h>
#include <rm_localization/UpdateMap.h>

#include <frame.h>
#include <keyframe.h>

class CaptureServer {
protected:

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
			sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicy;

	typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;

	ros::NodeHandle nh_;
	ros::NodeHandle nh_private;

	message_filters::Subscriber<sensor_msgs::Image> rgb_sub;
	message_filters::Subscriber<sensor_msgs::Image> depth_sub;
	message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub;

	tf::MessageFilter<sensor_msgs::Image> * rgb_tf_sub;

	boost::shared_ptr<Synchronizer> sync;

	int queue_size_;

	tf::TransformBroadcaster br;
	tf::TransformListener lr;

	Eigen::Vector3f intrinsics;

	tbb::concurrent_vector<keyframe::Ptr> keyframes;
	int closest_keyframe_idx;
	Sophus::SE3f camera_position;
	boost::mutex closest_keyframe_update_mutex;

	nav_msgs::Odometry odom;

	ros::Publisher odom_pub;
	ros::Publisher keyframe_pub;
	ros::ServiceServer update_map_service;
	ros::ServiceServer send_all_keyframes_service;
	ros::ServiceServer clear_keyframes_service;

public:

	CaptureServer() :
			nh_private("~") {

		ROS_INFO("Creating localization");

		double var = 1e-10;
		odom.pose.covariance = { {
				var, 0, 0, 0, 0, 0,
				0, var, 0, 0, 0, 0,
				0, 0, var, 0, 0, 0,
				0, 0, 0, var, 0, 0,
				0, 0, 0, 0, var, 0,
				0, 0, 0, 0, 0, var}};

		queue_size_ = 5;

		odom_pub = nh_.advertise<nav_msgs::Odometry>("vo", queue_size_);
		keyframe_pub = nh_.advertise<rm_localization::Keyframe>("keyframe",
				queue_size_);

		update_map_service = nh_.advertiseService("update_map",
				&CaptureServer::update_map, this);

		send_all_keyframes_service = nh_.advertiseService("send_all_keyframes",
				&CaptureServer::send_all_keyframes, this);

		clear_keyframes_service = nh_.advertiseService("clear_keyframes",
				&CaptureServer::clear_keyframes, this);

		rgb_sub.subscribe(nh_, "rgb/image_raw", queue_size_);
		depth_sub.subscribe(nh_, "depth/image_raw", queue_size_);
		info_sub.subscribe(nh_, "rgb/camera_info", queue_size_);

		rgb_tf_sub = new tf::MessageFilter<sensor_msgs::Image>(rgb_sub, lr,
				"base_footprint", queue_size_);

		// Synchronize inputs.
		sync.reset(
				new Synchronizer(SyncPolicy(queue_size_), *rgb_tf_sub,
						depth_sub, info_sub));

		sync->registerCallback(
				boost::bind(&CaptureServer::RGBDCallback, this, _1, _2, _3));

	}

	~CaptureServer(void) {
		delete rgb_tf_sub;
	}

	// 1.0 when 10 degrees rotation or 1m translation
	inline float distance(const Sophus::SE3f & t1, const Sophus::SE3f t2) {
		float distance = (t1.translation() - t2.translation()).norm();
		float angle = t1.unit_quaternion().angularDistance(
				t2.unit_quaternion());

		return angle / (M_PI / 18) + distance/0.3;

	}

	void get_closest_keyframe(int & res, float & dist) {

		res = -1;
		dist = std::numeric_limits<float>::max();

		for (size_t i = 0; i < keyframes.size(); i++) {
			Sophus::SE3f & t = keyframes[i]->get_pos();

			float current_dist = distance(camera_position, t);

			if (current_dist < dist) {
				res = i;
				dist = current_dist;
			}

		}
	}

	void publish_odom(const std::string & frame, const ros::Time & time) {

		tf::StampedTransform transform;
		try {
			lr.lookupTransform(frame, "base_footprint", time, transform);

			Eigen::Quaterniond q;
			Eigen::Vector3d t;

			tf::quaternionTFToEigen(transform.getRotation(), q);
			tf::vectorTFToEigen(transform.getOrigin(), t);

			Sophus::SE3f Mcb = Sophus::SE3f(q.cast<float>(), t.cast<float>());
			Sophus::SE3f Mob = camera_position * Mcb;

			odom.header.stamp = ros::Time::now();
			odom.header.frame_id = frame;

			tf::quaternionEigenToMsg(Mob.unit_quaternion().cast<double>(),
					odom.pose.pose.orientation);
			tf::pointEigenToMsg(Mob.translation().cast<double>(),
					odom.pose.pose.position);

			odom_pub.publish(odom);

		} catch (tf::TransformException & ex) {
			ROS_ERROR("%s", ex.what());
		}

	}

	void publish_tf(const std::string & frame, const ros::Time & time) {

		tf::StampedTransform transform;
		try {
			lr.lookupTransform(frame, "base_footprint", time, transform);

			Eigen::Quaterniond q;
			Eigen::Vector3d t;

			tf::quaternionTFToEigen(transform.getRotation(), q);
			tf::vectorTFToEigen(transform.getOrigin(), t);

			Sophus::SE3f Mcb = Sophus::SE3f(q.cast<float>(), t.cast<float>());

			Sophus::SE3f Mob = camera_position * Mcb;

			geometry_msgs::TransformStamped tr;

			tf::quaternionEigenToMsg(Mob.unit_quaternion().cast<double>(),
					tr.transform.rotation);
			tf::vectorEigenToMsg(Mob.translation().cast<double>(),
					tr.transform.translation);

			tr.header.frame_id = "odom_combined";
			tr.header.stamp = time;
			tr.child_frame_id = "base_footprint";

			br.sendTransform(tr);

		} catch (tf::TransformException & ex) {
			ROS_ERROR("%s", ex.what());
		}

	}

	bool send_all_keyframes(std_srvs::Empty::Request &req,
			std_srvs::Empty::Response &res) {

		//for (size_t i = 0; i < keyframes.size(); i++) {
		//	keyframe_pub.publish(keyframes[i]->to_msg(yuv2, i));
		//}

		return true;
	}

	bool clear_keyframes(std_srvs::Empty::Request &req,
			std_srvs::Empty::Response &res) {

		boost::mutex::scoped_lock lock(closest_keyframe_update_mutex);

		keyframes.clear();

		return true;
	}

	bool update_map(rm_localization::UpdateMap::Request &req,
			rm_localization::UpdateMap::Response &res) {

		boost::mutex::scoped_lock lock(closest_keyframe_update_mutex);

		Eigen::Vector3f intrinsics;
		intrinsics[0] = req.intrinsics[0];
		intrinsics[1] = req.intrinsics[1];
		intrinsics[2] = req.intrinsics[2];

		bool update_intrinsics = intrinsics[0] != 0.0f;

		if (update_intrinsics) {
			ROS_INFO("Updated camera intrinsics");
			this->intrinsics = intrinsics;
			ROS_INFO_STREAM("New intrinsics " << this->intrinsics.transpose());
		}

		for (size_t i = 0; i < req.idx.size(); i++) {

			Eigen::Quaternionf orientation;
			Eigen::Vector3f position;

			orientation.coeffs()[0] = req.transform[i].unit_quaternion[0];
			orientation.coeffs()[1] = req.transform[i].unit_quaternion[1];
			orientation.coeffs()[2] = req.transform[i].unit_quaternion[2];
			orientation.coeffs()[3] = req.transform[i].unit_quaternion[3];

			position[0] = req.transform[i].position[0];
			position[1] = req.transform[i].position[1];
			position[2] = req.transform[i].position[2];

			Sophus::SE3f new_pos(orientation, position);

			if (req.idx[i] == closest_keyframe_idx) {

				camera_position = new_pos
						* keyframes[req.idx[i]]->get_pos().inverse()
						* camera_position;
			}

			keyframes[req.idx[i]]->get_pos() = new_pos;

			if (update_intrinsics) {
				keyframes[req.idx[i]]->update_intrinsics(intrinsics);
				ROS_INFO_STREAM(
						"New intrinsics " << keyframes[req.idx[i]]->get_intrinsics().transpose());
			}

		}

		return true;
	}

	void init_camera_position(const std::string & frame,
			const ros::Time & time) {

		tf::StampedTransform transform;
		try {
			lr.lookupTransform("base_footprint", frame, time, transform);

			Eigen::Quaterniond q;
			Eigen::Vector3d t;

			tf::quaternionTFToEigen(transform.getRotation(), q);
			tf::vectorTFToEigen(transform.getOrigin(), t);

			camera_position = Sophus::SE3f(q.cast<float>(), t.cast<float>());
		} catch (tf::TransformException & ex) {
			ROS_ERROR("%s", ex.what());
		}

		//ROS_INFO_STREAM(
		//		"Initial camera position" << std::endl << camera_position.matrix());
	}

	void RGBDCallback(const sensor_msgs::Image::ConstPtr& yuv2_msg,
			const sensor_msgs::Image::ConstPtr& depth_msg,
			const sensor_msgs::CameraInfo::ConstPtr& info_msg) {

		cv_bridge::CvImageConstPtr yuv2 = cv_bridge::toCvShare(yuv2_msg);
		cv_bridge::CvImageConstPtr depth = cv_bridge::toCvShare(depth_msg);

		boost::mutex::scoped_lock lock(closest_keyframe_update_mutex);

		//init_camera_position(yuv2_msg->header.frame_id, yuv2_msg->header.stamp);

		if (keyframes.size() != 0) {

			float distance;
			get_closest_keyframe(closest_keyframe_idx, distance);

			keyframe::Ptr closest_keyframe = keyframes[closest_keyframe_idx];

			ROS_INFO("Closest keyframe %d", closest_keyframe_idx);

			if (distance > 1) {
				keyframe::Ptr k(
						new keyframe(yuv2->image, depth->image, camera_position,
								intrinsics));

				closest_keyframe->estimate_position(*k);

				camera_position = k->get_pos();

				keyframe_pub.publish(k->to_msg(yuv2, keyframes.size()));
				keyframes.push_back(k);
				ROS_INFO_STREAM(
						"Added keyframe with intrinsics " << k->get_intrinsics().transpose());
				ROS_INFO_STREAM(
						"Closest keyframe at distance " << distance);

				//publish_odom(yuv2_msg->header.frame_id, yuv2_msg->header.stamp);

			} else {
				frame f(yuv2->image, depth->image, camera_position, intrinsics);
				closest_keyframe->estimate_position(f);

				camera_position = f.get_pos();

				//publish_odom(yuv2_msg->header.frame_id, yuv2_msg->header.stamp);

			}

		} else {

			init_camera_position(yuv2_msg->header.frame_id,
					yuv2_msg->header.stamp);

			intrinsics << info_msg->K[0], info_msg->K[2], info_msg->K[5];

			keyframe::Ptr k(
					new keyframe(yuv2->image, depth->image, camera_position,
							intrinsics));
			keyframe_pub.publish(k->to_msg(yuv2, keyframes.size()));
			keyframes.push_back(k);
			ROS_INFO_STREAM(
					"Added keyframe with intrinsics " << k->get_intrinsics().transpose());
		}

		publish_tf(yuv2_msg->header.frame_id, yuv2_msg->header.stamp);

	}

};

int main(int argc, char** argv) {
	ros::init(argc, argv, "localization");

	CaptureServer cs;

	ros::spin();

	return 0;
}

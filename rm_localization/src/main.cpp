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

	ros::Publisher odom_pub;
	ros::Publisher keyframe_pub;
	ros::ServiceServer update_map_service;
	ros::ServiceServer send_all_keyframes_service;

public:

	CaptureServer() :
			nh_private("~") {

		ROS_INFO("Creating localization");

		queue_size_ = 5;

		odom_pub = nh_.advertise<nav_msgs::Odometry>("vo", queue_size_);
		keyframe_pub = nh_.advertise<rm_localization::Keyframe>("keyframe",
				queue_size_);

		update_map_service = nh_.advertiseService("update_map",
				&CaptureServer::update_map, this);

		send_all_keyframes_service = nh_.advertiseService("send_all_keyframes",
				&CaptureServer::send_all_keyframes, this);

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

	int get_closest_keyframe() {

		int res = -1;
		float dist = 10000.0;

		for (size_t i = 0; i < keyframes.size(); i++) {
			Sophus::SE3f t = keyframes[i]->get_pos();

			if ((camera_position.translation() - t.translation()).norm()
					< 0.5) {

				float current_dist = t.unit_quaternion().angularDistance(
						camera_position.unit_quaternion());

				if (current_dist < dist) {
					res = i;
					dist = current_dist;
				}
			}

		}

		return res;
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

			tr.header.frame_id = "odom";
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

	bool update_map(rm_localization::UpdateMap::Request &req,
			rm_localization::UpdateMap::Response &res) {

		boost::mutex::scoped_lock lock(closest_keyframe_update_mutex);

		Eigen::Vector3f intrinsics;
		memcpy(intrinsics.data(), req.intrinsics.data(), 3 * sizeof(float));

		bool update_intrinsics = intrinsics[0] != 0.0f;

		if(update_intrinsics) {
			ROS_INFO("Updated camera intrinsics");
			this->intrinsics = intrinsics;
			ROS_INFO_STREAM("New intrinsics" << std::endl << this->intrinsics);
		}

		for (size_t i = 0; i < req.idx.size(); i++) {

			Eigen::Quaternionf orientation;
			Eigen::Vector3f position, intrinsics;

			memcpy(orientation.coeffs().data(),
					req.transform[i].unit_quaternion.data(), 4 * sizeof(float));
			memcpy(position.data(), req.transform[i].position.data(),
					3 * sizeof(float));

			Sophus::SE3f new_pos(orientation, position);

			if (req.idx[i] == closest_keyframe_idx) {
				//closest_keyframe_update_mutex.lock();

				camera_position = new_pos
						* keyframes[req.idx[i]]->get_pos().inverse()
						* camera_position;

				keyframes[req.idx[i]]->get_pos() = new_pos;

				//if (update_intrinsics) {
				//	keyframes[req.idx[i]]->get_intrinsics() = intrinsics;
				//}

				//closest_keyframe_update_mutex.unlock();

			} else {
				keyframes[req.idx[i]]->get_pos() = new_pos;

				//if (update_intrinsics) {
				//	keyframes[req.idx[i]]->get_intrinsics() = intrinsics;
				//}

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

		ROS_INFO_STREAM(
				"Initial camera position" << std::endl << camera_position.matrix());
	}

	void RGBDCallback(const sensor_msgs::Image::ConstPtr& yuv2_msg,
			const sensor_msgs::Image::ConstPtr& depth_msg,
			const sensor_msgs::CameraInfo::ConstPtr& info_msg) {

		cv_bridge::CvImageConstPtr yuv2 = cv_bridge::toCvShare(yuv2_msg);
		cv_bridge::CvImageConstPtr depth = cv_bridge::toCvShare(depth_msg);

		boost::mutex::scoped_lock lock(closest_keyframe_update_mutex);

		if (keyframes.size() != 0) {

			closest_keyframe_idx = get_closest_keyframe();
			//ROS_INFO("Closest keyframe %d", closest_keyframe_idx);

			keyframe::Ptr closest_keyframe = keyframes[closest_keyframe_idx];

			Sophus::SE3f tt = closest_keyframe->get_pos();
			ROS_DEBUG("Closest keyframe %d", closest_keyframe_idx);

			if ((tt.translation() - camera_position.translation()).norm() > 0.3
					|| tt.unit_quaternion().angularDistance(
							camera_position.unit_quaternion()) > M_PI / 18) {
				keyframe::Ptr k(
						new keyframe(yuv2->image, depth->image, camera_position,
								intrinsics));

				closest_keyframe->estimate_position(*k);
				camera_position = k->get_pos();

				keyframe_pub.publish(k->to_msg(yuv2, keyframes.size()));
				keyframes.push_back(k);
				ROS_DEBUG("Adding new keyframe");

			} else {
				frame f(yuv2->image, depth->image, camera_position);
				closest_keyframe->estimate_position(f);
				camera_position = f.get_pos();
			}

		} else {

			intrinsics << info_msg->K[0], info_msg->K[2], info_msg->K[5];

			init_camera_position(yuv2_msg->header.frame_id,
					yuv2_msg->header.stamp);

			keyframe::Ptr k(
					new keyframe(yuv2->image, depth->image, camera_position,
							intrinsics));
			keyframe_pub.publish(k->to_msg(yuv2, keyframes.size()));
			keyframes.push_back(k);
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

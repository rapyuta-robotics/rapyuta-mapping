/*
 * g2o_worker.cpp
 *
 *  Created on: Sept 29, 2013
 *      Author: mayanks43
 */

#include <opencv2/core/core.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <util.h>
#include <util_mysql.h>
#include <util_mongo.h>

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <rm_multi_mapper_db/G2oWorkerAction.h>

class G2oWorkerAction {
protected:
	ros::NodeHandle nh_;
	actionlib::SimpleActionServer<rm_multi_mapper_db::G2oWorkerAction> as_;
	std::string action_name_;
	rm_multi_mapper_db::G2oWorkerFeedback feedback_;
	rm_multi_mapper_db::G2oWorkerResult result_;
	util::Ptr U;

public:
#ifdef MONGO
	G2oWorkerAction(std::string name, std::string ip) :
			as_(nh_, name, boost::bind(&G2oWorkerAction::executeCB, this, _1),
					false), action_name_(name), U(new util_mongo(ip)) {
		as_.start();

	}
#else
	G2oWorkerAction(std::string name) :
			as_(nh_, name, boost::bind(&G2oWorkerAction::executeCB, this, _1),
					false), action_name_(name), U(new util_mysql) {
		as_.start();

	}
#endif

	~G2oWorkerAction(void) {
	}

	void executeCB(const rm_multi_mapper_db::G2oWorkerGoalConstPtr & goal) {
		ros::Rate r(1);
		bool success = true;

		if (as_.isPreemptRequested() || !ros::ok()) {
			ROS_INFO("%s: Preempted", action_name_.c_str());
			as_.setPreempted();
			success = false;
		}

		for (size_t i = 0; i < goal->Overlap.size(); i++) {

			pcl::PointCloud<pcl::PointXYZ> keypoints3d1, keypoints3d2;
			cv::Mat descriptors1, descriptors2;

			//std::cerr << "Trying to match " << goal->Overlap[i].first << " "
			//		<< goal->Overlap[i].second << std::endl;

			U->get_keypoints(goal->Overlap[i].first, keypoints3d1, descriptors1);
			U->get_keypoints(goal->Overlap[i].second, keypoints3d2,
					descriptors2);

			Sophus::SE3f t;
			if (U->find_transform(keypoints3d1, keypoints3d2, descriptors1,
					descriptors2, t)) {
				U->add_measurement(goal->Overlap[i].first,
						goal->Overlap[i].second, t, "RANSAC");
			}

		}

		std::cout << "Done";
		if (success) {
			ROS_INFO("%s: Succeeded", action_name_.c_str());
			result_.reply = true;
			as_.setSucceeded(result_);
		}
	}

};

int main(int argc, char** argv) {
	ros::init(argc, argv, argv[1]);
	G2oWorkerAction worker(ros::this_node::getName(), argv[2]);
	ros::spin();

	return 0;
}


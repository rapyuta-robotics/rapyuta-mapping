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
#include <util_mongo.h>

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <rm_multi_mapper_db/G2oWorker2Action.h>

class G2oWorker2Action {
protected:
	ros::NodeHandle nh_;
	actionlib::SimpleActionServer<rm_multi_mapper_db::G2oWorker2Action> as_;
	std::string action_name_;
	rm_multi_mapper_db::G2oWorker2Feedback feedback_;
	rm_multi_mapper_db::G2oWorker2Result result_;
	util::Ptr U;
	int map_id;

public:
	G2oWorker2Action(std::string name, int id) :
			as_(nh_, name, boost::bind(&G2oWorker2Action::executeCB, this, _1),
					false), action_name_(name), U(new util_mongo), map_id(id) {
		as_.start();

	}

	~G2oWorker2Action(void) {
	}

	void executeCB(const rm_multi_mapper_db::G2oWorker2GoalConstPtr & goal) {
		ros::Rate r(1);
		bool success = true;

		if (as_.isPreemptRequested() || !ros::ok()) {
			ROS_INFO("%s: Preempted", action_name_.c_str());
			as_.setPreempted();
			success = false;
		}

		for (size_t i = 0; i < goal->keyframes.size(); i++) {

			std::vector<long> overlapping_keyframes;
			cout<<"worker "<<goal->keyframes[i]<<endl;
			U->get_overlapping_keyframes(goal->keyframes[i], map_id, overlapping_keyframes);
			pcl::PointCloud<pcl::PointXYZ> keypoints3d1, keypoints3d2;
			cv::Mat descriptors1, descriptors2;
			for(int j = 0; j < overlapping_keyframes.size(); j++) {
				//std::cerr << "Trying to match " << goal->Overlap[i].first << " "
				//		<< goal->Overlap[i].second << std::endl;

				U->get_keypoints(goal->keyframes[i], keypoints3d1, descriptors1);
				U->get_keypoints(overlapping_keyframes[j], keypoints3d2,
						descriptors2);

				Sophus::SE3f t;
				if (U->find_transform(keypoints3d1, keypoints3d2, descriptors1,
						descriptors2, t)) {
					U->add_measurement(goal->keyframes[i],
							overlapping_keyframes[j], t, "RANSAC");
				}
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

	ros::init(argc, argv, argv[2]);
	G2oWorker2Action worker(ros::this_node::getName(), atoi(argv[1]));
	ros::spin();

	return 0;
}

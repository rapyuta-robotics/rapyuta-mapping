/*
 * slam.cpp
 *
 *  Created on: Sept 12, 2013
 *      Author: mayanks43
 */
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>
#include <cstdlib>

#include <keyframe_map.h>
#include <util.h>

#include "ros/ros.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Int32.h"
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include "rm_multi_mapper/WorkerSlamAction.h"
#include "rm_multi_mapper/Matrix.h"

/*MySQL includes */
#include "mysql_connection.h"
#include <cppconn/resultset.h>

void get_pairs(std::vector<std::pair<int, int> > & overlapping_keyframes) {
	sql::ResultSet *res;
	util U;
	res =
			U.sql_query(
					"SELECT f1.id as id1, f2.id as id2 FROM positions f1, positions f2 WHERE (abs(f1.q0*f2.q0 + f1.q1*f2.q1 + f1.q2*f2.q2 + f1.q3*f2.q3) >=1.0 OR 2*acos(abs(f1.q0*f2.q0 + f1.q1*f2.q1 + f1.q2*f2.q2 + f1.q3*f2.q3)) < pi()/6) AND f1.id <> f2.id AND SQRT(POWER((f1.t0 - f2.t0), 2) + POWER((f1.t1 - f2.t1), 2) + POWER((f1.t2 - f2.t2), 2)) < 0.5;");

	while (res->next()) {
		overlapping_keyframes.push_back(
				std::make_pair(res->getInt("id1"), res->getInt("id2")));
	}
	delete res;

}

void matrix2eigen(const rm_multi_mapper::Matrix & m1, Eigen::MatrixXf & eigen) {

	for (int i = 0; i < (int) m1.matrix.size(); i++) {
		for (int j = 0; j < (int) m1.matrix[0].vector.size(); j++) {
			eigen(i, j) = m1.matrix[i].vector[j];

		}
	}
}

void vector2eigen(const rm_multi_mapper::Vector & v1, Eigen::VectorXf & eigen) {
	for (int i = 0; i < (int) v1.vector.size(); i++) {
		eigen[i] = v1.vector[i];
	}
}
typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp() {
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_usec + (timestamp_t) now.tv_sec * 1000000;
}
int main(int argc, char **argv) {
	std::vector<color_keyframe::Ptr> frames;

	util U;
	U.load("http://localhost/corridor_map2", frames);
	timestamp_t t0 = get_timestamp();
	std::vector<std::pair<int, int> > overlapping_keyframes;
	int size;
	int workers = argc - 1;
	ros::init(argc, argv, "panorama");

	std::vector<
			actionlib::SimpleActionClient<rm_multi_mapper::WorkerSlamAction>*> ac_list;
	for (int i = 0; i < workers; i++) {
		actionlib::SimpleActionClient<rm_multi_mapper::WorkerSlamAction>* ac =
				new actionlib::SimpleActionClient<
						rm_multi_mapper::WorkerSlamAction>(
						std::string(argv[i + 1]), true);
		ac_list.push_back(ac);
	}

	sql::ResultSet *res;

	size = frames.size();
	get_pairs(overlapping_keyframes);
	std::vector<rm_multi_mapper::WorkerSlamGoal> goals;
	int keyframes_size = (int) overlapping_keyframes.size();

	for (int k = 0; k < workers; k++) {
		rm_multi_mapper::WorkerSlamGoal goal;

		int last_elem = (keyframes_size / workers) * (k + 1);
		if (k == workers - 1)
			last_elem = keyframes_size;

		for (int i = (keyframes_size / workers) * k; i < last_elem; i++) {
			rm_multi_mapper::KeyframePair keyframe;

			keyframe.first = overlapping_keyframes[i].first;
			keyframe.second = overlapping_keyframes[i].second;
			goal.Overlap.push_back(keyframe);
		}
		goals.push_back(goal);
	}

	ROS_INFO("Waiting for action server to start.");
	for (int i = 0; i < workers; i++) {
		ac_list[i]->waitForServer();
	}

	ROS_INFO("Action server started, sending goal.");

	// send a goal to the action
	for (int i = 0; i < workers; i++) {
		ac_list[i]->sendGoal(goals[i]);
	}

	//wait for the action to return
	std::vector<bool> finished;
	for (int i = 0; i < workers; i++) {
		bool finished_before_timeout = ac_list[i]->waitForResult(
				ros::Duration(30.0));
		finished.push_back(finished_before_timeout);
	}

	bool success = true;
	for (int i = 0; i < workers; i++) {
		success = finished[i] && success;
	}

	Eigen::MatrixXf acc_JtJ;
	acc_JtJ.setZero(size * 6, size * 6);
	Eigen::VectorXf acc_Jte;
	acc_Jte.setZero(size * 6);

	if (success) {

		for (int i = 0; i < workers; i++) {
			Eigen::MatrixXf JtJ;
			JtJ.setZero(size * 6, size * 6);
			Eigen::VectorXf Jte;
			Jte.setZero(size * 6);

			rm_multi_mapper::Vector rosJte = ac_list[i]->getResult()->Jte;
			rm_multi_mapper::Matrix rosJtJ = ac_list[i]->getResult()->JtJ;

			vector2eigen(rosJte, Jte);

			matrix2eigen(rosJtJ, JtJ);

			acc_JtJ += JtJ;
			acc_Jte += Jte;

		}

	} else {
		ROS_INFO("Action did not finish before the time out.");
		std::exit(0);
	}

	Eigen::VectorXf update = -acc_JtJ.ldlt().solve(acc_Jte);

	float iteration_max_update = std::max(std::abs(update.maxCoeff()),
			std::abs(update.minCoeff()));

	ROS_INFO("Max update %f", iteration_max_update);

	/*for (int i = 0; i < (int)frames.size(); i++) {

	 frames[i]->get_pos() = Sophus::SE3f::exp(update.segment<6>(i))
	 * frames[i]->get_pos();

	 std::string query = "UPDATE `positions` SET `q0` = " + 
	 boost::lexical_cast<std::string>(frames[i]->get_pos().so3().data()[0]) +
	 ", `q1` = " +
	 boost::lexical_cast<std::string>(frames[i]->get_pos().so3().data()[1]) +
	 ", `q2` = " +
	 boost::lexical_cast<std::string>(frames[i]->get_pos().so3().data()[2]) +
	 ", `q3` = " +
	 boost::lexical_cast<std::string>(frames[i]->get_pos().so3().data()[3]) +
	 ", `t0` = " +
	 boost::lexical_cast<std::string>(frames[i]->get_pos().translation()[0]) +
	 ", `t1` = " +
	 boost::lexical_cast<std::string>(frames[i]->get_pos().translation()[1]) +
	 ", `t2` = " +
	 boost::lexical_cast<std::string>(frames[i]->get_pos().translation()[2]) +
	 ", `int0` = " +
	 boost::lexical_cast<std::string>(frames[i]->get_intrinsics().array()[0]) +
	 ", `int1` = " +
	 boost::lexical_cast<std::string>(frames[i]->get_intrinsics().array()[1]) +
	 ", `int2` = " +
	 boost::lexical_cast<std::string>(frames[i]->get_intrinsics().array()[2]) +
	 " WHERE `id` = " +
	 boost::lexical_cast<std::string>(i) +
	 ";";

	 res = U.sql_query(query);
	 delete res;
	 

	 }*/
	timestamp_t t1 = get_timestamp();

	double secs = (t1 - t0) / 1000000.0L;
	std::cout << secs << std::endl;
	return 0;

}

/*
 * master_g2o.cpp
 *
 *  Created on: Sept 29, 2013
 *      Author: mayanks43
 */
#include <sys/time.h>
#include <fstream>
#include <cmath>
#include <cstdlib>

#include "ros/ros.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Int32.h"
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <rm_multi_mapper_db/G2oWorkerAction.h>

#include <keyframe_map.h>
#include <util.h>

#include <boost/filesystem.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/estimate_propagator.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/edge_se3_offset.h>

#include "mysql_connection.h"
#include <cppconn/resultset.h>
#include <cppconn/statement.h>

typedef unsigned long long timestamp_t;
typedef rm_multi_mapper_db::G2oWorkerAction action_t;
typedef actionlib::SimpleActionClient<action_t> action_client;

int main(int argc, char **argv) {

	boost::shared_ptr<keyframe_map> map;
	std::vector<measurement> m;
	util U;

	//timestamp_t t0 = get_timestamp();

	std::vector<std::pair<long, long> > overlapping_keyframes;
	int workers = argc - 2;

	int map_id = boost::lexical_cast<int>(argv[1]);
	map = U.get_robot_map(map_id);

	ros::init(argc, argv, "multi_map");
	ros::NodeHandle nh;
	ros::Publisher pointcloud_pub = nh.advertise<
			pcl::PointCloud<pcl::PointXYZRGB> >("pointcloud", 1);

	std::vector<action_client*> ac_list;

	for (int i = 0; i < workers; i++) {
		action_client* ac = new action_client(std::string(argv[i + 2]), true);
		ac_list.push_back(ac);
	}

	U.get_overlapping_pairs(map_id, overlapping_keyframes);

	//for (int i = 0; i < overlapping_keyframes.size(); i++) {
	//	std::cerr << "Pair " << overlapping_keyframes[i].first << " "
	//			<< overlapping_keyframes[i].second << std::endl;
	//}

	std::vector<rm_multi_mapper_db::G2oWorkerGoal> goals;
	int keyframes_size = (int) overlapping_keyframes.size();

	for (int k = 0; k < workers; k++) {
		rm_multi_mapper_db::G2oWorkerGoal goal;

		int last_elem = (keyframes_size / workers) * (k + 1);
		if (k == workers - 1)
			last_elem = keyframes_size;

		for (int i = (keyframes_size / workers) * k; i < last_elem; i++) {
			rm_multi_mapper_db::KeyframePair keyframe;

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
				ros::Duration(3600.0));
		finished.push_back(finished_before_timeout);
	}

	bool success = true;
	for (int i = 0; i < workers; i++) {
		success = finished[i] && success;
	}

	if (success) {
		std::cout << success << std::endl;
		//U.load_measurements(m);

		//map->optimize_g2o_min(m);

	}

	return 0;

}

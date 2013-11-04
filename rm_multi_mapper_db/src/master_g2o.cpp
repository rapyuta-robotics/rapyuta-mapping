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
#include "rm_multi_mapper/G2oWorkerAction.h"

#include <keyframe_map.h>
#include <util.h>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

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

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_ros/point_cloud.h>

#include "mysql_connection.h"
#include <cppconn/resultset.h>
#include <cppconn/statement.h>

typedef unsigned long long timestamp_t;
typedef rm_multi_mapper::G2oWorkerAction action_t;
typedef actionlib::SimpleActionClient<action_t> action_client;


void truncate_measurement(util U) {
	sql::ResultSet *res;
	res = U.sql_query("TRUNCATE TABLE measurement");
	delete res;
}

int main(int argc, char **argv) {

	boost::shared_ptr<keyframe_map> map;
	std::vector<measurement> m;
	util U;

	//timestamp_t t0 = get_timestamp();

	std::vector<std::pair<long, long> > overlapping_keyframes;
	int size;
	int workers = argc - 1;

	map = U.get_robot_map(0);

	ros::init(argc, argv, "multi_map");
	ros::NodeHandle nh;
	ros::Publisher pointcloud_pub = nh.advertise<
			pcl::PointCloud<pcl::PointXYZRGB> >("pointcloud", 1);

	std::vector<action_client*> ac_list;

	for (int i = 0; i < workers; i++) {
		action_client* ac = new action_client(std::string(argv[i + 1]), true);
		ac_list.push_back(ac);
	}

	size = map->frames.size();
	U.get_overlapping_pairs(overlapping_keyframes);
	truncate_measurement(U);
	std::vector<rm_multi_mapper::G2oWorkerGoal> goals;
	int keyframes_size = (int) overlapping_keyframes.size();

	for (int k = 0; k < workers; k++) {
		rm_multi_mapper::G2oWorkerGoal goal;

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
				ros::Duration(3600.0));
		finished.push_back(finished_before_timeout);
	}

	bool success = true;
	for (int i = 0; i < workers; i++) {
		success = finished[i] && success;
	}

	if (success) {
		std::cout << success << std::endl;
		U.load_measurements(m);

		//map->optimize_g2o_min(m);

	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map->get_map_pointcloud();

	cloud->header.frame_id = "world";
	cloud->header.stamp = ros::Time::now();
	cloud->header.seq = 0;
	pointcloud_pub.publish(cloud);

	/*timestamp_t t1 = get_timestamp();

	 double secs = (t1 - t0) / 1000000.0L;
	 std::cout<<secs<<std::endl;*/
	return 0;

}

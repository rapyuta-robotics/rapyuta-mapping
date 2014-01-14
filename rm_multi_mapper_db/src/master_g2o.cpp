
#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <rm_multi_mapper_db/G2oWorkerAction.h>

#include <util.h>
#include <util_mysql.h>
#include <util_mongo.h>

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


typedef unsigned long long timestamp_t;
typedef rm_multi_mapper_db::G2oWorkerAction action_t;
typedef actionlib::SimpleActionClient<action_t> action_client;

void optimize_g2o(std::vector<util::position> & p, util::Ptr & U) {

	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(true);
	g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

	linearSolver = new g2o::LinearSolverCholmod<
			g2o::BlockSolver_6_3::PoseMatrixType>();

	g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver =
			new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	std::map<long, int> idx_to_pos;

	for (size_t i = 0; i < p.size(); i++) {
		idx_to_pos[p[i].idx] = i;

		g2o::SE3Quat pose(p[i].transform.unit_quaternion().cast<double>(),
				p[i].transform.translation().cast<double>());
		g2o::VertexSE3 * v_se3 = new g2o::VertexSE3();

		v_se3->setId(i);
		if (i < 1) {
			v_se3->setFixed(true);
		}
		v_se3->setEstimate(pose);
		optimizer.addVertex(v_se3);
	}

	for (size_t i = 0; i < p.size(); i++) {

		std::vector<util::measurement> m;
		U->load_measurements(p[i].idx, m);

		for (size_t it = 0; it < m.size(); it++) {
			int i = idx_to_pos[m[it].first];
			int j = idx_to_pos[m[it].second];

			Sophus::SE3f Mij = m[it].transform;

			g2o::EdgeSE3 * e = new g2o::EdgeSE3();

			e->setVertex(0,
					dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(
							i)->second));
			e->setVertex(1,
					dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(
							j)->second));
			e->setMeasurement(Eigen::Isometry3d(Mij.cast<double>().matrix()));
			e->information() = Sophus::Matrix6d::Identity();

			optimizer.addEdge(e);

		}
	}

	optimizer.save("debug.txt");

	optimizer.initializeOptimization();
	optimizer.setVerbose(true);

	std::cout << std::endl;
	std::cout << "Performing full BA:" << std::endl;
	optimizer.optimize(20);
	std::cout << std::endl;

	for (size_t i = 0; i < p.size(); i++) {
		g2o::HyperGraph::VertexIDMap::iterator v_it = optimizer.vertices().find(
				i);
		if (v_it == optimizer.vertices().end()) {
			std::cerr << "Vertex " << i << " not in graph!" << std::endl;
			exit(-1);
		}

		g2o::VertexSE3 * v_se3 = dynamic_cast<g2o::VertexSE3 *>(v_it->second);
		if (v_se3 == 0) {
			std::cerr << "Vertex " << i << "is not a VertexSE3Expmap!"
					<< std::endl;
			exit(-1);
		}

		double est[7];
		v_se3->getEstimateData(est);

		Eigen::Vector3d v(est);
		Eigen::Quaterniond q(est + 3);

		Sophus::SE3f t(q.cast<float>(), v.cast<float>());

		p[i].transform = t;
	}
}

int main(int argc, char **argv) {

	boost::shared_ptr<keyframe_map> map;
	util::Ptr U(new util_mongo);

	//timestamp_t t0 = get_timestamp();

	std::vector<std::pair<long, long> > overlapping_keyframes;
	int workers = argc - 2;

	int map_id = boost::lexical_cast<int>(argv[1]);
	map = U->get_robot_map(map_id);

	ros::init(argc, argv, "multi_map");
	ros::NodeHandle nh;
	ros::Publisher pointcloud_pub = nh.advertise<
			pcl::PointCloud<pcl::PointXYZRGB> >("pointcloud", 1);

	std::vector<action_client*> ac_list;

	for (int i = 0; i < workers; i++) {
		action_client* ac = new action_client(std::string(argv[i + 2]), true);
		ac_list.push_back(ac);
	}

	U->get_overlapping_pairs(map_id, overlapping_keyframes);

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

		std::vector<util::position> p;
		U->load_positions(map_id, p);
		optimize_g2o(p, U);

		for(size_t i=0; i<p.size(); i++ ){
			U->update_position(p[i]);
		}

	}

	return 0;

}

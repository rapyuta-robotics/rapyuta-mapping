#include <robot_mapper.h>
#include <tf_conversions/tf_eigen.h>

robot_mapper::robot_mapper(ros::NodeHandle & nh,
		const std::string & robot_prefix, const int robot_num) :
		robot_num(robot_num), prefix(
				"/" + robot_prefix
						+ boost::lexical_cast<std::string>(robot_num)), merged(
				false), action_client(prefix + "/turtlebot_move", true), map(
				new keyframe_map) {

	skip_first_n_in_optimization = 1;

	world_to_odom.setIdentity();
	world_to_odom.setOrigin(tf::Vector3(0, robot_num * 10.0, 0));

	pointcloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(
			prefix + "/pointcloud", 1);

	servo_pub = nh.advertise<std_msgs::Float32>(
			prefix + "/mobile_base/commands/servo_angle", 3);

	keyframe_sub = nh.subscribe(prefix + "/keyframe", 10,
			&robot_mapper::keyframeCallback, this);

	update_map_service = nh.serviceClient<rm_localization::UpdateMap>(
			prefix + "/update_map");

	clear_keyframes_service = nh.serviceClient<std_srvs::Empty>(
			prefix + "/clear_keyframes");

	set_camera_info_service = nh.serviceClient<sensor_msgs::SetCameraInfo>(
			prefix + "/rgb/set_camera_info");

	std_srvs::Empty emp;
	clear_keyframes_service.call(emp);

	publish_empty_cloud();

	boost::thread t(&robot_mapper::publish_tf, this);

}

void robot_mapper::keyframeCallback(
		const rm_localization::Keyframe::ConstPtr& msg) {

	boost::mutex::scoped_lock lock(merge_mutex);

	ROS_INFO("Received keyframe");
	map->add_frame(msg);

	if (!merged) {
		publish_cloud();
	}

}

void robot_mapper::publish_empty_cloud() {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud->push_back(pcl::PointXYZRGB(0, 0, 0));
	cloud->header.frame_id = prefix + "/odom_combined";
	cloud->header.stamp = ros::Time::now();
	cloud->header.seq = 0;
	pointcloud_pub.publish(cloud);
}

void robot_mapper::publish_cloud() {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = map->get_map_pointcloud();
	cloud->header.frame_id = prefix + "/odom_combined";
	cloud->header.stamp = ros::Time::now();
	cloud->header.seq = 0;
	pointcloud_pub.publish(cloud);
}

void robot_mapper::move_straight(float distance) {

	ROS_INFO_STREAM("Received move command for " << prefix);

	turtlebot_actions::TurtlebotMoveGoal goal;
	goal.forward_distance = distance;
	goal.turn_distance = 0;

	action_client.waitForServer();
	action_client.sendGoal(goal);

	bool finished_before_timeout = action_client.waitForResult(
			ros::Duration(500.0));

	if (finished_before_timeout) {
		actionlib::SimpleClientGoalState state = action_client.getState();
		ROS_INFO("Action finished: %s", state.toString().c_str());
	} else
		ROS_INFO("Action did not finish before the time out.");

}

void robot_mapper::full_rotation() {

	ROS_INFO_STREAM("Received turn command for " << prefix);
	turn(M_PI);
	turn(M_PI);

}

void robot_mapper::turn_to_initial_heading() {
	tf::StampedTransform transform;
	try {
		ros::Time t = ros::Time::now();
		lr.waitForTransform(prefix + "/base_footprint", prefix + "/odom_combined", t, ros::Duration(5));
		lr.lookupTransform(prefix + "/base_footprint", prefix + "/odom_combined", t, transform);

		tf::Quaternion q;
		q.setEuler(0,0,0);
		float angle = transform.getRotation().angle(q);
		turn(angle);

	} catch (tf::TransformException & ex) {
		ROS_ERROR("%s", ex.what());
	}

}

void robot_mapper::start_optimization_loop() {
	run_optimization = true;
	optimization_loop_thread.reset(
			new boost::thread(&robot_mapper::optimization_loop, this));
}
void robot_mapper::stop_optimization_loop() {
	robot_mapper::run_optimization = false;
	optimization_loop_thread->join();
}

void robot_mapper::optimization_loop() {
	while (run_optimization) {
		optmize();
	}
}

void robot_mapper::publish_tf() {

	tf::TransformBroadcaster br;

	while (true) {
		br.sendTransform(
				tf::StampedTransform(world_to_odom, ros::Time::now(), "/world",
						prefix + "/odom_combined"));
		usleep(33000);
	}

}

void robot_mapper::turn(float angle) {

	ROS_INFO("Turn angle %f", angle);
	//assert(angle >= -M_PI && angle <= M_PI);

	turtlebot_actions::TurtlebotMoveGoal goal;
	goal.forward_distance = 0;
	goal.turn_distance = angle;

	action_client.waitForServer();
	action_client.sendGoal(goal);

	//wait for the action to return
	bool finished_before_timeout = action_client.waitForResult(
			ros::Duration(30.0));

	if (finished_before_timeout) {
		actionlib::SimpleClientGoalState state = action_client.getState();
		ROS_INFO("Action finished: %s", state.toString().c_str());
	} else
		ROS_INFO("Action did not finish before the time out.");
}

void robot_mapper::capture_sphere() {

	std_msgs::Float32 angle_msg;
	angle_msg.data = 1 * M_PI / 18;
	float stop_angle = 0.9 * M_PI / 18;
	float delta = M_PI / 36;

	sleep(3);

	for (; angle_msg.data > stop_angle; angle_msg.data -= delta) {

		servo_pub.publish(angle_msg);
		sleep(1);

		full_rotation();

	}

	//angle_msg.data = 0;
	//servo_pub.publish(angle_msg);

}

void robot_mapper::optmize_panorama() {

	for (int level = 2; level >= 0; level--) {
		for (int i = 0; i < (level + 1) * (level + 1) * 10; i++) {
			float max_update = map->optimize_panorama(level);

			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
					map->get_map_pointcloud();

			update_map();

			publish_cloud();

			//usleep(100000);

			if (max_update < 1e-4)
				break;
		}
	}

	map->align_z_axis();
	update_map(true);
	publish_cloud();

	skip_first_n_in_optimization = map->frames.size();

}

void robot_mapper::optmize() {

	if (map->frames.size() < 2)
		return;

	map->optimize_slam(skip_first_n_in_optimization);
	publish_cloud();
	update_map();

}

void robot_mapper::save_map(const std::string & dirname) {
	map->save(dirname + boost::lexical_cast<std::string>(robot_num));
}

void robot_mapper::update_map(bool with_intrinsics) {

	rm_localization::UpdateMap update_map_msg;
	update_map_msg.request.idx.resize(map->frames.size());
	update_map_msg.request.transform.resize(map->frames.size());

	if (with_intrinsics) {

		Eigen::Vector3f intrinsics = map->frames[0]->get_intrinsics();

		/*
		 sensor_msgs::SetCameraInfo s;
		 s.request.camera_info.width = map->frames[0]->get_i(0).cols;
		 s.request.camera_info.height = map->frames[0]->get_i(0).rows;

		 // No distortion
		 s.request.camera_info.D.resize(5, 0.0);
		 s.request.camera_info.distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;

		 // Simple camera matrix: square pixels (fx = fy), principal point at center
		 s.request.camera_info.K.assign(0.0);
		 s.request.camera_info.K[0] = s.request.camera_info.K[4] = intrinsics[0];
		 s.request.camera_info.K[2] = intrinsics[1];
		 s.request.camera_info.K[5] = intrinsics[2];
		 s.request.camera_info.K[8] = 1.0;

		 // No separate rectified image plane, so R = I
		 s.request.camera_info.R.assign(0.0);
		 s.request.camera_info.R[0] = s.request.camera_info.R[4] = s.request.camera_info.R[8] = 1.0;

		 // Then P=K(I|0) = (K|0)
		 s.request.camera_info.P.assign(0.0);
		 s.request.camera_info.P[0] = s.request.camera_info.P[5] = s.request.camera_info.K[0]; // fx, fy
		 s.request.camera_info.P[2] = s.request.camera_info.K[2]; // cx
		 s.request.camera_info.P[6] = s.request.camera_info.K[5]; // cy
		 s.request.camera_info.P[10] = 1.0;

		 set_camera_info_service.call(s);
		 */

		update_map_msg.request.intrinsics[0] = intrinsics[0];
		update_map_msg.request.intrinsics[1] = intrinsics[1];
		update_map_msg.request.intrinsics[2] = intrinsics[2];

	} else {
		update_map_msg.request.intrinsics = { {0,0,0}};
	}

	for (size_t i = 0; i < map->frames.size(); i++) {
		update_map_msg.request.idx[i] = map->idx[i];

		Sophus::SE3f position = map->frames[i]->get_pos();


		update_map_msg.request.transform[i].unit_quaternion[0] = position.unit_quaternion().coeffs()[0];
		update_map_msg.request.transform[i].unit_quaternion[1] = position.unit_quaternion().coeffs()[1];
		update_map_msg.request.transform[i].unit_quaternion[2] = position.unit_quaternion().coeffs()[2];
		update_map_msg.request.transform[i].unit_quaternion[3] = position.unit_quaternion().coeffs()[3];

		update_map_msg.request.transform[i].position[0] = position.translation()[0];
		update_map_msg.request.transform[i].position[1] = position.translation()[1];
		update_map_msg.request.transform[i].position[2] = position.translation()[2];


	}

	update_map_service.call(update_map_msg);

}

bool robot_mapper::merge(robot_mapper & other) {

	Sophus::SE3f transform;
	if (map->find_transform(*other.map, transform)) {
		boost::mutex::scoped_lock lock(merge_mutex);
		boost::mutex::scoped_lock lock1(other.merge_mutex);

		ROS_INFO_STREAM("Transform" << std::endl << transform.matrix());

		map->merge(*other.map, transform);
		other.map = map;
		other.merged = true;

		Eigen::Affine3d t(transform.cast<double>().matrix());
		tf::transformEigenToTF(t, other.world_to_odom);

		other.publish_empty_cloud();
		publish_cloud();

		return true;
	} else {
		return false;
	}

}


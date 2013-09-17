
#include <util.h>
#include <iostream>
#include <keyframe_map.h>

int main(int argc, char **argv) {

	util U;
	int robot_id = U.get_new_robot_id();
	std::cerr << "New robot id " << robot_id << std::endl;

	keyframe_map map;
	map.load("d_floor_circle_optimized");

	long shift = robot_id*(1l << 32);

	for(int i=0; i < map.frames.size(); i++) {
		map.frames[i]->set_id(shift + i);
		U.add_keyframe(robot_id, map.frames[i]);
	}

	std::cerr << "Uploaded map to database" << std::endl;

	keyframe_map map1;
	for(int i=0; i < map.frames.size(); i++) {
		map1.frames.push_back(U.get_keyframe(map.frames[i]->get_id()));
	}
	map1.save("from_database");

	return 0;
}

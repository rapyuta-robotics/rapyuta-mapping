#include <util.h>
#include <iostream>
#include <keyframe_map.h>

int main(int argc, char **argv) {

	util U;
	int robot_id = U.get_new_robot_id();
	std::cerr << "New robot id " << robot_id << std::endl;

	keyframe_map map;
	map.load(argv[1]);

	long shift = robot_id * (1l << 32);

	for (int i = 0; i < map.frames.size(); i++) {
		map.frames[i]->set_id(shift + i);
		U.add_keyframe(robot_id, map.frames[i]);
	}

	return 0;
}

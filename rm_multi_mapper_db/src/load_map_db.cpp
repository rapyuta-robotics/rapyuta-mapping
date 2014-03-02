#include <util.h>
#include <util_mongo.h>
#include <util_mysql.h>
#include <iostream>
#include <keyframe_map.h>

int main(int argc, char **argv) {

#ifdef MONGO
	util::Ptr U(new util_mongo(argv[2]));
#else
	util::Ptr U(new util_mysql);
#endif
	int robot_id = U->get_new_robot_id();
	std::cerr << "New robot id " << robot_id << std::endl;

	keyframe_map map;
	map.load(argv[1]);
	cout<<"robot_id"<<robot_id<<endl;

	long shift = robot_id * (1l << 32);
	std::cout<<"shift is"<<shift<<std::endl;

	for (size_t i = 0; i < map.frames.size(); i++) {
		map.frames[i]->set_id(shift + i);
		U->add_keyframe(robot_id, map.frames[i]);
		U->add_keypoints(map.frames[i]);
		if (i != 0) {
			//std::cerr << map.frames[i - 1]->get_id() << " "
			//		<< map.frames[i]->get_id() << std::endl;
			U->add_measurement(map.frames[i - 1]->get_id(),
					map.frames[i]->get_id(),
					map.frames[i - 1]->get_pos().inverse()
							* map.frames[i]->get_pos(), "VO");
		}
	}

	return 0;
}

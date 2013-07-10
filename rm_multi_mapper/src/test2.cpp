/*
 * test.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <util.h>

int main() {

	keypoint_map map1("map1");
	keypoint_map map2("map2");

	map1.save("map11");
	map2.save("map22");

	map1.merge_keypoint_map(map2);

	map1.save("map_merged1");

	return 0;
}

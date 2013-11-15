/*
 * test.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <octomap/OcTree.h>

int main(int argc, char **argv) {

	octomap::OcTree tree(0.05);

	for (float x = -0.3; x <= 0.3; x += 0.01) {
		for (float y = -0.3; y <= 0.3; y += 0.01) {
			if (x * x + y * y <= 0.3 * 0.3) {
				tree.updateNode(x, y, 0.01, false);
			}
		}
	}

	tree.writeBinary("free_space.bt");

	return 0;
}

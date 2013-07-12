/*
 * test.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: vsu
 */

#include <util.h>
#include <pcl/visualization/pcl_visualizer.h>

int main() {

	keypoint_map map("map_2");

	/*
	map.remove_bad_points(3);

	pcl::visualization::PCLVisualizer vis;
	vis.removeAllPointClouds();
	vis.addPointCloud<pcl::PointXYZ>(map.keypoints3d.makeShared(), "keypoints");
	vis.spin();

	std::cerr << "Error " << map.compute_error() << " Mean error "
			<< map.compute_error() / map.observations.size() << std::endl;

	for (int i = 0; i < 1; i++) {
		map.optimize();

		std::cerr << "Error " << map.compute_error() << " Mean error "
				<< map.compute_error() / map.observations.size() << std::endl;

		vis.removeAllPointClouds();
		vis.addPointCloud<pcl::PointXYZ>(map.keypoints3d.makeShared(),
				"keypoints");
		vis.spin();
	}

	map.extract_surface();
	*/

	nav_msgs::OccupancyGrid grid;
	octomap::OcTree tree(0.05);
	map.get_octree(tree);
	map.compute_2d_map(tree, grid);

	return 0;
}

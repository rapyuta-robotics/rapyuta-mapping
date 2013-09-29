/*
 * test_diem.cpp
 *
 *  Created on: Jul 29, 2013
 *      Author: vsu
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>
#include <pcl/visualization/pcl_visualizer.h>

#include <keyframe_map.h>
#include <util.h>

int main(int argc, char **argv) {

	util U;
	U.add2DB((std::string)argv[1], 0);
	/*std::vector<std::pair<Sophus::SE3f, Eigen::Vector3f> > positions;

	std::ifstream f(((std::string)argv[1] + "/positions.txt").c_str(),
			std::ios_base::binary);
    int i=0;
	while (f) {
		Eigen::Quaternionf q;
		Eigen::Vector3f t;
		Eigen::Vector3f intrinsics;
                
		f.read((char *) q.coeffs().data(), sizeof(float) * 4);
		f.read((char *) t.data(), sizeof(float) * 3);
		f.read((char *) intrinsics.data(), sizeof(float) * 3);

        std::cout<<i<<","<<q.coeffs()[0]<<","<<q.coeffs()[1]<<",";
        std::cout<<q.coeffs()[2]<<","<<q.coeffs()[3];
        std::cout<<","<<t[0]<<","<<t[1]<<","<<t[2]<<","<<intrinsics[0];
        std::cout<<","<<intrinsics[1]<<","<<intrinsics[2]<<std::endl;
        i++;
	}*/

}



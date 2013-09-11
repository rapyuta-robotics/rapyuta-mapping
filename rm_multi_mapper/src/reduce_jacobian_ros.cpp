#include <reduce_jacobian_ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

reduce_jacobian_ros::reduce_jacobian_ros() : size(0), subsample_level(0) {

	JtJ.setZero(size * 3 + 3, size * 3 + 3);
	Jte.setZero(size * 3 + 3);

}


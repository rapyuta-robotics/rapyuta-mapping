/*
 * reduce_jacobian_icp.h
 *
 *  Created on: Aug 10, 2013
 *      Author: vsu
 */

#ifndef REDUCE_JACOBIAN_ICP_H_
#define REDUCE_JACOBIAN_ICP_H_

#include <keyframe.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_reduce.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>

struct reduce_jacobian_icp {

	Eigen::MatrixXf JtJ;
	Eigen::VectorXf Jte;
	int size;

	tbb::concurrent_vector<keyframe::Ptr> & frames;

	pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> ce;
	pcl::registration::CorrespondenceRejectorOneToOne cr;

	reduce_jacobian_icp(tbb::concurrent_vector<keyframe::Ptr> & frames,
			int size);

	reduce_jacobian_icp(reduce_jacobian_icp& rb, tbb::split);

	void operator()(
			const tbb::blocked_range<
					tbb::concurrent_vector<std::pair<int, int> >::iterator>& r);

	void join(reduce_jacobian_icp& rb);

};



#endif /* REDUCE_JACOBIAN_ICP_H_ */

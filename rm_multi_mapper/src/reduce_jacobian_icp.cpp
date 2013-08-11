
#include <reduce_jacobian_icp.h>
#include <opencv2/imgproc/imgproc.hpp>

reduce_jacobian_icp::reduce_jacobian_icp(
		tbb::concurrent_vector<keyframe::Ptr> & frames, int size) :
		size(size), frames(frames) {

	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);

}

reduce_jacobian_icp::reduce_jacobian_icp(reduce_jacobian_icp& rb, tbb::split) :
		size(rb.size), frames(rb.frames) {
	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);
}

void reduce_jacobian_icp::operator()(
		const tbb::blocked_range<
				tbb::concurrent_vector<std::pair<int, int> >::iterator>& r) {
	for (tbb::concurrent_vector<std::pair<int, int> >::iterator it = r.begin();
			it != r.end(); it++) {
		int i = it->first;
		int j = it->second;

		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_i, cloud_j;

		cloud_i = frames[i]->get_pointcloud();
		cloud_j = frames[j]->get_pointcloud();

		pcl::Correspondences cor;
		ce.setInputCloud(cloud_j);
		ce.setInputTarget(cloud_i);
		ce.determineCorrespondences(cor, 0.01);

		cr.getRemainingCorrespondences(cor, cor);

		for (size_t k = 0; k < cor.size(); k++) {
			if (cor[k].index_match >= 0) {
				pcl::PointXYZ & pi = cloud_i->at(cor[k].index_match);
				pcl::PointXYZ & pj = cloud_j->at(cor[k].index_query);

				Eigen::Vector3f error = pi.getVector3fMap()
						- pj.getVector3fMap();
				Eigen::Matrix<float, 3, 6> Ji, Jj;
				Ji.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();
				Jj.block<3, 3>(0, 0) = -Eigen::Matrix3f::Identity();

				Ji.block<3, 3>(0, 3) << 0, pi.z, -pi.y, -pi.z, 0, pi.x, pi.y, -pi.x, 0;
				Jj.block<3, 3>(0, 3) << 0, -pj.z, pj.y, pj.z, 0, -pj.x, -pj.y, pj.x, 0;

				JtJ.block<6, 6>(i * 6, i * 6) += Ji.transpose() * Ji;
				JtJ.block<6, 6>(j * 6, j * 6) += Jj.transpose() * Jj;
				JtJ.block<6, 6>(i * 6, j * 6) += Ji.transpose() * Jj;
				JtJ.block<6, 6>(j * 6, i * 6) += Jj.transpose() * Ji;

				Jte.segment<6>(i * 6) += Ji.transpose() * error;
				Jte.segment<6>(j * 6) += Jj.transpose() * error;

			}

		}
	}
}

void reduce_jacobian_icp::join(reduce_jacobian_icp& rb) {
	JtJ += rb.JtJ;
	Jte += rb.Jte;
}

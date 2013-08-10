#include <reduce_jacobian_icp_p2p.h>
#include <opencv2/imgproc/imgproc.hpp>

reduce_jacobian_icp_p2p::reduce_jacobian_icp_p2p(
		tbb::concurrent_vector<keyframe::Ptr> & frames, int size) :
		size(size), frames(frames) {

	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);

}

reduce_jacobian_icp_p2p::reduce_jacobian_icp_p2p(reduce_jacobian_icp_p2p& rb,
		tbb::split) :
		size(rb.size), frames(rb.frames) {
	JtJ.setZero(size * 6, size * 6);
	Jte.setZero(size * 6);
}

void reduce_jacobian_icp_p2p::operator()(
		const tbb::blocked_range<
				tbb::concurrent_vector<std::pair<int, int> >::iterator>& r) {
	for (tbb::concurrent_vector<std::pair<int, int> >::iterator it = r.begin();
			it != r.end(); it++) {
		int i = it->first;
		int j = it->second;

		pcl::PointCloud<pcl::PointNormal>::Ptr cloud_i_with_normals;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_i, cloud_j;

		cloud_i_with_normals =
				frames[i]->get_original_pointcloud_with_normals();
		cloud_i = frames[i]->get_original_pointcloud();
		cloud_j = frames[j]->get_transformed_pointcloud(
				frames[i]->get_position().inverse());

		Eigen::Matrix4f Miw = frames[i]->get_position().inverse().matrix();
		Eigen::Matrix4f Mwj = frames[j]->get_position().matrix();

		pcl::Correspondences cor;
		ce.setInputCloud(cloud_j);
		ce.setInputTarget(cloud_i);
		ce.determineCorrespondences(cor, 0.1);

		croto.getRemainingCorrespondences(cor, cor);

		for (size_t k = 0; k < cor.size(); k++) {
			if (cor[k].index_match >= 0) {
				pcl::PointNormal & pi = cloud_i_with_normals->at(
						cor[k].index_match);
				pcl::PointXYZ & pj = cloud_j->at(cor[k].index_query);

				if (std::isfinite(pi.x) && std::isfinite(pi.y)
						&& std::isfinite(pi.z) && std::isfinite(pi.normal_x)
						&& std::isfinite(pi.normal_y)
						&& std::isfinite(pi.normal_z) && std::isfinite(pj.x)
						&& std::isfinite(pj.y) && std::isfinite(pj.z)) {

					float error =
							(pi.getVector3fMap() - pj.getVector3fMap()).dot(
									pi.getNormalVector3fMap());

					Eigen::Matrix<float, 1, 6> Ji, Jj;

					Jj(0, 0) = -Miw(0, 0) * pi.normal_x
							- Miw(1, 0) * pi.normal_y - Miw(2, 0) * pi.normal_z;
					Jj(0, 1) = -Miw(0, 1) * pi.normal_x
							- Miw(1, 1) * pi.normal_y - Miw(2, 1) * pi.normal_z;
					Jj(0, 2) = -Miw(0, 2) * pi.normal_x
							- Miw(1, 2) * pi.normal_y - Miw(2, 2) * pi.normal_z;
					Jj(0, 3) = (Miw(0, 1) * pi.normal_x
							+ Miw(1, 1) * pi.normal_y + Miw(2, 1) * pi.normal_z)
							* (Mwj(2, 0) * pj.x + Mwj(2, 1) * pj.y
									+ Mwj(2, 2) * pj.z + Mwj(2, 3))
							- (Miw(0, 2) * pi.normal_x + Miw(1, 2) * pi.normal_y
									+ Miw(2, 2) * pi.normal_z)
									* (Mwj(1, 0) * pj.x + Mwj(1, 1) * pj.y
											+ Mwj(1, 2) * pj.z + Mwj(1, 3));
					Jj(0, 4) = -(Miw(0, 0) * pi.normal_x
							+ Miw(1, 0) * pi.normal_y + Miw(2, 0) * pi.normal_z)
							* (Mwj(2, 0) * pj.x + Mwj(2, 1) * pj.y
									+ Mwj(2, 2) * pj.z + Mwj(2, 3))
							+ (Miw(0, 2) * pi.normal_x + Miw(1, 2) * pi.normal_y
									+ Miw(2, 2) * pi.normal_z)
									* (Mwj(0, 0) * pj.x + Mwj(0, 1) * pj.y
											+ Mwj(0, 2) * pj.z + Mwj(0, 3));
					Jj(0, 5) = (Miw(0, 0) * pi.normal_x
							+ Miw(1, 0) * pi.normal_y + Miw(2, 0) * pi.normal_z)
							* (Mwj(1, 0) * pj.x + Mwj(1, 1) * pj.y
									+ Mwj(1, 2) * pj.z + Mwj(1, 3))
							- (Miw(0, 1) * pi.normal_x + Miw(1, 1) * pi.normal_y
									+ Miw(2, 1) * pi.normal_z)
									* (Mwj(0, 0) * pj.x + Mwj(0, 1) * pj.y
											+ Mwj(0, 2) * pj.z + Mwj(0, 3));

					Ji = -Jj;

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
}

void reduce_jacobian_icp_p2p::join(reduce_jacobian_icp_p2p& rb) {
	JtJ += rb.JtJ;
	Jte += rb.Jte;
}

/*
 * octomap_server.h
 *
 *  Created on: Jul 23, 2013
 *      Author: vsu
 */

#ifndef OCTOMAP_SERVER_H_
#define OCTOMAP_SERVER_H_

#include <octomap_server/OctomapServer.h>

class RmOctomapServer: public octomap_server::OctomapServer {

public:

	typedef boost::shared_ptr<RmOctomapServer> Ptr;

	RmOctomapServer(ros::NodeHandle private_nh_, std::string & prefix) :
			OctomapServer(private_nh_) {
		m_mapPub = m_nh.advertise < nav_msgs::OccupancyGrid
				> ("/" + prefix + "/map", 5, m_latchedTopics);
		m_worldFrameId = "/" + prefix + "/map";
		m_baseFrameId = "/" + prefix + "/base_footprint";

	}

	virtual void insertScan(const tf::Point& sensorOrigin,
			const PCLPointCloud& ground, const PCLPointCloud& nonground) {
		octomap_server::OctomapServer::insertScan(sensorOrigin, ground,
				nonground);
	}

	void publishAll(const ros::Time& rostime = ros::Time::now()) {
		octomap_server::OctomapServer::publishAll(rostime);
	}

};

#endif /* OCTOMAP_SERVER_H_ */

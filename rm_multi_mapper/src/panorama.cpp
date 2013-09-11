/*
 * panorama.cpp
 *
 *  Created on: Sept 10, 2013
 *      Author: mayanks43
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>

#include <keyframe_map.h>

#include "ros/ros.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Int32.h"
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include "rm_multi_mapper/WorkerAction.h"
#include "rm_multi_mapper/Matrix.h"

/*MySQL includes */
#include "mysql_connection.h"
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/prepared_statement.h>

void get_pairs(std::vector<std::pair<int, int> > & overlapping_keyframes) {
    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::PreparedStatement *pstmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "root", "123456");
        /* Connect to the MySQL test database */
        con->setSchema("panorama");

        /* Select in ascending order */
        pstmt = con->prepareStatement("SELECT f1.id as id1, f2.id as id2 FROM positions f1, positions f2 WHERE (abs(f1.q0*f2.q0 + f1.q1*f2.q1 + f1.q2*f2.q2 + f1.q3*f2.q3) >=1.0 OR 2*acos(abs(f1.q0*f2.q0 + f1.q1*f2.q1 + f1.q2*f2.q2 + f1.q3*f2.q3)) < pi()/6) AND f1.id <> f2.id;");
        res = pstmt->executeQuery();

        while (res->next())
        {
            overlapping_keyframes.push_back(std::make_pair(res->getInt("id1"), 
                                                           res->getInt("id2")));
        }


        delete res;
        delete pstmt;
        delete con;

    } catch (sql::SQLException &e) {
        std::cout << "# ERR: SQLException in " << __FILE__;
        std::cout << "(" << __FUNCTION__ << ") on line " 
        << __LINE__ <<std::endl;
        std::cout << "# ERR: " << e.what();
        std::cout << " (MySQL error code: " << e.getErrorCode();
        std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
    }
}

int main(int argc, char **argv) {

	std::vector<std::pair<int, int> > overlapping_keyframes;

    ros::init(argc, argv, "panorama");
	ros::NodeHandle n;
	
    actionlib::SimpleActionClient<rm_multi_mapper::WorkerAction> ac("worker1", true);

	get_pairs(overlapping_keyframes);
	rm_multi_mapper::WorkerGoal goal;
    for(int i=0; i<overlapping_keyframes.size();i++)
    {
        rm_multi_mapper::KeyframePair keyframe;
        keyframe.first = overlapping_keyframes[i].first;
        keyframe.second = overlapping_keyframes[i].second;
        std::cout<<overlapping_keyframes[i].first<<" "
            <<overlapping_keyframes[i].second<<std::endl;
        goal.Overlap.push_back(keyframe);
    }


    ROS_INFO("Waiting for action server to start.");
    ac.waitForServer(); //will wait for infinite time

    ROS_INFO("Action server started, sending goal.");
    
    // send a goal to the action
    ac.sendGoal(goal);

    //wait for the action to return
    bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

    if (finished_before_timeout)
    {
        actionlib::SimpleClientGoalState state = ac.getState();
        ROS_INFO("Action finished: %s",state.toString().c_str());
        rm_multi_mapper::Matrix Jte = ac.getResult()->Jte;
        rm_multi_mapper::Matrix JtJ = ac.getResult()->JtJ;
        std::cout<<Jte.matrix[0].matrow[0]<<std::endl;
    }
    else ROS_INFO("Action did not finish before the time out.");
    

}

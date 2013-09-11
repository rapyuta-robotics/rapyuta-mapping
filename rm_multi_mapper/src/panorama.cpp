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
#include <cstdlib>

#include <keyframe_map.h>
#include <util.h>

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
        sql::ResultSet *res;
        util U;
        res = U.sql_query("SELECT f1.id as id1, f2.id as id2 FROM positions f1, positions f2 WHERE (abs(f1.q0*f2.q0 + f1.q1*f2.q1 + f1.q2*f2.q2 + f1.q3*f2.q3) >=1.0 OR 2*acos(abs(f1.q0*f2.q0 + f1.q1*f2.q1 + f1.q2*f2.q2 + f1.q3*f2.q3)) < pi()/6) AND f1.id <> f2.id;");

        while (res->next())
        {
            overlapping_keyframes.push_back(std::make_pair(res->getInt("id1"), 
                                                           res->getInt("id2")));
        }
        delete res;

}

void matrix2eigen(const rm_multi_mapper::Matrix & m1, Eigen::MatrixXf & eigen) {

    for(int i=0;i<(int)m1.matrix.size();i++)
    {
        for(int j=0;j<(int)m1.matrix[0].vector.size();j++)
        {
            eigen(i,j) = m1.matrix[i].vector[j];
            
        }
    }    
}

void vector2eigen(const rm_multi_mapper::Vector & v1, Eigen::VectorXf & eigen) {
    for(int i=0;i<(int)v1.vector.size();i++)
    {
        eigen(i) = v1.vector[i];
    }
}

int main(int argc, char **argv) {

	std::vector<std::pair<int, int> > overlapping_keyframes;
	std::vector<std::pair<int, int> > overlap1;
	std::vector<std::pair<int, int> > overlap2;
	std::vector<color_keyframe::Ptr> frames;
	int size;

    ros::init(argc, argv, "panorama");
	ros::NodeHandle n;
	
    actionlib::SimpleActionClient<rm_multi_mapper::WorkerAction> ac1(argv[1], true);
    actionlib::SimpleActionClient<rm_multi_mapper::WorkerAction> ac2(argv[2], true);
    sql::ResultSet *res;

    util U;
    U.load("http://localhost/keyframe_map", frames);
    size = frames.size();
	get_pairs(overlapping_keyframes);
	rm_multi_mapper::WorkerGoal goal1;
	rm_multi_mapper::WorkerGoal goal2;
	
    for(int i=0; i<(int)overlapping_keyframes.size()/2;i++)
    {
        rm_multi_mapper::KeyframePair keyframe;
        keyframe.first = overlapping_keyframes[i].first;
        keyframe.second = overlapping_keyframes[i].second;
        goal1.Overlap.push_back(keyframe);
    }
    
    for(int i=overlapping_keyframes.size()/2; i<(int)overlapping_keyframes.size();i++)
    {
        rm_multi_mapper::KeyframePair keyframe;
        keyframe.first = overlapping_keyframes[i].first;
        keyframe.second = overlapping_keyframes[i].second;
        goal2.Overlap.push_back(keyframe);
    }

    ROS_INFO("Waiting for action server to start.");
    ac1.waitForServer(); 
    ac2.waitForServer();

    ROS_INFO("Action server started, sending goal.");
    
    // send a goal to the action
    ac1.sendGoal(goal1);
    ac2.sendGoal(goal2);

    //wait for the action to return
    bool finished_before_timeout1 = ac1.waitForResult(ros::Duration(30.0));
    bool finished_before_timeout2 = ac2.waitForResult(ros::Duration(30.0));

    Eigen::MatrixXf JtJ;
    Eigen::VectorXf Jte;
    Eigen::MatrixXf JtJ2;
    Eigen::VectorXf Jte2;
    if (finished_before_timeout1 && finished_before_timeout2)
    {

       	JtJ.setZero(size * 3 + 3, size * 3 + 3);
    	Jte.setZero(size * 3 + 3);
    	
       	JtJ2.setZero(size * 3 + 3, size * 3 + 3);
    	Jte2.setZero(size * 3 + 3);
    	
    	rm_multi_mapper::Vector rosJte = ac1.getResult()->Jte;
    	rm_multi_mapper::Matrix rosJtJ = ac1.getResult()->JtJ;
    	
    	rm_multi_mapper::Vector rosJte2 = ac2.getResult()->Jte;
    	rm_multi_mapper::Matrix rosJtJ2 = ac2.getResult()->JtJ;
    	
        vector2eigen(rosJte, Jte);
        matrix2eigen(rosJtJ, JtJ);
        
        vector2eigen(rosJte2, Jte2);
        matrix2eigen(rosJtJ2, JtJ2);
        
        JtJ += JtJ2;
        Jte += Jte2;
        
    }
    else 
    {
        ROS_INFO("Action did not finish before the time out.");
        std::exit(0);
    }
    
    Eigen::VectorXf update = -JtJ.ldlt().solve(Jte);

    float iteration_max_update = std::max(std::abs(update.maxCoeff()),
            std::abs(update.minCoeff()));

    ROS_INFO("Max update %f", iteration_max_update);

    for (int i = 0; i < (int)frames.size(); i++) {

        frames[i]->get_pos().so3() = Sophus::SO3f::exp(update.segment<3>(i * 3))
                * frames[i]->get_pos().so3();
        frames[i]->get_pos().translation() = frames[0]->get_pos().translation();
        frames[i]->get_intrinsics().array() =
                update.segment<3>(size * 3).array().exp()
		                * frames[i]->get_intrinsics().array();

        std::string query = "UPDATE `positions` SET `q0` = " + 
        boost::lexical_cast<std::string>(frames[i]->get_pos().so3().data()[0]) +
        ", `q1` = " +
        boost::lexical_cast<std::string>(frames[i]->get_pos().so3().data()[1]) +
        ", `q2` = " +
        boost::lexical_cast<std::string>(frames[i]->get_pos().so3().data()[2]) +
        ", `q3` = " +
        boost::lexical_cast<std::string>(frames[i]->get_pos().so3().data()[3]) +
        ", `t0` = " +
		boost::lexical_cast<std::string>(frames[i]->get_pos().translation()[0]) +
		", `t1` = " +
		boost::lexical_cast<std::string>(frames[i]->get_pos().translation()[1]) +
		", `t2` = " +
		boost::lexical_cast<std::string>(frames[i]->get_pos().translation()[2]) +
        ", `int0` = " +
       	boost::lexical_cast<std::string>(frames[i]->get_intrinsics().array()[0]) +
       	", `int1` = " +
       	boost::lexical_cast<std::string>(frames[i]->get_intrinsics().array()[1]) +
       	", `int2` = " +
       	boost::lexical_cast<std::string>(frames[i]->get_intrinsics().array()[2]) +
       	" WHERE `id` = " +
       	boost::lexical_cast<std::string>(i) +
       	";";

        res = U.sql_query(query);
        delete res;
        

    }
    

}

/*
 * worker.cpp
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
#include <reduce_jacobian_ros.h>
#include <util.h>

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include "rm_multi_mapper/WorkerAction.h"

/*MySQL includes */
#include "mysql_connection.h"
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/prepared_statement.h>

class WorkerAction
{
    protected:
        ros::NodeHandle nh_;
        actionlib::SimpleActionServer<rm_multi_mapper::WorkerAction> as_; 
        std::string action_name_;
        rm_multi_mapper::WorkerFeedback feedback_;
        rm_multi_mapper::WorkerResult result_;
        std::vector<color_keyframe::Ptr> frames_;

    public:

        WorkerAction(std::string name) :
            as_(nh_, name, boost::bind(&WorkerAction::executeCB, this, _1), false),
            action_name_(name)
        {
            as_.start();
        }

        ~WorkerAction(void)
        {
        }

        void eigen2vector(rm_multi_mapper::Vector & v1, const Eigen::VectorXf & Jte) {
            for(int i=0;i<Jte.size();i++)
            {
                v1.vector.push_back(Jte[i]);
            }    
        }
        
        void eigen2matrix(rm_multi_mapper::Matrix & m1, const Eigen::MatrixXf & JtJ) {
            for(int i=0;i<JtJ.rows();i++)
            {
                rm_multi_mapper::Vector row;
                for(int j=0;j<JtJ.cols();j++)
                {
                    row.vector.push_back(JtJ(i,j));

                }
                m1.matrix.push_back(row);
            }
        }

        void executeCB(const rm_multi_mapper::WorkerGoalConstPtr &goal)
        {
            ros::Rate r(1);
            bool success = true;

            if (as_.isPreemptRequested() || !ros::ok())
            {
                ROS_INFO("%s: Preempted", action_name_.c_str());
                as_.setPreempted();
                success = false;
            }
            
            util U;
            U.load("http://localhost/keyframe_map", frames_); 
            
            reduce_jacobian_ros rj(frames_, frames_.size(), 0);
            
            rj.reduce(goal);
            
            if(success)
            {
            
                eigen2vector(result_.Jte, rj.Jte);

                eigen2matrix(result_.JtJ, rj.JtJ);

                ROS_INFO("%s: Succeeded", action_name_.c_str());
                as_.setSucceeded(result_);
            }
        }


};


int main(int argc, char** argv)
{
    ros::init(argc, argv, argv[1]);
    WorkerAction worker(ros::this_node::getName());
    ros::spin();

    return 0;
}


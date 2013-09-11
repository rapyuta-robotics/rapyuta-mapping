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
        // create messages that are used to published feedback/result
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



        void executeCB(const rm_multi_mapper::WorkerGoalConstPtr &goal)
        {
            // helper variables
            ros::Rate r(1);
            bool success = true;

            // start executing the action
            // check that preempt has not been requested by the client
            if (as_.isPreemptRequested() || !ros::ok())
            {
                ROS_INFO("%s: Preempted", action_name_.c_str());
                // set the action state to preempted
                as_.setPreempted();
                success = false;
            }
            
            util U;
            U.load("http://localhost/keyframe_map", frames_); //will replace with server
            
            reduce_jacobian_ros rj(frames_, frames_.size(), 0);
            
            rj.reduce(goal);
            
            if(success)
            {
            
                //result_.Jte = rj.Jte;
                for(int i=0;i<rj.Jte.rows();i++)
                {
                    rm_multi_mapper::MatRow row;
                    for(int j=0;j<rj.Jte.cols();j++)
                    {
                        row.matrow.push_back(rj.Jte(i,j));

                    }
                    result_.Jte.matrix.push_back(row);
                }
                
                //result_.JtJ = rj.JtJ;                
                for(int i=0;i<rj.JtJ.rows();i++)
                {
                    rm_multi_mapper::MatRow row;
                    for(int j=0;j<rj.JtJ.cols();j++)
                    {
                        row.matrow.push_back(rj.JtJ(i,j));

                    }
                    result_.JtJ.matrix.push_back(row);
                }

                ROS_INFO("%s: Succeeded", action_name_.c_str());
                // set the action state to succeeded
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


/*
 * g2o_worker.cpp
 *
 *  Created on: Sept 29, 2013
 *      Author: mayanks43
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>

#include <keyframe_map.h>
#include <reduce_measurement_g2o_dist.h>
#include <util.h>

#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include "rm_multi_mapper/G2oWorkerAction.h"

class G2oWorkerAction
{
    protected:
        ros::NodeHandle nh_;
        actionlib::SimpleActionServer<rm_multi_mapper::G2oWorkerAction> as_;
        std::string action_name_;
        rm_multi_mapper::G2oWorkerFeedback feedback_;
        rm_multi_mapper::G2oWorkerResult result_;
        boost::shared_ptr<keyframe_map> map;
        util U;

    public:

        G2oWorkerAction(std::string name) :
            as_(nh_, name, boost::bind(&G2oWorkerAction::executeCB, this, _1),
            		false),
            action_name_(name)
        {
            as_.start();

        }

        ~G2oWorkerAction(void)
        {
        }

        void executeCB(const rm_multi_mapper::G2oWorkerGoalConstPtr &goal)
        {
        	map = U.get_robot_map(0);
            ros::Rate r(1);
            bool success = true;

            if (as_.isPreemptRequested() || !ros::ok())
            {
                ROS_INFO("%s: Preempted", action_name_.c_str());
                as_.setPreempted();
                success = false;
            }


            reduce_measurement_g2o_dist rj(map->frames, map->frames.size());

            rj.reduce(goal);

            U.save_measurements(rj.m);

            std::cout<<"Done";
            if(success)
            {
                ROS_INFO("%s: Succeeded", action_name_.c_str());
                result_.reply = true;
                as_.setSucceeded(result_);
            }
        }


};


int main(int argc, char** argv)
{
    ros::init(argc, argv, argv[1]);
    G2oWorkerAction worker(ros::this_node::getName());
    ros::spin();

    return 0;
}


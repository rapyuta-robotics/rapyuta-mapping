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
#include <web_image_loader.h>

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

        void load_mysql(std::vector<std::pair<Sophus::SE3f, Eigen::Vector3f> > & positions) {

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
                    pstmt = con->prepareStatement("SELECT * FROM positions");
                    res = pstmt->executeQuery();

                    while (res->next())
                    {
                        Eigen::Quaternionf q;
		                Eigen::Vector3f t;
                		Eigen::Vector3f intrinsics;
                        q.coeffs()[0] = res->getDouble("q0");
                        q.coeffs()[1] = res->getDouble("q1");
                        q.coeffs()[2] = res->getDouble("q2");
                        q.coeffs()[3] = res->getDouble("q3");
                        t[0] = res->getDouble("t0");
                        t[1] = res->getDouble("t1");
                        t[2] = res->getDouble("t2");
                        intrinsics[0] = res->getDouble("int0");
                        intrinsics[1] = res->getDouble("int1");
                        intrinsics[2] = res->getDouble("int2");
                        positions.push_back(std::make_pair(Sophus::SE3f(q, t), intrinsics));
                    }

                    delete res;
                    delete pstmt;
                    delete con;

                } catch (sql::SQLException &e) {
                    std::cout << "# ERR: SQLException in " << __FILE__;
                    std::cout << "(" << __FUNCTION__ << ") on line " 
                    << __LINE__ << std::endl;
                    std::cout << "# ERR: " << e.what();
                    std::cout << " (MySQL error code: " << e.getErrorCode();
                    std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
                }

        }

        void load(const std::string & dir_name) {

	        std::vector<std::pair<Sophus::SE3f, Eigen::Vector3f> > positions;

	        load_mysql(positions);

	        std::cerr << "Loaded " << positions.size() << " positions" << std::endl;
            web_image_loader loader;
	        for (size_t i = 0; i < positions.size(); i++) {
		        cv::Mat rgb = loader.stringtoMat(
				        dir_name + "/rgb/" + boost::lexical_cast<std::string>(i)
						        + ".png");
		        cv::Mat depth = loader.stringtoMat(
				        dir_name + "/depth/" + boost::lexical_cast<std::string>(i)
						        + ".png");

		        cv::Mat gray;
		        cv::cvtColor(rgb, gray, CV_RGB2GRAY);

		        color_keyframe::Ptr k(
				        new color_keyframe(rgb, gray, depth, positions[i].first,
						        positions[i].second));
		        frames_.push_back(k);
	        }

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
            
            load("http://localhost/keyframe_map"); //will replace with server
            
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
                
                Eigen::VectorXf update = -rj.JtJ.ldlt().solve(rj.Jte);

                float iteration_max_update = std::max(std::abs(update.maxCoeff()),
		                std::abs(update.minCoeff()));

                ROS_INFO("Max update %f", iteration_max_update);

                for (int i = 0; i < (int)frames_.size(); i++) {

	                frames_[i]->get_pos().so3() = Sophus::SO3f::exp(update.segment<3>(i * 3))
			                * frames_[i]->get_pos().so3();
	                frames_[i]->get_pos().translation() = frames_[0]->get_pos().translation();
                    //std::cout<<frames[i]->get_pos();
	                frames_[i]->get_intrinsics().array() =
			                update.segment<3>(frames_.size() * 3).array().exp()
					                * frames_[i]->get_intrinsics().array();
	                if (i == 0) {
		                Eigen::Vector3f intrinsics = frames_[i]->get_intrinsics();
		                ROS_INFO("New intrinsics %f, %f, %f", intrinsics(0), intrinsics(1),
				                intrinsics(2));
	                }

                }


                ROS_INFO("%s: Succeeded", action_name_.c_str());
                // set the action state to succeeded
                as_.setSucceeded(result_);
            }
        }


};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "worker1");
    WorkerAction worker(ros::this_node::getName());
    ros::spin();

    return 0;
}


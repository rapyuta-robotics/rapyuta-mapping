#include "Poco/URIStreamOpener.h"
#include "Poco/StreamCopier.h"
#include "Poco/Path.h"
#include "Poco/URI.h"
#include "Poco/Exception.h"
#include "Poco/Net/HTTPStreamFactory.h"
#include <memory>
#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include "opencv2/opencv.hpp"

/*MySQL includes */
#include "mysql_connection.h"
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/prepared_statement.h>

#include <keyframe_map.h>

static bool factoryLoaded = false;

class DataBuf : public streambuf
{
public:
   DataBuf(char * d, size_t s) {
      setg(d, d, d + s);
   }
};

class util {
    public :

		util();
		~util();

        cv::Mat loadFromURL(std::string url);
        cv::Mat stringtoMat(std::string file);
        sql::ResultSet* sql_query(std::string query);
        void load_mysql(std::vector<std::pair<Sophus::SE3f, Eigen::Vector3f> > & positions);
        void load(const std::string & dir_name, std::vector<color_keyframe::Ptr> & frames);

        int get_new_robot_id();
        void add_keyframe(int robot_id, const color_keyframe::Ptr & k);
        color_keyframe::Ptr get_keyframe(long frame_id);

    private:
        sql::Driver *driver;
        sql::Connection *con;
};



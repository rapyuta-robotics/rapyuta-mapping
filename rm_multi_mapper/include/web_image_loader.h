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

static bool factoryLoaded = false;

class web_image_loader {
    public :
        cv::Mat loadFromURL(std::string url);
        cv::Mat stringtoMat(std::string file);
};



#include <web_image_loader.h>

using Poco::URIStreamOpener;
using Poco::StreamCopier;
using Poco::Path;
using Poco::URI;
using Poco::Exception;
using Poco::Net::HTTPStreamFactory;
using namespace std;

cv::Mat web_image_loader::loadFromURL(string url)
{
    //Don't register the factory more than once
    if(!factoryLoaded){
        HTTPStreamFactory::registerFactory();
        factoryLoaded = true;
    }

    //Specify URL and open input stream
    URI uri(url);
    std::auto_ptr<std::istream> pStr(URIStreamOpener::defaultOpener().open(uri));

    //Copy image to our string and convert to cv::Mat
    string str;
    StreamCopier::copyToString(*pStr.get(), str);
    vector<char> data( str.begin(), str.end() );
    cv::Mat data_mat(data);
    cv::Mat image(cv::imdecode(data_mat,1));
    return image;
}
    
cv::Mat web_image_loader::stringtoMat(string file)
{
    cv::Mat image;

    if(file.compare(file.size()-4,4,".gif")==0)
    {
        cerr<<"UNSUPPORTED_IMAGE_FORMAT";
        return image;
    }

    else if(file.compare(0,7,"http://")==0)  // Valid URL only if it starts with "http://"
    {
        image = loadFromURL(file);
        if(!image.data)
            cerr<<"INVALID_IMAGE_URL";
        return image;
    }
    else
    {
        image = cv::imread(file,1); // Try if the image path is in the local machine
        if(!image.data)
            cerr<<"IMAGE_DOESNT_EXIST";
        return image;
    }
}




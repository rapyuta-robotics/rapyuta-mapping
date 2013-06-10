import cv2, numpy as np
import itertools
rgbg = cv2.imread("../rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png")
depth = cv2.imread("../rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png", cv2.CV_LOAD_IMAGE_UNCHANGED)
rgb2g = cv2.imread("../rgbd_dataset_freiburg1_desk/rgb/1305031453.391690.png")
depth2 = cv2.imread("../rgbd_dataset_freiburg1_desk/depth/1305031453.404816.png", cv2.CV_LOAD_IMAGE_UNCHANGED)

rgb = cv2.cvtColor(rgbg, cv2.COLOR_BGR2GRAY)

surfDetector = cv2.FeatureDetector_create("SURF")
surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")

keypoints = surfDetector.detect(rgb, (depth != 0).view(np.uint8))
(keypoints, descriptors) = surfDescriptorExtractor.compute(rgb, keypoints)


samples = np.array(descriptors)
responses = np.arange(len(keypoints),dtype = np.float32)

knn = cv2.KNearest()
knn.train(samples,responses)

cam = np.identity(4)
K = np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]])

rgb2 = cv2.cvtColor(rgb2g, cv2.COLOR_BGR2GRAY)
keypoints2 = surfDetector.detect(rgb2, (depth2 != 0).view(np.uint8))
(keypoints2, descriptors2) = surfDescriptorExtractor.compute(rgb2, keypoints2)

for h,des in enumerate(descriptors2):
    des = np.array(des,np.float32).reshape((1,128))
    retval, results, neigh_resp, dists = knn.find_nearest(des,1)
    res,dist =  int(results[0][0]),dists[0][0]
    print res, dist
    if dist<0.2: # draw matched keypoints in red color
        color = (0,0,255)
    else:  # draw unmatched in blue color
        
        color = (255,0,0)

    #Draw matched key points on original image
    x,y = keypoints[res].pt
    center = (int(x),int(y))
    cv2.circle(rgbg,center,2,color,-1)

    #Draw matched key points on template image
    x,y = keypoints2[h].pt
    center = (int(x),int(y))
    cv2.circle(rgb2g,center,2,color,-1)

cv2.imshow('img',rgbg)
cv2.imshow('tm',rgb2g)
cv2.waitKey(0)
cv2.destroyAllWindows()

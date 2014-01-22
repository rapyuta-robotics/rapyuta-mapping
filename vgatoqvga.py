#!/usr/bin/python
#
# this script requires ROS diamondback
# for installation instructions, see http://www.ros.org 

import argparse
import sys
import os
import rospy
import rosbag
import sensor_msgs.msg
import cv2, cv
from cv_bridge import CvBridge, CvBridgeError
import struct
import tf
import numpy as np
import math

bridge = CvBridge()

if __name__ == '__main__':
    
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This scripts reads a bag file containing RGBD data, 
    converts the depth and rgb images to qvga, and saves it again into a bag file.
    ''')

    parser.add_argument('--compress', help='compress output bag file', action='store_true')
    parser.add_argument('inputbag', help='input bag file')
    args = parser.parse_args()
    
    args.outputbag = os.path.splitext(args.inputbag)[0] + "-qvga.bag"
      
    print "Processing bag file:"
    print "  in:",args.inputbag
    print "  out:",args.outputbag

    inbag = rosbag.Bag(args.inputbag,'r')
    if args.compress:
        param_compression = rosbag.bag.Compression.BZ2
    else:
        param_compression = rosbag.bag.Compression.NONE
        
    outbag = rosbag.Bag(args.outputbag, 'w', compression=param_compression)
    
    for topic, msg, t in inbag.read_messages():

        if topic == "/camera/depth/camera_info":
            depth_camera_info = msg
            depth_camera_info.height /= 2
            depth_camera_info.width /= 2
            
            K = np.array(depth_camera_info.K)/2.0
            K[-1] = 1.0
            depth_camera_info.K = tuple(K)

            P = np.array(depth_camera_info.P)/2.0
            P[-1] = 1.0
            depth_camera_info.P = tuple(P)

            outbag.write("/cloudbot0/depth/camera_info",depth_camera_info,t)            
            continue

        if topic == "/camera/rgb/camera_info":
            rgb_camera_info = msg

            rgb_camera_info.height /= 2
            rgb_camera_info.width /= 2

            K = np.array(rgb_camera_info.K)/2.0
            K[-1] = 1.0
            rgb_camera_info.K = tuple(K)

            P = np.array(rgb_camera_info.P)/2.0
            P[-1] = 1.0
            rgb_camera_info.P = tuple(P)

            outbag.write("/cloudbot0/rgb/camera_info",rgb_camera_info,t)
            continue
            
        if topic == "/camera/rgb/image_color":
            rgb_image_color = msg
            
            cv_rgb_image = np.asarray(bridge.imgmsg_to_cv(rgb_image_color, 'rgb8'))
            qvga_cv_rgb_image = (cv_rgb_image[::2,::2,:].astype(np.uint32) + cv_rgb_image[1::2,::2,:] + cv_rgb_image[::2,1::2,:] + cv_rgb_image[1::2,1::2,:]) / 4.0
            qvga_cv_rgb_image = qvga_cv_rgb_image.astype(np.uint8)
            
            #qvga_cv_yuv_image = cv2.cvtColor(qvga_cv_rgb_image, cv.CV_BGR2YCrCb)

            new_rgb_image_color = bridge.cv_to_imgmsg(cv.fromarray(qvga_cv_rgb_image), "rgb8")
            new_rgb_image_color.header = rgb_image_color.header

            outbag.write("/cloudbot0/rgb/image_raw", new_rgb_image_color,t)

            continue

        
        if topic == "/camera/depth/image":
            depth_image = msg
            cv_depth_image = np.asarray(bridge.imgmsg_to_cv(depth_image))

            cv_depth_image[np.isnan(cv_depth_image)] = 0
            cv_depth_image = (cv_depth_image * 1000).astype(np.uint16)

            # Median filtering of the depth image
            depth_qvga_sort = np.empty((cv_depth_image.shape[0]/2, cv_depth_image.shape[1]/2, 4), dtype=np.uint16)
            depth_qvga_sort[:,:,0] = cv_depth_image[::2,::2]
            depth_qvga_sort[:,:,1] = cv_depth_image[1::2,::2]
            depth_qvga_sort[:,:,2] = cv_depth_image[::2,1::2]
            depth_qvga_sort[:,:,3] = cv_depth_image[1::2,1::2]

            depth_qvga_sort = np.sort(depth_qvga_sort, axis=2)

            new_depth_image = bridge.cv_to_imgmsg(cv.fromarray(depth_qvga_sort[:,:,2].copy()))
            new_depth_image.header = depth_image.header

            outbag.write("/cloudbot0/depth/image_raw", new_depth_image,t)

            continue

        if topic == "/tf":
            new_transforms = []
            for i in range(len(msg.transforms)):
                if msg.transforms[i].header.frame_id not in ['/world', '/kinect']:
                    new_transforms.append(msg.transforms[i])
            msg.transforms = new_transforms
            if(len(msg.transforms) > 0):
                outbag.write("/tf", msg, t)
 
        # anything else: pass through
        outbag.write(topic,msg,t)
                
    print 'Done'
    outbag.close()


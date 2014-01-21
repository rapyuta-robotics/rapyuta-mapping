#!/usr/bin/python
#
# this script requires ROS diamondback
# for installation instructions, see http://www.ros.org 

import argparse
import sys
import os

if __name__ == '__main__':
    
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This scripts reads a bag file containing RGBD data, 
    converts the depth and rgb images to qvga, and saves it again into a bag file. 
    Optional arguments allow to select only a portion of the original bag file.  
    ''')
    parser.add_argument('--start', help='skip the first N seconds of  input bag file (default: 0.0)',default=0.00)
    parser.add_argument('--duration', help='only process N seconds of input bag file (default: off)')
    parser.add_argument('--nth', help='only process every N-th frame of input bag file (default: 15)',default=15)
    parser.add_argument('--skip', help='skip N blocks in the beginning (default: 1)', default=1)
    parser.add_argument('--compress', help='compress output bag file', action='store_true')
    parser.add_argument('inputbag', help='input bag file')
    parser.add_argument('outputbag', nargs='?',help='output bag file')
    args = parser.parse_args()

    #import roslib; roslib.load_manifest('rgbd_benchmark_tools')
    import rospy
    import rosbag
    import sensor_msgs.msg
    import cv2, cv
    from cv_bridge import CvBridge, CvBridgeError
    import struct
    import tf
    import numpy as np
    import math
    
    if not args.outputbag:
        args.outputbag = os.path.splitext(args.inputbag)[0] + "-qvga.bag"
      
    print "Processing bag file:"
    print "  in:",args.inputbag
    print "  out:",args.outputbag
    print "  starting from: %s seconds"%(args.start)
        
    if args.duration:
        print "  duration: %s seconds"%(args.duration)
        
    print "  saving every %s-th frame"%(args.nth)
    args.skip = float(args.skip)
    print "  skipping %s blocks"%(args.skip)

    inbag = rosbag.Bag(args.inputbag,'r')
    if args.compress:
        param_compression = rosbag.bag.Compression.BZ2
    else:
        param_compression = rosbag.bag.Compression.NONE
        
    outbag = rosbag.Bag(args.outputbag, 'w', compression=param_compression)
    
    depth_camera_info = None
    rgb_camera_info = None
    depth_image = None
    rgb_image_color = None
    cortex = None

    nan = float('nan')
    bridge = CvBridge()
    frame = 0 
    transforms = dict()
    
    time_start = None
    for topic, msg, t in inbag.read_messages():
        if time_start==None:
            time_start=t
        if t - time_start < rospy.Duration.from_sec(float(args.start)):
            continue
        if args.duration and (t - time_start > rospy.Duration.from_sec(float(args.start) + float(args.duration))):
            break
        print "t=%f\r"%(t-time_start).to_sec(),
        if topic == "/tf":
            for transform in msg.transforms:
                transforms[ (transform.header.frame_id,transform.child_frame_id) ] = transform
            continue
        if topic == "/imu":
            imu = msg
            continue
        if topic == "/camera/depth/camera_info":
            depth_camera_info = msg
            continue
        if topic == "/camera/rgb/camera_info":
            rgb_camera_info = msg
            continue
        if topic == "/camera/rgb/image_color" and rgb_camera_info:
            rgb_image_color = msg
            continue
        if topic == "/camera/depth/image" and depth_camera_info and rgb_image_color and rgb_camera_info and imu:
            depth_image = msg
            # now process frame
            
            if depth_image.header.stamp - rgb_image_color.header.stamp > rospy.Duration.from_sec(1/30.0):
                continue
            
            frame += 1
            if frame % float(args.nth) ==0:
                if args.skip > 0:
                    args.skip -= 1
                else:
                    #print "depth header", depth_image.header.stamp.secs
                    
                    # store messages
                    msg = tf.msg.tfMessage()
                    msg.transforms = list( transforms.itervalues() ) 
                    outbag.write("/tf",msg,t)
                    transforms = dict()
                    outbag.write("/imu",imu,t)
                    
                    cv_depth_image = bridge.imgmsg_to_cv(depth_image)
                    cv_rgb_image = bridge.imgmsg_to_cv(rgb_image_color)
                    cv_depth_image_np = np.asarray(cv_depth_image[:,:])
                    cv_rgb_image_np = np.asarray(cv_rgb_image[:,:])

                    qvga_rgb_image = np.empty([240, 320, 3], dtype='uint8')
                    qvga_depth_image = np.empty([240, 320], dtype='float32')

                    for i,l in zip(range(240), range(0,480,2)):
                        for j,m in zip(range(320), range(0,640,2)):
                            for k in range(3):
                                qvga_rgb_image[i,j,k] = cv_rgb_image_np[l,m,k]/4\
                                                    + cv_rgb_image_np[l+1,m,k]/4 \
                                                    + cv_rgb_image_np[l,m+1,k]/4 \
                                                    + cv_rgb_image_np[l+1,m+1,k]/4
                            
                            pixel4 = cv_depth_image_np[l,m], cv_depth_image_np[l+1,m],\
                                      cv_depth_image_np[l,m+1], cv_depth_image_np[l+1,m+1]
                            pixel4 = sorted([0 if math.isnan(x) else x for x in pixel4])
                            qvga_depth_image[i,j] = pixel4[2]
                    
                    new_depth_image = bridge.cv_to_imgmsg(cv.fromarray(qvga_depth_image))
                    new_depth_image.header = depth_image.header
                    new_rgb_image_color = bridge.cv_to_imgmsg(cv.fromarray(qvga_rgb_image))
                    new_rgb_image_color.header = rgb_image_color.header
                    depth_camera_info.height = 320
                    depth_camera_info.width = 240
                    depth_camera_info.K = [x/2.0 for  x in depth_camera_info.K]
                    depth_camera_info.P = [x/2.0 for  x in depth_camera_info.P]
                    rgb_camera_info.height = 320
                    rgb_camera_info.width = 240
                    rgb_camera_info.K = [x/2.0 for  x in rgb_camera_info.K]
                    rgb_camera_info.P = [x/2.0 for  x in rgb_camera_info.P]
                    
                    
                    #print t
                    outbag.write("/camera/depth/camera_info",depth_camera_info,t)
                    outbag.write("/camera/depth/image", new_depth_image,t)
                    outbag.write("/camera/rgb/camera_info",rgb_camera_info,t)
                    outbag.write("/camera/rgb/image_color", new_rgb_image_color,t)
                    
            # consume the images
            imu = None
            depth_image = None
            rgb_image_color = None
            continue
        if topic not in ["/tf","/imu",
                         "/camera/depth/camera_info","/camera/rgb/camera_info",
                         "/camera/rgb/image_color","/camera/depth/image"]:
            # anything else: pass thru
            outbag.write(topic,msg,t)
                
    print
    outbag.close()


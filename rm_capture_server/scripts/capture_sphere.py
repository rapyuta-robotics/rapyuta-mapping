#!/usr/bin/env python
import roslib
roslib.load_manifest('turtlebot_actions')
roslib.load_manifest('rm_capture_server')

import rospy

import os
import sys
import time
import math
import cv2
import numpy as np
from os.path import join
from turtlebot_actions.msg import *
from actionlib_msgs.msg import *
from rm_capture_server.srv import *


import actionlib

'''
  Very simple move action test - commands the robot to turn 45 degrees and travel 0.5 metres forward.
'''

def main():
  rospy.init_node("test_move_action_client")

  # Construct action ac
  rospy.loginfo("Starting action client...")
  action_client = actionlib.SimpleActionClient('/cloudbot1/turtlebot_move', TurtlebotMoveAction)
  action_client.wait_for_server()
  rospy.loginfo("Action client connected to action server.")

  total_angle = 0

  rospy.wait_for_service('/cloudbot1/capture')
  capture = rospy.ServiceProxy('/cloudbot1/capture', Capture)

  dataset_folder = sys.argv[1]
  rgb_folder = join(dataset_folder, 'rgb')
  depth_folder = join(dataset_folder, 'depth')
  rgb_filename = join(dataset_folder, 'rgb.txt')
  depth_filename = join(dataset_folder, 'depth.txt')
  
  os.makedirs(dataset_folder)
  os.makedirs(rgb_folder)
  os.makedirs(depth_folder)
  
  rgb_file = open(rgb_filename, 'w')
  depth_file = open(depth_filename, 'w')
  

  # Call the action
  for i in range(18):
    rospy.loginfo("Calling the action server...")
    action_goal = TurtlebotMoveGoal()
    action_goal.turn_distance = math.pi/40
    action_goal.forward_distance = 0.0 # metres
    
    if action_client.send_goal_and_wait(action_goal, rospy.Duration(50.0), rospy.Duration(50.0)) == GoalStatus.SUCCEEDED:
      rospy.loginfo('Call to action server succeeded')
      total_angle += action_client.get_result().turn_distance * 180/math.pi
      print action_client.get_result().turn_distance * 180/math.pi
      print total_angle
      time.sleep(1)
      r = capture(0)
      rgb_buf = np.fromstring(r.rgb_png_data, dtype=np.uint8)
      depth_buf = np.fromstring(r.depth_png_data, dtype=np.uint8)
      img = cv2.imdecode(rgb_buf, cv2.CV_LOAD_IMAGE_UNCHANGED)
      depth_img = cv2.imdecode(depth_buf, cv2.CV_LOAD_IMAGE_UNCHANGED)

      cv2.imwrite(join(rgb_folder, str(r.header.stamp.secs) + '.' + str(r.header.stamp.nsecs) + '.png'), img)
      cv2.imwrite(join(depth_folder, str(r.header.stamp.secs) + '.' + str(r.header.stamp.nsecs) + '.png'), depth_img*5)
      rgb_file.write(str(r.header.stamp.secs) + '.' + str(r.header.stamp.nsecs) + ' ' + join('rgb', str(r.header.stamp.secs) + '.' + str(r.header.stamp.nsecs) + '.png') + '\n')
      depth_file.write(str(r.header.stamp.secs) + '.' + str(r.header.stamp.nsecs) + ' ' + join('depth', str(r.header.stamp.secs) + '.' + str(r.header.stamp.nsecs) + '.png') + '\n')

      cv2.imshow('img', img)
      cv2.imshow('depth', depth_img*5)
      cv2.waitKey(2)
    else:
      rospy.logerr('Call to action server failed')
    
  rgb_file.close()
  depth_file.close()


if __name__ == "__main__":
  main()

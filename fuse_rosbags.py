#!/usr/bin/env python

import rosbag
import rospy
import sys

start_time = rospy.Time.from_sec(1370000000)
real_start_time = 0

robot_counter = 0

with rosbag.Bag('fused.bag', 'w') as outbag:

    for bag_name in sys.argv[1:]:
        real_start_time = 0
        robot_counter += 1
        prefix = "/cloudbot" + str(robot_counter)

        for topic, msg, t in rosbag.Bag(bag_name).read_messages():

            if real_start_time == 0:
                real_start_time = t

            if topic == "/tf" and msg.transforms:
                for i in range(len(msg.transforms)):
                    msg.transforms[0].header.frame_id = prefix + msg.transforms[0].header.frame_id
                    msg.transforms[0].child_frame_id = prefix + msg.transforms[0].child_frame_id
		    msg.transforms[0].header.stamp = start_time + (msg.transforms[0].header.stamp - real_start_time) 
                outbag.write(topic, msg, msg.transforms[0].header.stamp)
            else:
                msg.header.frame_id = prefix + msg.header.frame_id
		msg.header.stamp = start_time + (msg.header.stamp - real_start_time)
                outbag.write(prefix + topic, msg, msg.header.stamp)

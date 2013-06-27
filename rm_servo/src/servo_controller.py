#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState

import sys
import alsaaudio
import numpy as np
import math

period = 640 * 3

zero_correction = 0
zero_pos = 144 + zero_correction
gain = 57.6/(np.pi/4)
gain_inv = 1.0/gain

ii16 = np.iinfo(np.int16)
data = np.zeros(period, dtype=np.int16)
data[0:zero_pos] = ii16.max

current_angle = np.zeros(1, dtype=np.float32)
angles = np.zeros(50, dtype=np.float32)

joint_states_pub = rospy.Publisher('joint_states', JointState)

def callback(angle_data):
	angle = angle_data.data
	if angle < -np.pi/4 or angle > np.pi/4:
		rospy.loginfo(rospy.get_name() + ": Angle %f is bigger then servo limit" % angle)
	#val = int(round(zero_pos + angle*gain))
	#data[:] = 0
	#data[0:val] = ii16.max
	current_angle[:] = angle


if __name__ == '__main__':
	
	rospy.init_node('servo_controller',)
	rospy.Subscriber("servo_angle", Float32, callback)
	
	card = 'default'

	# Open the device in playback mode. 
	out = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK, card=card)

	# Set attributes: Mono, 44100 Hz, 16 bit little endian frames
	out.setchannels(1)
	out.setrate(96000)
	out.setformat(alsaaudio.PCM_FORMAT_S16_LE)

	# The period size controls the internal number of frames per period.
	# The significance of this parameter is documented in the ALSA api.
	out.setperiodsize(period)

	while not rospy.is_shutdown():
		angles = np.roll(angles, -1)
		angles[-1] = current_angle[0]
		
		mean_angle = np.mean(angles)
		
		val = int(round(zero_pos + mean_angle*gain))
		data[:] = 0
	        data[0:val] = ii16.max

		
		out.write(data.tostring())
		js = JointState(name = ["servo_joint"], position=[mean_angle], velocity=[0], effort=[0])
		js.header.stamp = rospy.get_rostime()
		joint_states_pub.publish(js)


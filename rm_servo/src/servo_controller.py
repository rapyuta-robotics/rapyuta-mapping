#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32

import sys
import alsaaudio
import numpy as np
import math


zero_pos = 144
gain = 57.6/(np.pi/4)

ii16 = np.iinfo(np.int16)
data = np.zeros(640, dtype=np.int16)
data[0:zero_pos] = ii16.max

def callback(data):
	if data < -np.pi/4 or data > np.pi/4:
		rospy.loginfo(rospy.get_name() + ": Angle %f is bigger then servo limit" % data)
	val = int(round(zero_pos + data*gain))
	data[:] = 0
	data[0:val] = ii16.max


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
	out.setperiodsize(640)

	# Read data from stdin
	while True:
		out.write(data.tostring())
		rospy.spinOnce()



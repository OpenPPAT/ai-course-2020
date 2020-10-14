#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
import rospkg

from sensor_msgs.msg import Image, LaserScan
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from control.cfg import pos_PIDConfig, ang_PIDConfig
from std_srvs.srv import SetBool, SetBoolResponse
from PID import PID_control
from subt_msgs.msg import BoundingBoxes, BoundingBox


class Tracking():
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Initializing " % (self.node_name))
        
        # Image definition
        self.width = 640
        self.height = 480
        self.h_w = 10.
        self.const_SA = 0.4
        self.predict_prob = 0.5

        self.pos_ctrl_max = 1
        self.pos_ctrl_min = -1
        self.cmd_ctrl_max = 1
        self.cmd_ctrl_min = -1

        rospy.loginfo("[%s] Initializing " % (self.node_name))
        self.image_sub = rospy.Subscriber(
            "BoundingBoxes", BoundingBoxes, self.box_cb, queue_size=1, buff_size=2**24)

        self.pub_cmd = rospy.Publisher(
            'track_vel', Twist, queue_size=1)
        self.cmd_msg = Twist()

        self.pos_control = PID_control("Position_tracking")
        self.ang_control = PID_control("Angular_tracking")

        self.pos_srv = Server(
            pos_PIDConfig, self.pos_pid_cb, "Position_tracking")
        self.ang_srv = Server(
            ang_PIDConfig, self.ang_pid_cb, "Angular_tracking")

        self.initialize_PID()

        self.timer = rospy.Timer(rospy.Duration(0.2), self.cb_publish)

    def cb_publish(self, event):
        self.pub_cmd.publish(self.cmd_msg)

    def box_cb(self, msg):
        goal_angle, goal_distance = self.BBx2AngDis(msg.bounding_boxes[0])
        pos_output, ang_output = self.control(goal_distance, goal_angle)

        self.cmd_msg = Twist()
        self.cmd_msg.linear.x = pos_output
        self.cmd_msg.angular.z = ang_output

    def BBx2AngDis(self, bbox):
        center = [(bbox.xmin+bbox.xmax)/2., (bbox.ymin+bbox.ymax)/2.]
        angle = (center[0]-self.width/2.)/(self.width/2.)
        dis = float(self.height - (bbox.ymax - bbox.ymin))/(self.height)
        return angle, dis

    def control(self, goal_distance, goal_angle):
        self.pos_control.update(5*(goal_distance - self.const_SA))
        self.ang_control.update(goal_angle)

        # pos_output will always be positive
        pos_output = -self.pos_constrain(self.pos_control.output)

        # -1 = -180/180 < output/180 < 180/180 = 1
        ang_output = self.ang_control.output
        return pos_output, ang_output

    def cmd_constarin(self, input):
        if input > self.cmd_ctrl_max:
            return self.cmd_ctrl_max
        if input < self.cmd_ctrl_min:
            return self.cmd_ctrl_min
        return input

    def pos_constrain(self, input):
        if input > self.pos_ctrl_max:
            return self.pos_ctrl_max
        if input < self.pos_ctrl_min:
            return self.pos_ctrl_min
        return input

    def initialize_PID(self):
        self.pos_control.setSampleTime(1)
        self.ang_control.setSampleTime(1)

        self.pos_control.SetPoint = 0.0
        self.ang_control.SetPoint = 0.0

    def get_goal_angle(self, robot_yaw, robot, goal):
        robot_angle = np.degrees(robot_yaw)
        p1 = [robot[0], robot[1]]
        p2 = [robot[0], robot[1]+1.]
        p3 = goal
        angle = self.get_angle(p1, p2, p3)
        result = angle - robot_angle
        result = self.angle_range(-(result + 90.))
        return result

    def get_angle(self, p1, p2, p3):
        v0 = np.array(p2) - np.array(p1)
        v1 = np.array(p3) - np.array(p1)
        angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        return np.degrees(angle)

    # limit the angle to the range of [-180, 180]
    def angle_range(self, angle):
        if angle > 180:
            angle = angle - 360
            angle = self.angle_range(angle)
        elif angle < -180:
            angle = angle + 360
            angle = self.angle_range(angle)
        return angle

    def get_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def pos_pid_cb(self, config, level):
        print(
            "Position: [Kp]: {Kp}   [Ki]: {Ki}   [Kd]: {Kd}\n".format(**config))
        Kp = float("{Kp}".format(**config))
        Ki = float("{Ki}".format(**config))
        Kd = float("{Kd}".format(**config))
        self.pos_control.setKp(Kp)
        self.pos_control.setKi(Ki)
        self.pos_control.setKd(Kd)
        return config

    def ang_pid_cb(self, config, level):
        print(
            "Angular: [Kp]: {Kp}   [Ki]: {Ki}   [Kd]: {Kd}\n".format(**config))
        Kp = float("{Kp}".format(**config))
        Ki = float("{Ki}".format(**config))
        Kd = float("{Kd}".format(**config))
        self.ang_control.setKp(Kp)
        self.ang_control.setKi(Ki)
        self.ang_control.setKd(Kd)
        return config


if __name__ == '__main__':
    rospy.init_node('Tracking')
    foo = Tracking()
    rospy.spin()

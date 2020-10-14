#!/usr/bin/env python

import rospy
from pure_pursuit import PurePursuit
from astar.srv import GoToPos, GoToPosResponse, GoToPosRequest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from nav_msgs.srv import GetPlan, GetPlanRequest, GetPlanResponse
from visualization_msgs.msg import Marker
from subt_msgs.msg import ArtifactPoseArray, ArtifactPose
import tf
import numpy as np


class Navigation(object):
    def __init__(self):
        rospy.loginfo("Initializing %s" % rospy.get_name())

        self.pursuit = PurePursuit()
        self.pursuit.set_look_ahead_distance(0.1)

        self.target_global = None
        self.listener = tf.TransformListener()

        self.pub_goal_marker = rospy.Publisher(
            "goal_marker", Marker, queue_size=1)

        self.pub_target_marker = rospy.Publisher(
            "target_marker", Marker, queue_size=1)

        self.pub_pid_goal = rospy.Publisher(
            "pid_control/goal", PoseStamped, queue_size=1)

        self.req_path_srv = rospy.ServiceProxy("plan_service", GetPlan)

        self.sub_box = rospy.Subscriber(
            "artifact_pose", ArtifactPoseArray, self.cb_pose, queue_size=1)

        self.sub_goal = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.cb_goal, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(0.1), self.tracking)

    def to_marker(self, goal, color=[0, 1, 0]):
        marker = Marker()
        marker.header.frame_id = goal.header.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.pose.orientation.w = 1
        marker.pose.position.x = goal.pose.position.x
        marker.pose.position.y = goal.pose.position.y
        marker.id = 0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        return marker

    def transform_pose(self, pose, target_frame, source_frame):
        try:
            (trans_c, rot_c) = self.listener.lookupTransform(
                target_frame, source_frame, rospy.Time(0))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("faile to catch tf %s 2 %s" %
                         (target_frame, source_frame))
            return

        trans_mat = tf.transformations.translation_matrix(trans_c)
        rot_mat = tf.transformations.quaternion_matrix(rot_c)
        tran_mat = np.dot(trans_mat, rot_mat)

        target_mat = np.array([[1, 0, 0, pose.position.x],
                               [0, 1, 0, pose.position.y],
                               [0, 0, 1, pose.position.z],
                               [0, 0, 0, 1]])
        target = np.dot(tran_mat, target_mat)
        quat = tf.transformations.quaternion_from_matrix(target)
        trans = tf.transformations.translation_from_matrix(target)

        t_pose = PoseStamped()
        t_pose.header.frame_id = target_frame
        t_pose.pose.position.x = trans[0]
        t_pose.pose.position.y = trans[1]
        t_pose.pose.position.z = trans[2]
        t_pose.pose.orientation.x = quat[0]
        t_pose.pose.orientation.y = quat[1]
        t_pose.pose.orientation.z = quat[2]
        t_pose.pose.orientation.w = quat[3]

        return t_pose

    def tracking(self, event):

        if self.target_global is None:
            rospy.logerr("%s : no goal" % rospy.get_name())
            return

        end_p = self.transform_pose(
            self.target_global.pose, "base_footprint", "map")
        self.pub_target_marker.publish(self.to_marker(end_p, [0, 0, 1]))

        start_p = PoseStamped()
        start_p.pose.position.x = 0.3
        start_p.pose.position.y = 0
        start_p.pose.position.z = 0

        req_path = GetPlanRequest()
        req_path.start = start_p
        req_path.goal = end_p

        try:
            res_path = self.req_path_srv(req_path)
        except:
            rospy.logerr("%s : path request fail" % rospy.get_name())
            return

        self.pursuit.set_path(res_path.plan)

        goal = self.pursuit.get_goal(start_p)
        if goal is None:
            rospy.logwarn("goal reached")
            return

        goal = self.transform_pose(goal.pose, "map", "base_footprint")
        self.pub_goal_marker.publish(self.to_marker(goal))

        self.pub_pid_goal.publish(goal)

    def cb_pose(self, msg):
        for i in range(msg.count):
            if msg.pose_array[i].Class == "backpack":
                msg.pose_array[i].pose.position.x -= 0.3
                self.target_global = self.transform_pose(
                    msg.pose_array[i].pose, "map", "camera_link")
                return

    def cb_goal(self, msg):
        self.target_global = msg


if __name__ == "__main__":
    rospy.init_node("navigation")
    nav = Navigation()
    rospy.spin()

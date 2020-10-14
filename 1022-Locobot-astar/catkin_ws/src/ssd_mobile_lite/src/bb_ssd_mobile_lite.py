#!/usr/bin/env python

import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import rospkg
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from subt_msgs.msg import *
import os 
import message_filters

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

class bb_ssd_mobile_lite(object):
	def __init__(self):

		model = "v1"
		r = rospkg.RosPack()
		path = r.get_path('ssd_mobile_lite')
		model_name = "Epoch-630-Loss-0.4744.pth"
		self.prob_threshold = 0.5
		self.cv_bridge = CvBridge() 

		self.labels = ['BACKGROUND', 'backpack']
		if model == "v2_lite":
			self.network = create_mobilenetv2_ssd_lite(len(self.labels), is_test=True) 
		elif model == "v1":
			self.network = create_mobilenetv1_ssd(len(self.labels), is_test=True) 	
		elif model == "v1_lite":
			self.network = create_mobilenetv1_ssd_lite(len(self.labels), is_test=True) 

		state_dict = torch.load(os.path.join(path, "weights/", model_name))
		self.network.load_state_dict(state_dict)
		DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.network.to(DEVICE)
		if model == "v2_lite":
			self.predictor = create_mobilenetv2_ssd_lite_predictor(self.network, candidate_size=200, device = DEVICE)
		elif model == "v1_lite":
			self.predictor = create_mobilenetv1_ssd_lite_predictor(self.network, candidate_size=200, device = DEVICE)
		elif model == "v1":	
			self.predictor = create_mobilenetv1_ssd_predictor(self.network, candidate_size=200, device = DEVICE)

		## Publisher
		self.image_pub = rospy.Publisher("camera/predict_img/", Image, queue_size=1)
		self.BoundingBoxes_pub = rospy.Publisher("BoundingBoxes/", BoundingBoxes, queue_size = 1)

		## msg filter 
		self.depth_sub = message_filters.Subscriber("camera/aligned_depth_to_color/image_raw", Image)
		self.image_sub = message_filters.Subscriber("camera/color/image_raw", Image)
		self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 5, 5)
		self.ts.registerCallback(self.img_cb)

		#self.image_sub = rospy.Subscriber("camera/color/image_raw", Image, self.img_cb, queue_size=1)

		print("Start Predicting image")

	def img_cb(self, rgb_data, depth_data):
		cv_image = self.cv_bridge.imgmsg_to_cv2(rgb_data, "bgr8")
		cv_image = cv2.resize(cv_image, (640,480))
		img = cv_image.copy()
		

		(rows, cols, channels) = cv_image.shape
		self.width = cols
		self.height = rows
                #obj_list = []
		predict_img, obj_list = self.predict(cv_image)
		try:
			self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(predict_img, "bgr8"))
			#self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

		bbox_out = BoundingBoxes()

		for obj in obj_list:
			bbox = BoundingBox()
			out = bb_input()
			mask = np.zeros((rows, cols), dtype = np.uint8)
			point_list = [(int(obj[0]), int(obj[1])),(int(obj[0] + obj[2]),int(obj[1])),\
				(int(obj[0] + obj[2]), int(obj[1] + obj[3])), (int(obj[0]), int(obj[1] + obj[3]))]

			bbox.Class = self.labels[obj[4]]
			bbox.xmin = int(obj[0])
			bbox.xmax = int(obj[0] + obj[2])
			bbox.ymin = int(obj[1])
			bbox.ymax = int(obj[1] + obj[3])
			bbox.probability = obj[5]
			bbox_out.bounding_boxes.append(bbox)

			# cv2.fillConvexPoly(mask, np.asarray(point_list,dtype = np.int), 255)
			# # print point_list
			# out.image = self.cv_bridge.cv2_to_imgmsg(img, "bgr8")
			# out.mask = self.cv_bridge.cv2_to_imgmsg(mask, "8UC1")
			# out.depth = depth_data
			# out.header = rgb_data.header
			# self.mask_pub.publish(out.mask)
			# self.origin.publish(out)

		if len(obj_list) != 0:
			bbox_out.header = rgb_data.header
			bbox_out.depth = depth_data
			bbox_out.count = len(obj_list)
			bbox_out.camera = "camera"
			self.BoundingBoxes_pub.publish(bbox_out)

	def predict(self, img):
		# Preprocessing
		objs = []
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		time = rospy.get_time()
		boxes, labels, probs = self.predictor.predict(image, 10, self.prob_threshold)
		# print(1./(rospy.get_time() - time))
		for i in range(boxes.size(0)):
			box = boxes[i, :]
			check = True
			for j in box:
				if not 0 <= j < 640:
					check = False
			if check:
				if (box[0], box[1]) != (box[2], box[3]):
					cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

					label = "{}: {:.2f}".format(self.labels[labels[i]], probs[i])

					objs.append([box[0], box[1], box[2]-box[0]+1, box[3]-box[1]+1, labels[i], probs[i]])
					cv2.putText(img, label,(box[0] + 20, box[1] + 40),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)


		return img, objs


	def onShutdown(self):
		rospy.loginfo("Shutdown.")


if __name__ == '__main__': 
	rospy.init_node('bb_ssd_mobile_lite',anonymous=False)
	bb_ssd_mobile_lite = bb_ssd_mobile_lite()
	rospy.on_shutdown(bb_ssd_mobile_lite.onShutdown)
	rospy.spin()

#include <ros/ros.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include "subt_msgs/bb_input.h"
#include "subt_msgs/BoundingBoxes.h"
#include "subt_msgs/ArtifactPoseArray.h"
#include "subt_msgs/ArtifactPose.h"

using namespace ros;
using namespace std;
using namespace cv;

class pub_arti_pose_ssd{
  public:
    pub_arti_pose_ssd();
    void callback(const subt_msgs::BoundingBoxes);
    void getXYZ(float* , float* ,float );
  private:
    Publisher pub_pose;
    ros::Subscriber ssd_result;
    float fx;
    float fy;
    float cx;
    float cy;
};

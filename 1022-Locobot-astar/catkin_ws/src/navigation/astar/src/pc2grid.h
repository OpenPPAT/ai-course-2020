#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <string>
#include <math.h>
#include <signal.h>
#include <Eigen/Geometry>
// Ros
#include <ros/ros.h>
#include <ros/timer.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_srvs/Trigger.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


// Pcl load and ros
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
// Pcl icp
#include <pcl/registration/icp.h>
// Pcl downsampling
#include <pcl/filters/voxel_grid.h>

using namespace ros;
using namespace pcl;
using namespace std;

class Pc2Grid{
  public:
	Pc2Grid();
	void pcCallback(const sensor_msgs::PointCloud2ConstPtr &pc);   // sensor_msgs::PointCloud2

  private:
	Publisher pc_map;
	Publisher pub_map;
	Subscriber sub_map;

	PointCloud<PointXYZ>::Ptr depth_cam;
	VoxelGrid<PointXYZ> voxel;
    PassThrough<PointXYZ> passZ;

	sensor_msgs::PointCloud2 ros_cloud_msg;
	tf::StampedTransform transform;
	Eigen::Matrix4f trans;
	
	tf::TransformBroadcaster br;
	tf::TransformListener listener;

};
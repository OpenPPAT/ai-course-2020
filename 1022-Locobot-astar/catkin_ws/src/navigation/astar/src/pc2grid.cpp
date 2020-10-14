#include "pc2grid.h"

Pc2Grid::Pc2Grid()
{
	NodeHandle nh;
	depth_cam.reset(new PointCloud<PointXYZ>());
	pc_map = nh.advertise<sensor_msgs::PointCloud2>("transform_d435_pc", 1);
	pub_map = nh.advertise<nav_msgs::OccupancyGrid>("depthcam_map", 1);

	float leaf_size = 0.1;
	voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
	passZ.setFilterFieldName("z");

	sub_map = nh.subscribe("/camera/depth_registered/points", 1, &Pc2Grid::pcCallback, this);
}

void Pc2Grid::pcCallback(const sensor_msgs::PointCloud2ConstPtr &pc)
{
	fromROSMsg(*pc, *depth_cam);

	passZ.setInputCloud(depth_cam);
	passZ.setFilterLimits(0, 7);
	passZ.filter(*depth_cam);

	voxel.setInputCloud(depth_cam);
	voxel.filter(*depth_cam);

	try
	{
		ros::Duration timeout(1.0);
		listener.waitForTransform("base_footprint", pc->header.frame_id, ros::Time(0), timeout);
		listener.lookupTransform("base_footprint", pc->header.frame_id, ros::Time(0), transform);
	}
	catch (tf::TransformException ex)
	{
		ROS_ERROR("%s", ex.what());
		return;
	}
	pcl_ros::transformAsMatrix(transform, trans);
	pcl::transformPointCloud(*depth_cam, *depth_cam, trans);

	passZ.setInputCloud(depth_cam);
	passZ.setFilterLimits(0.2, 2);
	passZ.filter(*depth_cam);

	toROSMsg(*depth_cam, ros_cloud_msg);
	ros_cloud_msg.header.frame_id = "base_footprint";
	ros_cloud_msg.header.stamp = pc->header.stamp;
	pc_map.publish(ros_cloud_msg);

	// to grid ---------------------------------------------------------------------
	nav_msgs::OccupancyGrid map;
	map.header.frame_id = "base_footprint";
	map.header.stamp = pc->header.stamp;
	map.info.map_load_time = pc->header.stamp;
	map.info.resolution = 0.1;
	map.info.origin.position.x = 0;
	map.info.origin.position.y = -7;
	map.info.origin.position.z = 0;
	map.info.origin.orientation.w = 1;
	map.info.height = 140;
	map.info.width = 70;
	map.data = std::vector<int8_t>(map.info.height * map.info.width, 0);
	for (int i = 0; i < depth_cam->size(); i++)
	{
		int c = int((depth_cam->points[i].x - map.info.origin.position.x) / map.info.resolution);
		int r = int((depth_cam->points[i].y - map.info.origin.position.y) / map.info.resolution);
		r = (r > 139) ? 139 : r;
		c = (c > 69) ? 69 : c;
		if (map.data[r * map.info.width + c] < 100)
		{
			map.data[r * map.info.width + c] += 10;
		}
	}
	pub_map.publish(map);

	// reset ---------------------------------------------------------------------
	depth_cam->points.clear();
}

int main(int argc, char **argv)
{
	init(argc, argv, "pc2grid");
	Pc2Grid global_map;
	spin();
	return 0;
}

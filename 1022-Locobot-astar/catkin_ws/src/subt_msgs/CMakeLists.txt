cmake_minimum_required(VERSION 2.8.12)

project(subt_msgs)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED
  COMPONENTS
    geometry_msgs
    sensor_msgs
    std_msgs
    message_generation
)

add_message_files(
  FILES
  BoundingBox.msg
  BoundingBoxes.msg
  bb_input.msg
  arti_input.msg
  mask_center.msg
  masks.msg
  ArtifactPose.msg
  ArtifactPoseArray.msg
  AnchorBallDetection.msg
  SubTInfo.msg
  GloraPack.msg
)

add_service_files(
  FILES
  int8.srv
  int32.srv
  pause.srv
  node_stat.srv
  start.srv
  stop.srv
  status.srv
  pose.srv
  ultra_info.srv
  frame_switch.srv
  report.srv
  artifact.srv
)

generate_messages(
  DEPENDENCIES
    geometry_msgs
    sensor_msgs
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    sensor_msgs
    message_runtime
    std_msgs
)

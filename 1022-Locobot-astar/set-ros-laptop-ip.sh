#! /bin/bash

if [ ! "$1" ]; then
    echo "Please input laptop wifi IP"
    return 
fi

export ROS_IP=$1
echo "ROS_IP=$ROS_IP"
#! /bin/bash

if [ ! "$1" ]; then
    echo "Please input master IP"
    return 
fi

export ROS_MASTER_URI=http://$1:11311
echo "ROS_MASTER_URI=$ROS_MASTER_URI"
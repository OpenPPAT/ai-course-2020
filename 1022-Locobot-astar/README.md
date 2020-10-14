# ncsist_threat_processing
Using mobile manipulation platform to process threatening items

## download model
```
$ source download_model.sh
```

## make
```
$ source catkin_make.sh
```

## docker run
```
$ source docker_run.sh
```

## docker join
```
$ source docker_join.sh
```

## start tx2
```
$ source start_tx2.sh
```

## launch back pack tracking
```
$ source environment.sh
$ roslaunch astar tracking.launch
```

## move robot
```
tx2	$ locobot
locobot	$ roslaunch locobot_control main.launch use_base:=true use_camera:=true
```

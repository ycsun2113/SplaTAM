#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
IMAGE_TAG=splatam:latest
CONTAINER_NAME=splatam
DATA_DIR=/home/$(whoami)/data/

DOCKER_OPTIONS=""
DOCKER_OPTIONS+="--gpus all "
DOCKER_OPTIONS+="--env NVIDIA_DISABLE_REQUIRE=1 "
DOCKER_OPTIONS+="-it "
DOCKER_OPTIONS+="-e DISPLAY=$DISPLAY "
DOCKER_OPTIONS+="-v /tmp/.X11-unix:/tmp/.X11-unix "
DOCKER_OPTIONS+="-v $SCRIPT_DIR/../:/home/$(whoami)/$(basename $(readlink -f $SCRIPT_DIR/../)) "
# DOCKER_OPTIONS+="-v $HOME/.Xauthority:/home/$(whoami)/.Xauthority "
DOCKER_OPTIONS+="-e XAUTHORITY=/tmp/.Xauthority "
DOCKER_OPTIONS+="-v $HOME/.Xauthority:/tmp/.Xauthority:ro "
# DOCKER_OPTIONS+="-e XAUTHORITY=/tmp/.docker.xauth " 
# DOCKER_OPTIONS+="-v /tmp/.docker.xauth:/tmp/.docker.xauth:ro " 
DOCKER_OPTIONS+="-v $DATA_DIR:/home/$(whoami)/data "
DOCKER_OPTIONS+="-v /etc/group:/etc/group:ro "
DOCKER_OPTIONS+="--name $CONTAINER_NAME "
DOCKER_OPTIONS+="--privileged "
DOCKER_OPTIONS+="--gpus=all "
DOCKER_OPTIONS+="-e NVIDIA_DRIVER_CAPABILITIES=all "
DOCKER_OPTIONS+="--cap-add=SYS_PTRACE "
DOCKER_OPTIONS+="--ipc=host "
DOCKER_OPTIONS+="--network=host "
DOCKER_OPTIONS+="--pid=host "
DOCKER_OPTIONS+="--security-opt seccomp=unconfined "
DOCKER_OPTIONS+="--runtime=nvidia "
DOCKER_OPTIONS+="-e SDL_VIDEODRIVER=x11 "
DOCKER_OPTIONS+="-u $(id -u):$(id -g) "
DOCKER_OPTIONS+="--shm-size 32G "
DOCKER_OPTIONS+="--group-add $(id -g) "
DOCKER_OPTIONS+=" -v /mnt/ws-frb/:/mnt/ws-frb "

for cam in /dev/video*; do 
  DOCKER_OPTIONS+="--device=${cam} "
done


echo $CONTAINER_NAME

if [ ${1:-""} == "restart" ]; then 
  echo "Restarting Container"
  docker rm -f $CONTAINER_NAME
  docker run $DOCKER_OPTIONS $IMAGE_TAG /bin/bash
# https://stackoverflow.com/questions/38576337/how-to-execute-a-bash-command-only-if-a-docker-container-with-a-given-name-does
elif [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then # If container isn't running
    
    # If it exists, but needs to be started
    if [  "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then

          echo "Resuming Container"
          docker start $CONTAINER_NAME
          docker exec -it $CONTAINER_NAME /entrypoint.sh
    else
      echo "Running Container"
      docker run $DOCKER_OPTIONS $IMAGE_TAG
    fi
else
  echo "Attaching to existing container"
  docker exec -it $CONTAINER_NAME /entrypoint.sh
fi
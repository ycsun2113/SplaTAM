#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
IMAGE_TAG=splatam

DOCKER_OPTIONS=""
DOCKER_OPTIONS+="-t $IMAGE_TAG:latest "
DOCKER_OPTIONS+="-f $SCRIPT_DIR/container.Dockerfile "
DOCKER_OPTIONS+="--build-arg USER_ID=$(id -u) --build-arg USER_NAME=$(whoami) "

# Copy requirements.txt into build context (docker/ dir) so COPY can find it
cp "$SCRIPT_DIR/../requirements.txt" "$SCRIPT_DIR/requirements.txt"
trap "rm -f $SCRIPT_DIR/requirements.txt" EXIT

DOCKER_CMD="docker build --no-cache $DOCKER_OPTIONS $SCRIPT_DIR"
echo $DOCKER_CMD
eval $DOCKER_CMD
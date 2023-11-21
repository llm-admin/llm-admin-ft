#!/bin/bash
set -euxo pipefail

DOCKER_REPO="${LLMADMIN_DOCKER_REPO:-docker.io/vincentpli/aviary}"
VERSION="0.0.1"
DOCKER_TAG="$DOCKER_REPO:base-$VERSION"
DOCKER_FILE="${LLMADMIN_DOCKER_FILE:-deploy/ray/Dockerfile-base}"

sudo docker build . -f $DOCKER_FILE -t $DOCKER_TAG

# sudo docker build . -f $DOCKER_FILE -t $DOCKER_TAG -t $DOCKER_REPO:latest
# sudo docker push "$DOCKER_TAG"

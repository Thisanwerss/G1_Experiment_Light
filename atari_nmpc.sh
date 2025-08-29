#!/usr/bin/env bash

# Copyright (c) 2024, The ATARI-NMPC Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# Container and image configuration
IMAGE_NAME="atari-nmpc-image"
CONTAINER_NAME="ATARI_NMPC"
DOCKERFILE_PATH=".devcontainer/Dockerfile"

#==
# Helper functions
#==

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-b] [-e] [-a] [-d] -- Utility to manage ATARI NMPC."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help           Display the help content."
    echo -e "\t-i, --install        Install ATARI NMPC, build docker container and setup dependencies."
    echo -e "\t-b, --build          Build the docker image."
    echo -e "\t-e, --enter          Enter the ATARI NMPC docker container."
    echo -e "\t-a, --attach         Attach shell to existing ATARI NMPC docker container."
    echo -e "\t-d, --delete         Delete all existing ATARI NMPC docker containers and images."
    echo -e "\t-s, --status         Show status of ATARI NMPC containers and images."
    echo -e "\n" >&2
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed. Please install Docker first." >&2
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "[Error] Docker daemon is not running. Please start Docker first." >&2
        exit 1
    fi
}

# Check if NVIDIA Docker runtime is available
check_nvidia_docker() {
    if ! docker info 2>/dev/null | grep -q nvidia; then
        echo "[Warning] NVIDIA Docker runtime not detected. GPU acceleration may not be available."
        echo "         Please install nvidia-docker2 for GPU support."
    fi
}

# Build the Docker image
build_image() {
    echo "Building Docker image '$IMAGE_NAME'..."
    
    # Check if Dockerfile exists
    if [ ! -f "$DOCKERFILE_PATH" ]; then
        echo "[Error] Dockerfile not found at $DOCKERFILE_PATH" >&2
        exit 1
    fi
    
    # Build the image with current user UID/GID
    docker build \
        --build-arg USERNAME=atari \
        --build-arg USER_UID=$(id -u) \
        --build-arg USER_GID=$(id -g) \
        -t "$IMAGE_NAME" \
        -f "$DOCKERFILE_PATH" \
        .
    
    echo "Docker image '$IMAGE_NAME' built successfully!"
}

# Check if image exists
image_exists() {
    docker image inspect "$IMAGE_NAME" >/dev/null 2>&1
}

# Check if container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Check if container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

# Setup X11 forwarding for GUI applications
setup_x11() {
    if [ -n "$DISPLAY" ]; then
        xhost +local:root 2>/dev/null || true
    fi
}

# check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 1
fi

# Check Docker availability
check_docker

# Pass the arguments
while [[ $# -gt 0 ]]; do
    # Read the key
    case "$1" in
        -i|--install)
            echo "Installing ATARI NMPC..."
            
            # Get submodules
            echo "Updating git submodules..."
            git submodule update --init --recursive
            
            # Check NVIDIA Docker support
            check_nvidia_docker
            
            # Build Docker Image if it doesn't exist
            if ! image_exists; then
                build_image
            else
                echo "Docker image '$IMAGE_NAME' already exists. Skipping build."
                echo "Use -b/--build to rebuild the image."
            fi
            
            echo "ATARI NMPC installation completed!"
            echo "Use '$0 -e' to enter the container."
            
            shift
            ;;

        -b|--build)
            echo "Building Docker image..."
            build_image
            shift
            ;;

        -e|--enter)
            # Setup X11 forwarding
            setup_x11
            
            # Check if image exists
            if ! image_exists; then
                echo "[Error] Docker image '$IMAGE_NAME' not found. Run '$0 -i' to install first." >&2
                exit 1
            fi
            
            # Check if a container with the same name exists
            if ! container_exists; then
                # If container doesn't exist, create and start a new container
                echo "Creating and starting new container '$CONTAINER_NAME'..."
                docker run --shm-size=1g -it --privileged \
                    --name "$CONTAINER_NAME" \
                    --net=host \
                    --pid=host \
                    --ipc=host \
                    --runtime=nvidia \
                    --gpus all \
                    --device-cgroup-rule="c *:* rmw" \
                    --env DISPLAY="$DISPLAY" \
                    --env LIBGL_ALWAYS_INDIRECT=0 \
                    --env XAUTHORITY="$XAUTHORITY" \
                    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                    -v /dev:/dev:rw \
                    -v "$PWD":/home/atari/workspace:rw \
                    -v "$HOME/.Xauthority:/home/atari/.Xauthority:rw" \
                    "$IMAGE_NAME" \
                    bash -c "cd ~ && if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; cd workspace; exec bash"
            else
                # If container exists, check if it's running
                if container_running; then
                    echo "Container '$CONTAINER_NAME' is already running. Attaching to it..."
                    docker exec -it "$CONTAINER_NAME" bash -c "cd ~ && if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi; cd workspace; exec bash"
                else
                    echo "Starting existing container '$CONTAINER_NAME'..."
                    docker start -i "$CONTAINER_NAME"
                fi
            fi
            shift
            ;;

        -a|--attach)
            # Attach shell to existing docker container
            if container_running; then
                echo "Attaching to running container '$CONTAINER_NAME'..."
                docker exec -it "$CONTAINER_NAME" /bin/bash
            else
                echo "[Error] Container '$CONTAINER_NAME' is not running." >&2
                echo "Use '$0 -e' to start the container first." >&2
                exit 1
            fi
            shift
            ;;

        -d|--delete)
            echo "Deleting ATARI NMPC containers and images..."
            
            # Stop and remove container if it exists
            if container_exists; then
                if container_running; then
                    echo "Stopping container '$CONTAINER_NAME'..."
                    docker stop "$CONTAINER_NAME"
                fi
                echo "Removing container '$CONTAINER_NAME'..."
                docker rm "$CONTAINER_NAME"
            fi
            
            # Remove image if it exists
            if image_exists; then
                echo "Removing image '$IMAGE_NAME'..."
                docker rmi "$IMAGE_NAME"
            fi
            
            # Clean up unused resources
            echo "Cleaning up unused Docker resources..."
            docker container prune -f
            docker image prune -f
            
            echo "Cleanup completed!"
            shift
            ;;

        -s|--status)
            echo "ATARI NMPC Docker Status:"
            echo "========================="
            
            # Check image status
            if image_exists; then
                echo "✓ Image '$IMAGE_NAME' exists"
                docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}\t{{.Size}}"
            else
                echo "✗ Image '$IMAGE_NAME' not found"
            fi
            
            echo ""
            
            # Check container status
            if container_exists; then
                if container_running; then
                    echo "✓ Container '$CONTAINER_NAME' is running"
                else
                    echo "⚠ Container '$CONTAINER_NAME' exists but not running"
                fi
                docker ps -a --filter "name=^${CONTAINER_NAME}$" --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}"
            else
                echo "✗ Container '$CONTAINER_NAME' not found"
            fi
            
            shift
            ;;

        -h|--help)
            print_help
            exit 0
            ;;
        
        *) # Unknown option
            echo "[Error] Invalid argument provided: $1"
            print_help
            exit 1
            ;;
    esac
done 
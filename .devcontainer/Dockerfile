# Base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set noninteractive environment for apt
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG en_US.UTF-8

# Update and install system-related packages
RUN apt-get update && apt-get install -y \
    locales \
    cmake \
    make \
    nano \
    vim \
    curl \
    git \
    pip \
    software-properties-common \
    build-essential \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libzmq3-dev \
    '^libxcb.*-dev' \
    libx11-dev \
    libx11-xcb-dev \
    libxi-dev \
    libxrender-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libglu1-mesa-dev \
    python3-tk \
    libglfw3-dev \
    kbd \
    libeigen3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Configure locale
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

# Create a non-root user
ARG USERNAME=atari
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USERNAME \
    && mkdir -p /home/$USERNAME/.config \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

# Install sudo and grant sudo permissions
RUN apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER $USERNAME

# User permissions
RUN sudo chown -R $USERNAME:$USERNAME /home/$USERNAME/

# Create workspace directory
RUN mkdir -p /home/$USERNAME/workspace/
WORKDIR /home/$USERNAME/workspace/

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3.sh && \
    bash Miniconda3.sh -b -p /home/$USERNAME/miniconda && \
    rm Miniconda3.sh
ENV PATH="/home/$USERNAME/miniconda/bin:$PATH"

# Initialize conda for bash
RUN /home/$USERNAME/miniconda/bin/conda init bash

# Create and activate the Conda environment
COPY ./environment.yml .

# Add conda-forge channel
RUN /home/$USERNAME/miniconda/bin/conda config --add channels conda-forge
RUN /home/$USERNAME/miniconda/bin/conda install conda-build
RUN bash -c "source /home/$USERNAME/miniconda/etc/profile.d/conda.sh && \
    conda env create -n atari_mpc -f environment.yml python=3.12"
RUN rm environment.yml

# Ensure the environment is always activated
RUN echo "source /home/$USERNAME/miniconda/etc/profile.d/conda.sh && conda activate atari_mpc" >> ~/.bashrc

# Set the working directory
WORKDIR /home/$USERNAME/

# Clone and install acados with Python interface
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --recursive --init && \
    mkdir -p build && cd build && \
    cmake -DACADOS_WITH_QPOASES=ON .. && \
    make install -j$(nproc) && \
    bash -c "source /home/$USERNAME/miniconda/etc/profile.d/conda.sh && conda activate atari_mpc && \
    conda develop ../interfaces/acados_template"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USERNAME/acados/lib"
ENV ACADOS_SOURCE_DIR="/home/$USERNAME/acados"

# Clone and install mj_pin_utils
RUN git clone https://github.com/Atarilab/mj_pin_utils.git && \
    bash -c "source /home/$USERNAME/miniconda/etc/profile.d/conda.sh && conda activate atari_mpc && \
    conda develop ./mj_pin_utils"

# Install ROS 2 (Humble)
ENV ROS_DISTRO=humble

# Install ROS 2 dependencies
RUN sudo apt-get update && sudo apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    && sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && sudo apt-get update && sudo apt-get install -y \
    ros-$ROS_DISTRO-desktop \
    ros-$ROS_DISTRO-ament-cmake \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-colcon-mixin \
    python3-vcstool \
    ros-$ROS_DISTRO-gazebo-ros-pkgs \
    ros-$ROS_DISTRO-xacro \
    ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
    ros-$ROS_DISTRO-rosidl-generator-dds-idl \
    ros-$ROS_DISTRO-joint-state-publisher-gui \
    ros-$ROS_DISTRO-plotjuggler-ros \
    && sudo rosdep init \
    && rosdep update

# Clone and install unitree_ros2
RUN git clone https://github.com/unitreerobotics/unitree_ros2

# Install CycloneDDS
WORKDIR /home/$USERNAME/unitree_ros2/cyclonedds_ws/src
RUN git clone https://github.com/ros2/rmw_cyclonedds -b $ROS_DISTRO && \
    git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
WORKDIR /home/$USERNAME/unitree_ros2/cyclonedds_ws/
RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon build --packages-select cyclonedds"

# Install unitree_sdk2_python
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && \
    cd unitree_sdk2_python && \
    conda develop .

# Install unitree_mujoco
RUN git clone https://github.com/unitreerobotics/unitree_mujoco.git

# Install code formatters and linters
RUN conda install black flake8 isort pre-commit

# Set the final working directory
WORKDIR /home/$USERNAME/workspace/

# Source ROS 2 Environments
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc 


CMD ["bash"]
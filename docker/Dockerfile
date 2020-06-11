# Start with Ubuntu base image
From ubuntu:18.04

MAINTAINER Kai Arulkumaran <design@kaixhin.com>
MAINTAINER Gd. Z <zgd_521@yahoo.com>

ENV DEBIAN_FRONTEND=noninteractive

# Install LXDE, VNC server, XRDP and Firefox
RUN apt update && apt install -y \
    firefox lxde-core lxterminal tightvncserver xrdp wget

# Set user for VNC server (USER is only for build)
ENV USER root
# Set default password
COPY password.txt .
RUN cat password.txt password.txt | vncpasswd && \
  rm password.txt
# Expose VNC port
EXPOSE 5901

# Set up sources and keys
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# Install ROS desktop full with rosinstall
RUN apt update && apt install -y \
    ros-melodic-desktop-full python-rosinstall python-rosdep \
    htop vim git
# Initialise rosdep
RUN rosdep init && rosdep update

# Set XDRP to use TightVNC port
RUN sed -i '0,/port=-1/{s/port=-1/port=5901/}' /etc/xrdp/xrdp.ini

# Copy VNC script that handles restarts
COPY vnc.sh /opt/

# Set up environment
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

CMD ["/opt/vnc.sh"]

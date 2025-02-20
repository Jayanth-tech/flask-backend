#!/bin/bash

# Update the apt package list and install necessary dependencies for OpenCV and video processing
apt-get update

# Install general OpenCV dependencies
apt-get install -y \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libglib2.0-0 \
  libgthread-2.0-0 \
  libfontconfig1 \
  libx11-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libdcmtk14-dev \
  libgstreamer1.0-0 \
  libgstreamer-plugins-base1.0-0 \
  libv4l-dev \
  ffmpeg \
  libeigen3-dev \
  libgtk2.0-dev

# Install Python dependencies (OpenCV and others)
pip install --no-cache-dir opencv-python-headless

# Start the application with Gunicorn and Uvicorn worker
gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app

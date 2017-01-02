#!/bin/bash

#NOTE(TFK): You'll need this if you use GCC.

export PATH=/efs/tfk/tapir/bin:$PATH
export LD_LIBRARY_PATH=/efs/tfk/tapir/lib:$LD_LIBRARY_PATH
export CXX=clang++
export OPENCV_ROOT=/home/armafire/tools/opencv-3-install-test/
export LD_LIBRARY_PATH=$OPENCV_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/efs/home/tfk/archive-linux/lib/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export EXTRA_CFLAGS="-fdetach -fno-exceptions -DOPENCV_FLANN_HPP"

#source /afs/csail.mit.edu/proj/courses/6.172/scripts/.bashrc_silent
##export PATH=/efs/tfk/tapir/bin:$PATH
##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efs/tfk/tapir/lib
#export CXX=g++
#export OPENCV_ROOT=/efs/tools/OpenCV3
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENCV_ROOT/lib
#export EXTRA_CFLAGS=-fcilkplus
#export OMP_NUM_THREADS=1



$@

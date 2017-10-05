#!/bin/bash

#NOTE(TFK): You'll need this if you use GCC.

export TAPIR_PREFIX=/efs/home/neboat/tapir-ex/src/build-release
#export TAPIR_PREFIX=/efs/tapir-new-install/tapir-install

export PATH=$TAPIR_PREFIX/bin:/efs/home/wheatman/install_dir/protobufs/bin:$PATH
export CXX=clang++
#export OPENCV_ROOT=/home/armafire/tools/opencv-3-install-test/
export OPENCV_ROOT=/efs/tools/OpenCV3
#export OPENCV_ROOT=/efs/home/lemon510/cv
export LD_LIBRARY_PATH=$OPENCV_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/efs/home/tfk/archive-linux/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TAPIR_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export EXTRA_CFLAGS="-ftapir -Wall"
export LD_PRELOAD=/efs/tools/jemalloc/lib/libjemalloc.so
#export LD_PRELOAD=/efs/home/tfk/maprecurse/sift_features/test.so
export N_TEMPORARY_BYTES=500000000
#/efs/home/neboat/tapir-ex/src/build-release/bin

# for wheatman stuff.
export PYTHONPATH=$PYTHONPATH:/efs/python_local/lib/python2.7/site-packages
export LD_LIBRARY_PATH=/efs/home/wheatman/install_dir/protobufs/lib:$LD_LIBRARY_PATH

#source /afs/csail.mit.edu/proj/courses/6.172/scripts/.bashrc_silent
##export PATH=/efs/tfk/tapir/bin:$PATH
##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efs/tfk/tapir/lib
#export CXX=g++
#export OPENCV_ROOT=/efs/tools/OpenCV3
##export OPENCV_ROOT=/home/armafire/tools/opencv-3-install-test/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENCV_ROOT/lib
##export LD_LIBRARY_PATH=$OPENCV_ROOT/lib:$LD_LIBRARY_PATH
##export LD_LIBRARY_PATH=/efs/home/tfk/archive-linux/lib/:$LD_LIBRARY_PATH
#export EXTRA_CFLAGS="-fcilkplus"
#export OMP_NUM_THREADS=1
#export LD_PRELOAD=/efs/tools/jemalloc/lib/libjemalloc.so
protoc --cpp_out=. --python_out=. AlignData.proto
$@

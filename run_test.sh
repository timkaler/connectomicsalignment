#!/bin/bash


#<<<<<<< HEAD
#
#export CILK_NWORKERS=8
##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efs/tools/OpenCV3/lib/
#
##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efs/tfk/tapir/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efs/tools/OpenCV3/lib/
##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/armafire/tools/opencv-3-install-test/lib/
#WORKINGDIR=$PWD 
#OUTPUTDIR=$PWD/temp
#
#time ./run_align 1 \
#    11 1 \
#=======
source /afs/csail.mit.edu/proj/courses/6.172/scripts/.bashrc_silent
export OMP_NUM_THREADS=1
export CILK_NWORKERS=16
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efs/home/lemon510/cv/lib
WORKINGDIR=$PWD 
OUTPUTDIR=$PWD/temp

./run_align 1 \
    9 1 \
    $PWD/data/txtspecs.txt \
    $OUTPUTDIR \
    $OUTPUTDIR

#    ./run_align 1 \
#    11 2 \
#    $PWD/txtspecs.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR
#
#    ./run_align 1 \
#    13 1 \
#    $PWD/txtspecs.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR





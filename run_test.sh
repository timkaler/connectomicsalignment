#!/bin/bash

export CILK_NWORKERS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/armafire/tools/opencv-3-install-test/lib/
WORKINGDIR=$PWD 
OUTPUTDIR=$PWD/temp

time ./run_align 1 \
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





#!/bin/bash

export CILK_NWORKERS=16
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efs/tools/OpenCV3/lib/
#WORKINGDIR=$PWD 
OUTPUTDIR=/efs/home/tfk/rh_aligner/test_integration31/
#OUTPUTDIR=$PWD/temp

    ./setup.sh ./run_align 1 \
    9 2 \
    $PWD/data/txtspecs.txt \
    $OUTPUTDIR \
    $OUTPUTDIR && ./setup.sh ./run_align 1 \
    11 2 \
    $PWD/data/txtspecs.txt \
    $OUTPUTDIR \
    $OUTPUTDIR && ./setup.sh ./run_align 1 \
    13 1 \
    $PWD/data/txtspecs.txt \
    $OUTPUTDIR \
    $OUTPUTDIR





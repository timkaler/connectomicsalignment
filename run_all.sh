#!/bin/bash

export CILK_NWORKERS=16
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efs/tools/OpenCV3/lib/
#WORKINGDIR=$PWD
OUTPUTDIR=/efs/home/tfk/rh_aligner/test_iarpa_jan3/
#OUTPUTDIR=$PWD/temp

    ./setup.sh ./run_align 1 \
    0 5 \
    $PWD/data/txtspecs_iarpa.txt \
    $OUTPUTDIR \
    $OUTPUTDIR && ./setup.sh ./run_align 1 \
    5 10 \
    $PWD/data/txtspecs_iarpa.txt \
    $OUTPUTDIR \
    $OUTPUTDIR && ./setup.sh ./run_align 1 \
    15 20 \
    $PWD/data/txtspecs_iarpa.txt \
    $OUTPUTDIR \
    $OUTPUTDIR





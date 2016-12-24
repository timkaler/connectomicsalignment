#!/bin/bash

echo export CILK_NWORKERS=1
export CILK_NWORKERS=16

WORKINGDIR=$PWD 
OUTPUTDIR=$PWD/temp

    ./run_align 1 \
    9 2 \
    $PWD/data/txtspecs.txt \
    $OUTPUTDIR \
    $OUTPUTDIR

    ./run_align 1 \
    11 2 \
    $PWD/data/txtspecs.txt \
    $OUTPUTDIR \
    $OUTPUTDIR

    ./run_align 1 \
    13 1 \
    $PWD/data/txtspecs.txt \
    $OUTPUTDIR \
    $OUTPUTDIR





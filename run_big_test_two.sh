#!/bin/bash



export CILK_NWORKERS=16
NUMANODE=0
SECTION=0
WORKINGDIR=$PWD
OUTPUTDIR=$PWD/temp/

#tee output_file_tfk.out &
#./setup.sh gdb --args ./run_align 1 \
#./setup.sh numactl --cpunodebind=$NUMANODE ./run_align 1 \
#    $SECTION 2 \
#    $PWD/data/3dtest1.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR

./setup.sh numactl --cpunodebind=$NUMANODE ./run_align 1 \
    $SECTION 1 \
    $PWD/data/testing3d.txt \
    $OUTPUTDIR \
    $OUTPUTDIR

#./setup.sh ./run_align 1 \
#    24 1 \
#    $PWD/data/txtspecs_iarpa.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR


#./setup.sh ./run_align 1 \
#    24 1 \
#    $PWD/data/txtspecs_iarpa.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR


#./setup.sh ./run_align 1 \
#    25 1 \
#    $PWD/data/txtspecs_iarpa.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR


#./setup.sh ./run_align 1 \
#    10 2 \
#    $PWD/data/txtspecs.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR 
#&& ./setup.sh ./run_align 1 \
#    11 1 \
#    $PWD/data/txtspecs.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR && ./setup.sh ./run_align 1 \
#    12 1 \
#    $PWD/data/txtspecs.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR && ./setup.sh ./run_align 1 \
#    13 1 \
#    $PWD/data/txtspecs.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR









#
#    ./run_align 1 \
#    13 1 \
#    $PWD/txtspecs.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR

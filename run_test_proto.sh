#!/bin/bash



export CILK_NWORKERS=18
# no longer needed since its added to setup.sh
#export LD_LIBRARY_PATH=/efs/home/wheatman/install_dir/protobufs/lib:$LD_LIBRARY_PATH

NUMANODE=0
SECTION=0
WORKINGDIR=$PWD
OUTPUTDIR=$PWD/temp/

./setup.sh ./run_align 1 \
    0 4\
    $PWD/data/proto_data21 \
    $OUTPUTDIR \
    $OUTPUTDIR


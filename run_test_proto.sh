#!/bin/bash



export CILK_NWORKERS=18
# no longer needed since its added to setup.sh
#export LD_LIBRARY_PATH=/efs/home/wheatman/install_dir/protobufs/lib:$LD_LIBRARY_PATH

NUMANODE=0
SECTION=0
WORKINGDIR=$PWD
OUTPUTDIR=$PWD/temp/

<<<<<<< HEAD
./setup.sh gdb --args ./run_align 1 \
    0 4\
=======
./setup.sh ./run_align 1 \
    0 1\
>>>>>>> d0927bd94baab91f7763fda47842283486ebc1bd
    $PWD/data/proto_data21 \
    $OUTPUTDIR \
    $OUTPUTDIR


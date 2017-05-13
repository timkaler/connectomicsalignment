#!/bin/bash
export CILK_NWORKERS=16
NUMANODE=0
SECTION=0
WORKINGDIR=$PWD
#OUTPUTDIR=$PWD/tests/compare_multiplemfov/
OUTPUTDIR=$PWD/temp/

./setup.sh ./run_align 1 \
    $SECTION 2 \
    $PWD/data/testing3d.txt \
    $OUTPUTDIR \
    $OUTPUTDIR

cd tilespec_comparison
./run_test.sh python compare.py --old ../tests/compare_multiplemfov_gt/ --new ../tests/compare_multiplemfov/
#

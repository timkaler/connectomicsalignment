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

#./setup.sh numactl --cpunodebind=$NUMANODE ./run_align 1 \
#    $SECTION 6 \
#    $PWD/data/9mfov.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR

#./setup.sh ./run_align 1 \
#    $SECTION 7 \
#    $PWD/data/9mfov.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR

#./setup.sh /usr/bin/time -v gdb --args ./run_align 1 \
#./setup.sh /usr/bin/time -v ./run_align 1 \
#    24 2 \
#    $PWD/data/txtspecs_iarpa_full.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR
./setup.sh /usr/bin/time -v gdb --args ./run_align 1 \
    100 100\
    $PWD/data/worm_ts.txt \
    $OUTPUTDIR \
    $OUTPUTDIR

#./setup.sh numactl --cpunodebind=0 ./run_align 1 \
#    0 125\
#    $PWD/data/worm_ts.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR &
#
#./setup.sh numactl --cpunodebind=0 ./run_align 1 \
#    125 125\
#    $PWD/data/worm_ts.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR &
#
#./setup.sh numactl --cpunodebind=1 ./run_align 1 \
#    250 125\
#    $PWD/data/worm_ts.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR &
#
#./setup.sh numactl --cpunodebind=1 ./run_align 1 \
#    375 125\
#    $PWD/data/worm_ts.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR &
#
#./setup.sh numactl --cpunodebind=2 ./run_align 1 \
#    500 125\
#    $PWD/data/worm_ts.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR &
#
#./setup.sh numactl --cpunodebind=2 ./run_align 1 \
#    625 125\
#    $PWD/data/worm_ts.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR &
#
#./setup.sh numactl --cpunodebind=3 ./run_align 1 \
#    750 125\
#    $PWD/data/worm_ts.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR &
#
#./setup.sh numactl --cpunodebind=3 ./run_align 1 \
#    875 120\
#    $PWD/data/worm_ts.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR &

#./setup.sh /usr/bin/time -v ./run_align 1 \
#    500 498\
#    $PWD/data/worm_ts.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR &

#./gen_results.sh
#./setup.sh /usr/bin/time -v ./run_align 1 \
#    25 24\
#    $PWD/data/txtspecs_iarpa_full.txt \
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

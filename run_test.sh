
#!/bin/bash



export CILK_NWORKERS=16
WORKINGDIR=$PWD 
OUTPUTDIR=$PWD/temp

#./setup.sh gdb --args ./run_align 1 \
#./setup.sh ./run_align 1 \
#    9 1 \
#    $PWD/data/txtspecs.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR

./setup.sh ./run_align 1 \
    9 1 \
    $PWD/data/txtspecs.txt \
    $OUTPUTDIR \
    $OUTPUTDIR


# && ./setup.sh ./run_align 1 \
#    10 1 \
#    $PWD/data/txtspecs.txt \
#    $OUTPUTDIR \
#    $OUTPUTDIR && ./setup.sh ./run_align 1 \
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

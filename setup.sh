#!/bin/bash

source /afs/csail.mit.edu/proj/courses/6.172/scripts/.bashrc_silent
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efs/home/lemon510/cv/lib
export OMP_NUM_THREADS=1

$@

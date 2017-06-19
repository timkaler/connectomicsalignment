#!/bin/bash

#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 10000 --to_x 50000 --from_y 10000 --to_y 50000 $PWD/tests/compare_onemfov/W01_Sec001_montaged.json section1 &
#
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 10000 --to_x 50000 --from_y 10000 --to_y 50000 $PWD/tests/compare_onemfov/W01_Sec002_montaged.json section2 &
#


#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 10000 --to_x 30000 --from_y 15000 --to_y 40000 $PWD/temp/W01_Sec001_montaged.json section1 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 10000 --to_x 30000 --from_y 15000 --to_y 40000 $PWD/temp/W01_Sec002_montaged.json section2 &
#1=$1
#2=$2
for j in $@; do
#echo $i
i=`printf "%03d\n" $j`

/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec${i}_montaged.json section$i
done
#

#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 0 --to_x 50000 --from_y 0 --to_y 50000 $PWD/temp/W01_Sec203_montaged.json section203 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 0 --to_x 50000 --from_y 0 --to_y 50000 $PWD/temp/W01_Sec209_montaged.json section209 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec001_montaged.json section1 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec002_montaged.json section2 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec003_montaged.json section3 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec004_montaged.json section4 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec005_montaged.json section5 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec006_montaged.json section6 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec007_montaged.json section7 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec008_montaged.json section8 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec203_montaged.json section203 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec209_montaged.json section209 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec010_montaged.json section10 &
#
#sleep 120
#
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec011_montaged.json section11 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec012_montaged.json section12 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec013_montaged.json section13 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec014_montaged.json section14 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec015_montaged.json section15 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec016_montaged.json section16 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec017_montaged.json section17 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec018_montaged.json section18 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec019_montaged.json section19 &
#/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec020_montaged.json section20 &
#
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec021_montaged.json section21 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec022_montaged.json section22 &
##efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec023_montaged.json section23 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec024_montaged.json section24 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec025_montaged.json section25 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec026_montaged.json section26 &
##
##
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec027_montaged.json section27 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec028_montaged.json section28 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec029_montaged.json section29 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec030_montaged.json section30 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec031_montaged.json section31 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec032_montaged.json section32 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec033_montaged.json section33 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec034_montaged.json section34 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec035_montaged.json section35 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec036_montaged.json section36 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec037_montaged.json section37 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec038_montaged.json section38 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec039_montaged.json section39 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec040_montaged.json section40 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec041_montaged.json section41 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec042_montaged.json section42 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec043_montaged.json section43 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec044_montaged.json section44 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec045_montaged.json section45 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec046_montaged.json section46 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec047_montaged.json section47 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec048_montaged.json section48 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec049_montaged.json section49 &
##/efs/home/tfk/rh_aligner/run_render.sh -s 0.1 --from_x 50000 --to_x 100000 --from_y 50000 --to_y 150000 $PWD/temp/W01_Sec050_montaged.json section50 &
##
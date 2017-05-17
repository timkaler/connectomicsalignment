// Copyright 2014

#include <stdio.h>
#include <cilk/cilk.h>
#include <cilk/reducer_list.h>
#include <cilk/reducer_min.h>
#include <cilk/reducer_max.h>
#include <cilk/holder.h>

#include <vector>
#include <cmath>
#include <list>
#include <map>
#include <string>
#include <set>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "opencv2/opencv.hpp"

typedef struct {
  cv::Point2f center;
  double a11;
  double a01;
  double a10;
  double a00;
  double offset_x;
  double offset_y;
} Hexagon;



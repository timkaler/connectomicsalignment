#include "render.hpp"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cstdio>

#ifndef TFK_DATA
#define TFK_DATA

namespace tfk {

  void sample_stack(Stack* stack, int num_samples, int box_size, std::string filename_prefix);
  bool mesh_overlaps(Stack* stack);
  std::vector<Triangle>* bad_triangles(Section* section);
  void overlay_triangles(Section* section, std::pair<cv::Point2f, cv::Point2f> bbox,
    tfk::Resolution resolution, std::string filename);
  void overlay_triangles_stack(Stack* stack,
    std::pair<cv::Point2f, cv::Point2f> bbox, tfk::Resolution resolution,
    std::string filename_prefix); 
}


#endif

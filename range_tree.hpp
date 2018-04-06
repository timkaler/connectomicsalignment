
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc/types_c.h>
#include <tuple>
#include "triangle.h"
#ifndef TFK_RANGE_TREE
#define TFK_RANGE_TREE

namespace tfk {
  class RangeTree {
    public:
      Triangle* items;
      int size;

      bool leaf;
      std::pair<cv::Point2f, cv::Point2f> bbox;
      std::vector<RangeTree*> children;
      RangeTree(Triangle* items, int size,
                std::pair<cv::Point2f, cv::Point2f> bbox);

      int get_total_item_count();

      bool bbox_contains(std::pair<cv::Point2f, cv::Point2f> bbox, Triangle item);
      bool node_contains(Triangle item);
      Triangle find_triangle(cv::Point2f pt);
      bool point_in_triangle(cv::Point2f pt, Triangle tri);
      float Dot(cv::Point2f, cv::Point2f);
  };
}


#endif

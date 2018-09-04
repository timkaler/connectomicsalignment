
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
#include "cilk_tools/Graph.h"
#include "triangle.h"
#include "range_tree.hpp"

#include "AlignData.pb.h"
//#include "AlignData.pb.cc"

#ifndef TFK_TRIANGLE_MESH
#define TFK_TRIANGLE_MESH

namespace tfk {

  class TriangleMesh {
    public:
      std::pair<cv::Point2f, cv::Point2f> bbox;
      //std::vector<cv::Point2f>* mesh_orig_save;
      //std::vector<cv::Point2f>* mesh_old;

      TriangleMesh(TriangleMeshProto triangleMesh);

      TriangleMesh(double hex_spacing,
                   std::pair<cv::Point2f, cv::Point2f> bbox);
      Triangle find_triangle(cv::Point2f pt);
      Triangle find_triangle_post(cv::Point2f pt);
      void build_index_post();

      std::vector<cv::Point2f>* mesh_orig;
      std::vector<cv::Point2f>* mesh;
      std::vector<std::pair<int,int> >* triangle_edges;
      std::vector<tfkTriangle>* triangles;
      RangeTree* index;
      RangeTree* index_post;
    private:
      void build_index();

      std::vector<cv::Point2f>* generate_hex_grid(double* bounding_box, double spacing);
  };
}


#endif

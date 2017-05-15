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

//#include "common.h"
#include "simple_mutex.h"



#ifndef GRAPH_H_
#define GRAPH_H_

typedef struct {
  int vertex_id;
  int mfov_id; // mfov identifier.
  int tile_index; // index within mfov
  int tile_id; // index in code.

  // Data for ORIGINAL bounding box.
  double start_x;
  double end_x;
  double start_y;
  double end_y;

  double corner_x[4];
  double corner_y[4];

  cv::Point2f corner_points[4];

  cv::Point2f original_center_point;
  cv::Point2f center_point;
  int z;
  double scale;
  double neighbor_grad_x;
  double neighbor_grad_y;
  bool boundary;
  // parameters for affine warp.
  double a00;
  double a01;
  double a10;
  double a11;
  double last_error_value;

  // dynamic data we'll update.
  double offset_x;
  double offset_y;

  double ia00;
  double ia01;
  double ia10;
  double ia11;

  // dynamic data we'll update.
  double ioffset_x;
  double ioffset_y;

  int iteration_count;
  double last_radius_value;
  int converged;
} vdata;

typedef struct {
  int neighbor_id;
  double weight;
  std::vector<cv::Point2f>* v_points;
  std::vector<cv::Point2f>* n_points;
} edata;



template<typename VertexType, typename EdgeType>
class Graph {
 private:
    VertexType* vertexData;
 public:
    Graph();
    int* vertexColors;
    simple_mutex_t* vertexLocks;
    int vertexCount;
    int section_id;
    int num_vertices();
    int compute_trivial_coloring();
    std::vector<std::vector<EdgeType> > edgeData;
    void resize(int size);
    VertexType* getVertexData(int vid);
    void insert_matches(int atile_id, int btile_id,
        std::vector<cv::Point2f>& filtered_match_points_a,
        std::vector<cv::Point2f>& filtered_match_points_b, double weight);
    void insertEdge(int vid, EdgeType edge);
};

#include "./Graph.cpp"
#endif  // GRAPH_H_

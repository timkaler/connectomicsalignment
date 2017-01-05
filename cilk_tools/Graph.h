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

  // Data for ORIGINAL bounding box.
  double start_x;
  double end_x;
  double start_y;
  double end_y;

  // dynamic data we'll update.
  float offset_x;
  float offset_y;
  int iteration_count;
} vdata;

typedef struct {
  int neighbor_id;
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
    int num_vertices();
    int compute_trivial_coloring();
    std::vector<std::vector<EdgeType> > edgeData;
    void resize(int size);
    VertexType* getVertexData(int vid);
    void insert_matches(int atile_id, int btile_id,
        std::vector<cv::Point2f>& filtered_match_points_a,
        std::vector<cv::Point2f>& filtered_match_points_b);
    void insertEdge(int vid, EdgeType edge);
};

#include "./Graph.cpp"
#endif  // GRAPH_H_
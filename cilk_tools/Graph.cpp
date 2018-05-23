#include "Graph.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>
#include "simple_mutex.h"
Graph::Graph() {
  vertexData = NULL;
}

vdata* Graph::getVertexData(int vid){
  return &vertexData[vid];
}

void Graph::insertEdge(int vid, edata edge){
  simple_acquire(&vertexLocks[vid]);
  edgeData[vid].push_back(edge);
  simple_release(&vertexLocks[vid]);
}

int Graph::num_vertices(){
  return vertexCount;
}

void Graph::resize(int size){
  vertexCount = size;
  vertexData = (vdata*) realloc(vertexData, vertexCount * sizeof(vdata));
  edgeData.clear();
  edgeData.resize(size);
  vertexColors = (int*)calloc(vertexCount+1, sizeof(int));
  vertexLocks = (simple_mutex_t*)calloc(vertexCount, sizeof(simple_mutex_t));
}


Graph::~Graph() {
  free(vertexData);
  free(vertexColors);
  free(vertexLocks);
  for (int i = 0; i < edgeData.size(); i++) {
    //for (int j = 0; j < edgeData[i].size(); j++) {
    //  delete edgeData[i][j].v_points;
    //  delete edgeData[i][j].n_points;
    //}
    edgeData[i].clear();
  }
  edgeData.clear();
}


int Graph::compute_trivial_coloring(){
  vertexColors = (int*)calloc(sizeof(int), vertexCount);
  int maxColor = 0;
  for (int i = 0; i < num_vertices(); i++) {
    std::vector<edata> edges = edgeData[i];
    std::set<int> taken_colors;
    for (int j = 0; j < edges.size(); j++) {
      int nid = edges[j].neighbor_id;
      int ncolor = vertexColors[nid];
      taken_colors.insert(ncolor);
    }
    for (int c = 0; c < edges.size()+1; c++) {
      if (taken_colors.find(c) == taken_colors.end()) {
        vertexColors[i] = c;
        if (c > maxColor) maxColor = c;
        break;
      }
    }
  }

  return maxColor+1;
}




void Graph::insert_matches(int atile_id, int btile_id,
    std::vector<cv::Point2f>& filtered_match_points_a,
    std::vector<cv::Point2f>& filtered_match_points_b, double weight){

    std::vector<cv::Point2f>* vedges = new std::vector<cv::Point2f>();
    std::vector<cv::Point2f>* nedges = new std::vector<cv::Point2f>();

    for (int i = 0; i < filtered_match_points_a.size(); i++) {
      vedges->push_back(cv::Point2f(filtered_match_points_a[i]));
    }

    for (int i = 0; i < filtered_match_points_b.size(); i++) {
      nedges->push_back(cv::Point2f(filtered_match_points_b[i]));
    }

    edata edge1;
    edge1.v_points = vedges;
    edge1.n_points = nedges;
    edge1.neighbor_id = btile_id;
    edge1.weight = weight;
    this->insertEdge(atile_id, edge1);

    edata edge2;
    edge2.v_points = nedges;
    edge2.n_points = vedges;
    edge2.neighbor_id = atile_id;
    edge2.weight = weight;
    this->insertEdge(btile_id, edge2);
}


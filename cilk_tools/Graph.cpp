#include "Graph.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>

template<typename VertexType, typename EdgeType>
Graph<VertexType, EdgeType>::Graph() {
  vertexData = NULL;
}

template<typename VertexType, typename EdgeType>
VertexType* Graph< VertexType,  EdgeType>::getVertexData(int vid){
  return &vertexData[vid];
}

template <typename VertexType, typename EdgeType>
void Graph< VertexType,  EdgeType>::insertEdge(int vid, EdgeType edge){
  simple_acquire(&vertexLocks[vid]);
  edgeData[vid].push_back(edge);
  simple_release(&vertexLocks[vid]);
}

template<typename VertexType, typename EdgeType>
int Graph< VertexType,  EdgeType>::num_vertices(){
  return vertexCount;
}

template<typename VertexType, typename EdgeType>
void Graph<VertexType, EdgeType>::resize(int size){
  vertexCount = size;
  vertexData = (VertexType*) realloc(vertexData, vertexCount * sizeof(VertexType));
  edgeData.clear();
  edgeData.resize(size);
  vertexColors = (int*)calloc(vertexCount+1, sizeof(int));
  vertexLocks = (simple_mutex_t*)calloc(vertexCount, sizeof(simple_mutex_t));
}

template<typename VertexType, typename EdgeType>
int Graph< VertexType,  EdgeType>::compute_trivial_coloring(){
  vertexColors = (int*)calloc(sizeof(int), vertexCount);
  int maxColor = 0;
  for (int i = 0; i < num_vertices(); i++) {
    std::vector<EdgeType> edges = edgeData[i];
    std::set<int> taken_colors;
    for (int j = 0; j < edges.size(); j++) {
      int nid = edges[j].neighbor_id;
      int ncolor = vertexColors[nid];
      taken_colors.insert(ncolor);
    }
    for (int c = 0; c < edges.size(); c++) {
      if (taken_colors.find(c) == taken_colors.end()) {
        vertexColors[i] = c;
        if (c > maxColor) maxColor = c;
        break;
      }
    }
  }

  return maxColor+1;
}




template<typename VertexType, typename EdgeType>
void Graph< VertexType,  EdgeType>::insert_matches(int atile_id, int btile_id,
    std::vector<cv::Point2f>& filtered_match_points_a,
    std::vector<cv::Point2f>& filtered_match_points_b){

    std::vector<cv::Point2f>* vedges = new std::vector<cv::Point2f>();
    std::vector<cv::Point2f>* nedges = new std::vector<cv::Point2f>();

    for (int i = 0; i < filtered_match_points_a.size(); i++) {
      vedges->push_back(cv::Point2f(filtered_match_points_a[i]));
    }

    for (int i = 0; i < filtered_match_points_b.size(); i++) {
      nedges->push_back(cv::Point2f(filtered_match_points_b[i]));
    }

    EdgeType edge1;
    edge1.v_points = vedges;
    edge1.n_points = nedges;
    edge1.neighbor_id = btile_id;
    this->insertEdge(atile_id, edge1);

    EdgeType edge2;
    edge2.v_points = nedges;
    edge2.n_points = vedges;
    edge2.neighbor_id = atile_id;
    this->insertEdge(btile_id, edge2);
}

template class Graph<vdata, edata>;

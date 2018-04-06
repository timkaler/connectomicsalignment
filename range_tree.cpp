#include "range_tree.hpp"
#include "cilk_tools/Graph.h"
namespace tfk {


// NOTE(TFK): Recursive rendering should probably solve the issue with big triangles.
float RangeTree::Dot(cv::Point2f a, cv::Point2f b) {
  return a.x*b.x + a.y*b.y;
}

bool RangeTree::point_in_triangle(cv::Point2f p, Triangle tri) {
    cv::Point2f a = tri.points[0];
    cv::Point2f b = tri.points[1];
    cv::Point2f c = tri.points[2];
    float u,v,w;

    cv::Point2f v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = Dot(v0, v0);
    float d01 = Dot(v0, v1);
    float d11 = Dot(v1, v1);
    float d20 = Dot(v2, v0);
    float d21 = Dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;

    if (v>=0 && w >= 0 && u >=0) return true;
    return false;
}

Triangle RangeTree::find_triangle(cv::Point2f pt) {
  if (this->leaf) {
    for (int i = 0; i < size; i++) {
      if (point_in_triangle(pt, items[i])) {
        return items[i];
      }
    }
  } else {
    for (int i = 0; i < children.size(); i++) {
      if (children[i]->bbox.first.x < pt.x &&
          children[i]->bbox.first.y < pt.y &&
          children[i]->bbox.second.x >= pt.x &&
          children[i]->bbox.second.y >= pt.y) {
         return children[i]->find_triangle(pt);
       }
    }
  }

  printf("Tried to perform a point in triangle query that wasn't contained in any triangle!\n");
  exit(1);
}


bool RangeTree::bbox_contains(std::pair<cv::Point2f, cv::Point2f> bbox, Triangle item) {
  for (int i = 0; i < 3; i++) {
    cv::Point2f p = item.points[i];
    if (!(bbox.first.x < p.x &&
        bbox.first.y < p.y &&
        bbox.second.x >= p.x &&
        bbox.second.y >= p.y)) {
      return false;
    }
  }
  return true;
}


bool RangeTree::node_contains(Triangle item) {
  for (int i = 0; i < 3; i++) {
    cv::Point2f p = item.points[i];
    if (!(bbox.first.x < p.x &&
        bbox.first.y < p.y &&
        bbox.second.x >= p.x &&
        bbox.second.y >= p.y)) {
      return false;
    }
  }
  return true;
}

RangeTree::RangeTree(Triangle* items,
                     int _size,
                     std::pair<cv::Point2f, cv::Point2f> bbox) {

  this->bbox = bbox;
  this->size = _size;
  this->items = items;


  if ((std::abs(bbox.second.x-bbox.first.x) <= 250 &&
      std::abs(bbox.second.y-bbox.first.y) <= 250) ||
      size <= 4) {
    leaf = true;

    return;
  } else {
    leaf = false;
  }

  // divide the box into four sections.
  std::vector<std::pair<cv::Point2f, cv::Point2f> > quad_list;

  cv::Point2f corner1 = cv::Point2f(bbox.first.x, bbox.first.y);
  cv::Point2f corner2 = cv::Point2f(bbox.second.x, bbox.first.y);
  cv::Point2f corner3 = cv::Point2f(bbox.first.x, bbox.second.y);
  cv::Point2f corner4 = cv::Point2f(bbox.second.x, bbox.second.y);


  cv::Point2f center = cv::Point2f((bbox.first.x+bbox.second.x)/2,
                                   (bbox.first.y+bbox.second.y)/2);

  auto bb1 = std::make_pair(corner1, center);

  cv::Point2f corner1_2 = cv::Point2f((bbox.first.x+bbox.second.x)/2,
                                      bbox.first.y);
  cv::Point2f corner2_4 = cv::Point2f(bbox.second.x,
                                      (bbox.first.y+bbox.second.y)/2);
  auto bb2 = std::make_pair(corner1_2, corner2_4);

  cv::Point2f corner1_3 = cv::Point2f(bbox.first.x,
                                      (bbox.first.y+bbox.second.y)/2);
  cv::Point2f corner3_4 = cv::Point2f((bbox.first.x+bbox.second.x)/2,
                                      bbox.second.y);

  auto bb3 = std::make_pair(corner1_3, corner3_4);

  auto bb4 = std::make_pair(center, corner4);

  quad_list.push_back(bb1);
  quad_list.push_back(bb2);
  quad_list.push_back(bb3);
  quad_list.push_back(bb4);


  // partition the list
  int iter_count = 0;

  for (int j = 0; j < quad_list.size(); j++) {
    for (int i = iter_count; i < size; i++) {
      if (bbox_contains(quad_list[j], items[i])) {
        auto tmp = items[i];
        items[i] = items[iter_count];
        items[iter_count] = tmp;
        iter_count++;
      }
    }
    children.push_back(new RangeTree(items, iter_count, quad_list[j]));
    items += iter_count; // increment pointer.
    size -= iter_count;
    iter_count = 0;
  }
}

int RangeTree::get_total_item_count() {
  if (this->leaf) {
    return size;
  } else {
    int sum = 0;
    for (int i = 0; i < children.size(); i++) {
      sum += children[i]->get_total_item_count();
    }
    return sum+size;
  }
}


// end namespace tfk
}

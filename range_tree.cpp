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

    float zero = 0.0;
    if (v>=zero && w >= zero && u >= zero) return true;
    return false;
}

Triangle RangeTree::find_triangle(cv::Point2f pt) {
  //printf("Finding triangle bbox %f %f %f %f\n", bbox.first.x, bbox.first.y, bbox.second.x, bbox.second.y);
  if (this->leaf) {
    for (int i = 0; i < size; i++) {
      if (point_in_triangle(pt, items[i])) {
        //printf("returning in leaf\n");
        return items[i];
      }
    }
  } else {

    for (int i = 0; i < children.size(); i++) {
      if (children[i]->bbox.first.x < pt.x&&
          children[i]->bbox.first.y < pt.y&&
          children[i]->bbox.second.x >= pt.x&&
          children[i]->bbox.second.y >= pt.y) {
         Triangle tri = children[i]->find_triangle(pt);
         if (tri.index > -1){
          //printf("returning in parent\n");
          return tri;
         }
       }
    }

    for (int i = 0; i < size; i++) {
      if (point_in_triangle(pt, items[i])) {
        //printf("returning internally\n");
        return items[i];
      }
    }

  }

  //if (this->leaf) printf("this is leaf with size %d\n", size);
  Triangle tri;
  tri.index=-1;
  //printf("returning -1\n");
  return tri; 
  //printf("Tried to perform a point in triangle query that wasn't contained in any triangle! %f %f\n", pt.x, pt.y);
  //exit(1);
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
  this->children.resize(0);
  for (int i = 0; i < size; i++) {
    if (!this->node_contains(this->items[i]) && !this->bbox_contains(bbox, this->items[i])) {
      printf("node doesn't contain item!\n");
      exit(1);
    }
  }
  //cv::Point2f testp = cv::Point2f(16806, 20157);
  //  if (bbox.first.x < testp.x &&
  //      bbox.first.y < testp.y &&
  //      bbox.second.x >= testp.x &&
  //      bbox.second.y >= testp.y) {
  //    printf("In bbox %f %f %f %f\n", bbox.first.x, bbox.first.y, bbox.second.x, bbox.second.y);
  //    bool found_tri = false;
  //    for (int i = 0; i < size; i++) {
  //      if (point_in_triangle(testp, items[i])) {
  //        found_tri = true;
  //        break;
  //      }
  //    }
  //    if (!found_tri) {
  //      printf("did not find tri\n");
  //    } else {
  //      printf("found tri\n");
  //    }
  //  }


  if ((std::abs(bbox.second.x-bbox.first.x) <= 300 &&
      std::abs(bbox.second.y-bbox.first.y) <= 300) ||
      size <= 40) {
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

  std::vector<Triangle*> sub_arrays;
  std::vector<int> sub_arrays_size;

  for (int i = 0; i < quad_list.size()+1; i++) {
    sub_arrays.push_back(new Triangle[size]);
    sub_arrays_size.push_back(0);
  }


  int orig_size = size;
  for (int i = 0; i < size; i++) {
    bool added = false;
    for (int j = 0; j < quad_list.size(); j++) {
      if (bbox_contains(quad_list[j], items[i])) {
        added = true;
        sub_arrays[j][sub_arrays_size[j]++] = items[i];
        break;
      }
    }
    if (!added) {
      sub_arrays[4][sub_arrays_size[4]++] = items[i];
    }
  }

  for (int j = 0; j < quad_list.size(); j++) {
    children.push_back(new RangeTree(sub_arrays[j], sub_arrays_size[j], quad_list[j]));
  }
  this->items = sub_arrays[4];
  this->size = sub_arrays_size[4];

  //// partition the list
  //for (int j = 0; j < quad_list.size(); j++) {

  //  Triangle* in_list = new Triangle[size];
  //  int in_size = 0;
  //  Triangle* out_list = new Triangle[size];
  //  int out_size = 0;
  //  for (int i = 0; i < size; i++) {
  //    if (bbox_contains(quad_list[j], items[i])) {
  //      in_list[in_size++] = items[i];
  //      //auto tmp = items[i];
  //      //items[i] = items[iter_count];
  //      //items[iter_count] = tmp;
  //      //iter_count++;
  //    } else {
  //      out_list[out_size++] = items[i];
  //    }
  //  }
  //  for (int i = 0; i < in_size; i++) {
  //    if (!bbox_contains(quad_list[j],in_list[i])) {
  //      printf("WTF!\n");
  //      exit(1);
  //    }
  //  }
  //  //int iter = 0;
  //  //for (int i = 0; i < in_size; i++) {
  //  //  items[iter++] = in_list[i];
  //  //}
  //  //for (int i = 0; i < out_size; i++) {
  //  //  items[iter++] = out_list[i];
  //  //}
  //  //printf("adding children\n");
  //  children.push_back(new RangeTree(in_list, in_size, quad_list[j]));

  //  items = out_list; // increment pointer.
  //  this->size = out_size;
  //}

  int sum = 0;
  for (int i = 0; i < children.size(); i++) {
    sum += children[i]->get_total_item_count();
  }
  if (orig_size != size + sum) {
    printf("Orig size %d, my size %d, children size is %d\n", orig_size, this->size, sum);
    exit(0);
  }
    //if (bbox.first.x < testp.x &&
    //    bbox.first.y < testp.y &&
    //    bbox.second.x >= testp.x &&
    //    bbox.second.y >= testp.y) {
    //  printf("2In bbox %f %f %f %f\n", bbox.first.x, bbox.first.y, bbox.second.x, bbox.second.y);
    //  bool found_tri = false;
    //  for (int i = 0; i < size; i++) {
    //    if (point_in_triangle(testp, items[i])) {
    //      found_tri = true;
    //      break;
    //    }
    //  }
    //  if (!found_tri) {
    //    printf("2did not find tri\n");
    //    for (int i = 0; i < children.size(); i++) {
    //      Triangle tri = children[i]->find_triangle(testp);
    //      if (tri.index > -1) {
    //      printf("3found triangle in child\n");
    //      break;
    //      }
    //    }
    //  } else {
    //    printf("2found tri\n");
    //  }
    //}

  //size = new_size;
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


std::set<int> RangeTree::get_index_set() {
  std::set<int> index_set;
    for (int i = 0; i < size; i++) {
      index_set.insert(items[i].index);
    }
  if (!this->leaf) {
    for (int i = 0; i < children.size(); i++) {
      std::set<int> child_set = children[i]->get_index_set();
      for (auto iter = child_set.begin(); iter != child_set.end(); ++iter) {
        index_set.insert(*iter);
      }
    }
  }
  return index_set;
}

// end namespace tfk
}

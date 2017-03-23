// Copyright 2016 - Supertech Research Group
//   Tim Kaler, Tao B. Schardl, Haoran Xu, Charles E. Leiserson, Alex Matveev.



////////////////////////////////////////////////////////////////////////////////
// INCLUDES
////////////////////////////////////////////////////////////////////////////////

#include "./match.h"
#include <cilk/cilk.h>
#include <cilk/reducer_list.h>
#include <opencv2/videostab/outlier_rejection.hpp>
#include <mutex>
#include <set>
#include <list>
#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include "./common.h"
#include "./align.h"
#include "./fasttime.h"
#include "./simple_mutex.h"
#include "cilk_tools/engine.h"

static bool ASSUME_PRE_ALIGNED = false;



static cv::Point2f transform_point(vdata* vertex, cv::Point2f point_local) {
  float new_x = point_local.x*vertex->a00 + point_local.y * vertex->a01 + vertex->offset_x + vertex->start_x;
  float new_y = point_local.x*vertex->a10 + point_local.y * vertex->a11 + vertex->offset_y + vertex->start_y;
  //float new_x = point_local.x+vertex->offset_x+vertex->start_x;
  //float new_y = point_local.y+vertex->offset_y+vertex->start_y;
  return cv::Point2f(new_x, new_y);
}

static void match_features(std::vector< cv::DMatch > &matches,
                           cv::Mat &descs1, cv::Mat &descs2,
                           float rod);
static int number_of_no_matches = 0;

std::string matchPadTo(std::string str, const size_t num, const char paddingChar = '0')
{
    if(num > str.size())
        str.insert(0, num - str.size(), paddingChar);
    return str;
}


vdata getAffineTransform(std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
  cv::Mat warp_mat = cv::getAffineTransform(pts1, pts2);
  //std::cout << warp_mat << std::endl;
  vdata tmp;
  tmp.a00 = warp_mat.at<double>(0, 0); 
  tmp.a01 = warp_mat.at<double>(0, 1);
  tmp.offset_x = warp_mat.at<double>(0, 2);
  tmp.a10 = warp_mat.at<double>(1, 0); 
  tmp.a11 = warp_mat.at<double>(1, 1); 
  tmp.offset_y = warp_mat.at<double>(1, 2);
  tmp.start_x = 0.0;
  tmp.start_y = 0.0;
  //printf("tmp values are %f %f %f %f %f %f\n", tmp.a00, tmp.a01, tmp.a10, tmp.a11, tmp.offset_x, tmp.offset_y);

  return tmp;
}

std::pair<double,double> tfk_simple_ransac_strict_ret_affine(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {

  double best_dx = 0.0;
  double best_dy = 0.0;

  double best_a00 = 1.0;
  double best_a01 = 0.0;
  double best_a10 = 0.0;
  double best_a11 = 1.0;

  vdata best_vertex_data;

  int maxInliers = 0;
  int prevMaxInliers = 0;
  double thresh = 1.0;

  int num_iterations = 0;
  std::mt19937 g1 (1);  // mt19937 is a standard mersenne_twister_engine
  std::uniform_int_distribution<int> distribution(0,match_points_a.size()); 
  //for (; thresh <= _thresh || maxInliers < 10; thresh += 1.0) {
  for (thresh = _thresh; num_iterations < 1;) {
    num_iterations++;
    int limit = 100000;
    bool random = false;
    if (match_points_a.size() < limit) {
      limit = match_points_a.size();
      random = true;
    }

    vdata tmp_vertex_data;

    for (int _i = 0; _i < match_points_a.size()*100; _i++) {
      //int i = _i;//distribution(g1);

      int i1 = distribution(g1);
      int i2 = distribution(g1);
      int i3 = distribution(g1);
      std::vector<cv::Point2f> pts1;
      std::vector<cv::Point2f> pts2;
      pts1.push_back(match_points_a[i1]);
      pts1.push_back(match_points_a[i2]);
      pts1.push_back(match_points_a[i3]);
      pts2.push_back(match_points_b[i1]);
      pts2.push_back(match_points_b[i2]);
      pts2.push_back(match_points_b[i3]);
      tmp_vertex_data = getAffineTransform(pts1, pts2);
      vdata tmp = tmp_vertex_data;
    if (std::abs(tmp_vertex_data.a00) > 1.5 || std::abs(tmp_vertex_data.a01) > 0.5 || std::abs(tmp_vertex_data.a10) > 0.5 || std::abs(tmp_vertex_data.a11 > 1.5)) {
      continue; 
    }

      cv::Point2f test_point_a = pts1[0];
      cv::Point2f test_point_b = pts2[0];

      cv::Point2f test_point_a_transformed = transform_point(&tmp_vertex_data, test_point_a);

      if(std::abs(test_point_a_transformed.x - test_point_b.x) > 1e-1) {
      printf("tmp values are %f %f %f %f %f %f\n", tmp.a00, tmp.a01, tmp.a10, tmp.a11, tmp.offset_x, tmp.offset_y);
        printf("There is an error in the transform. point a before %f %f; point a after %f %f; point b %f %f\n", test_point_a.x, test_point_a.y, test_point_a_transformed.x, test_point_a_transformed.y, test_point_b.x, test_point_b.y);
        continue;
      }

      //double dx = match_points_b[i].x - match_points_a[i].x;
      //double dy = match_points_b[i].y - match_points_a[i].y;
      int inliers = 0;
      for (int j = 0; j < match_points_b.size(); j++) {
        cv::Point2f point_a = transform_point(&tmp_vertex_data, match_points_a[j]);      
        cv::Point2f point_b = match_points_b[j];

        //double ndx = match_points_b[j].x - match_points_a[j].x - dx;
        //double ndy = match_points_b[j].y - match_points_a[j].y - dy;
        double ndx = point_b.x - point_a.x;
        double ndy = point_b.y - point_a.y;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          inliers++;
        }
      }
      if (inliers > maxInliers) {
        maxInliers = inliers;
        best_vertex_data = tmp_vertex_data;
        //best_dx = dx;
        //best_dy = dy;
      }
    }
  }
/*
    if (maxInliers > 5000 && num_iterations < 10) {
      thresh = thresh*0.9 - 1.0;
      maxInliers = prevMaxInliers;
    } else {
      prevMaxInliers = maxInliers;
    }
*/
/*
    if (maxInliers > 500) {
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }
      return;
    }*/

   for (int j = 0; j < match_points_b.size(); j++) {
        cv::Point2f point_a = transform_point(&best_vertex_data, match_points_a[j]);      
        cv::Point2f point_b = match_points_b[j];
        double ndx = point_b.x - point_a.x;
        double ndy = point_b.y - point_a.y;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }

    printf("Number of inliers is %d and best_dx %f best_dy %f\n", maxInliers, best_dx, best_dy);
    return std::pair<double,double>(best_dx,best_dy);
}
std::pair<double,double> tfk_simple_ransac_strict_ret(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {

  double best_dx = 0.0;
  double best_dy = 0.0;
  int maxInliers = 0;
  int prevMaxInliers = 0;
  double thresh = 1.0;

  int num_iterations = 0;
  std::mt19937 g1 (1);  // mt19937 is a standard mersenne_twister_engine
  std::uniform_int_distribution<int> distribution(0,match_points_a.size()); 
  //for (; thresh <= _thresh || maxInliers < 10; thresh += 1.0) {
  for (thresh = _thresh; num_iterations < 1;) {
    num_iterations++;
    int limit = 100000;
    bool random = false;
    if (match_points_a.size() < limit) {
      limit = match_points_a.size();
      random = true;
    }
    for (int _i = 0; _i < match_points_a.size(); _i++) {
      int i = _i;//distribution(g1);
      double dx = match_points_b[i].x - match_points_a[i].x;
      double dy = match_points_b[i].y - match_points_a[i].y;
      int inliers1 = 0;
      int inliers2 = 0;
      int inliers3 = 0;
      int inliers4 = 0;
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          if (ndx >= 0) {
            if (ndy >= 0) {
              inliers1++;
            } else {
              inliers2++;
            }
          } else {
            if (ndy >= 0) {
              inliers3++;
            } else {
              inliers4++;
            }

          }
          //inliers++;
        }
      }
      double inliers = inliers1;
      if (inliers2>inliers) {
        inliers=inliers2;
      }
      if (inliers3>inliers) {
        inliers=inliers3;
      }
      if (inliers4>inliers) {
        inliers=inliers4;
      }
      if (inliers > maxInliers) {
        maxInliers = inliers;
        best_dx = dx;
        best_dy = dy;
      }
    }
/*
    if (maxInliers > 5000 && num_iterations < 10) {
      thresh = thresh*0.9 - 1.0;
      maxInliers = prevMaxInliers;
    } else {
      prevMaxInliers = maxInliers;
    }
*/
/*
    if (maxInliers > 500) {
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }
      return;
    }*/

   for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }

    

    printf("Number of inliers is %d and best_dx %f best_dy %f\n", maxInliers, best_dx, best_dy);
    return std::pair<double,double>(best_dx,best_dy);
  }

  int test_count = 0;
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
          test_count++; 
        }
      }
  printf("The test count is %d\n", test_count);
}


void tfk_simple_ransac_strict(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {

  double best_dx = 0.0;
  double best_dy = 0.0;
  int maxInliers = 0;
  int prevMaxInliers = 0;
  double thresh = _thresh;

  int num_iterations = 0;
  std::mt19937 g1 (1);  // mt19937 is a standard mersenne_twister_engine
  std::uniform_int_distribution<int> distribution(0,match_points_a.size()); 
  //for (; thresh <= _thresh; thresh += 1.0) {
  for (thresh = _thresh; num_iterations < 1;) {
    num_iterations++;
    int limit = 100000;
    //if (match_points_a.size() < limit) limit = match_points_a.size();
    for (int _i = 0; _i < match_points_a.size(); _i++) {
      int i = _i;//distribution(g1);
      double dx = match_points_b[i].x - match_points_a[i].x;
      double dy = match_points_b[i].y - match_points_a[i].y;
      int inliers = 0;
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          inliers++;
        }
      }
      if (inliers > maxInliers) {
        maxInliers = inliers;
        best_dx = dx;
        best_dy = dy;
      }
    }
/*
    if (maxInliers > 5000 && num_iterations < 10) {
      thresh = thresh*0.9 - 1.0;
      maxInliers = prevMaxInliers;
    } else {
      prevMaxInliers = maxInliers;
    }
*/
/*
    if (maxInliers > 500) {
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }
      return;
    }*/
  }

  int test_count = 0;
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
          test_count++; 
        }
      }
  printf("The test count is %d\n", test_count);
}


void tfk_simple_ransac(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {

  double best_dx = 0.0;
  double best_dy = 0.0;
  int maxInliers = 0;
  int prevMaxInliers = 0;
  double thresh = 1.0;

  int num_iterations = 0;
  for (; thresh <= _thresh || maxInliers < 4; thresh += 1.0) {
    num_iterations++;
    for (int i = 0; i < match_points_a.size(); i++) {
      double dx = match_points_b[i].x - match_points_a[i].x;
      double dy = match_points_b[i].y - match_points_a[i].y;
      int inliers = 0;
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          inliers++;
        }
      }
      if (inliers > maxInliers) {
        maxInliers = inliers;
        best_dx = dx;
        best_dy = dy;
      }
    }

    if (maxInliers > 5000 && num_iterations < 10) {
      thresh = thresh*0.9 - 1.0;
      maxInliers = prevMaxInliers;
    } else {
      prevMaxInliers = maxInliers;
    }

    if (maxInliers > 500) {
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }
      return;
    }
  }

      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }

}

void concat_two_tiles_all(tile_data_t* a_tile, int atile_id, std::vector< cv::KeyPoint >& atile_kps_in_overlap, std::vector < cv::Mat >& atile_kps_desc_in_overlap_list, std::vector<int>& atile_kps_tile_list) {

  for (size_t pt_idx = 0; pt_idx < a_tile->p_kps_3d->size(); ++pt_idx) {
    cv::Point2f pt = (*a_tile->p_kps_3d)[pt_idx].pt;
    //if (pt.x < ref_tile->x_start + radius) continue;
    //if (pt.y < ref_tile->y_start + radius) continue;
    //if (pt.x > ref_tile->x_end - radius) continue;
    //if (pt.y > ref_tile->y_end - radius) continue;
    atile_kps_in_overlap.push_back((*a_tile->p_kps_3d)[pt_idx]);
    atile_kps_desc_in_overlap_list.push_back(a_tile->p_kps_desc_3d->row(pt_idx).clone());
    atile_kps_tile_list.push_back(atile_id);
  }
}



void concat_two_tiles(tile_data_t* a_tile, int atile_id, std::vector< cv::KeyPoint >& atile_kps_in_overlap, std::vector < cv::Mat >& atile_kps_desc_in_overlap_list, std::vector<int>& atile_kps_tile_list, tile_data_t* ref_tile, double radius) {
  //printf("Comparing the tiles a_tile %d b_tile %d\n", atile_id, btile_id);
  //std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;
  //atile_kps_in_overlap.reserve(a_tile->p_kps->size());
  //btile_kps_in_overlap.reserve(b_tile->p_kps->size());
  //std::vector< cv::Mat > atile_kps_desc_in_overlap_list;
  //atile_kps_desc_in_overlap_list.reserve(a_tile->p_kps->size());
  //std::vector< cv::Mat > btile_kps_desc_in_overlap_list;
  //btile_kps_desc_in_overlap_list.reserve(b_tile->p_kps->size());

  //cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
  // Filter the points in a_tile.

  for (size_t pt_idx = 0; pt_idx < a_tile->p_kps_3d->size(); ++pt_idx) {
    cv::Point2f pt = (*a_tile->p_kps_3d)[pt_idx].pt;
    if (pt.x+a_tile->x_start < ref_tile->x_start + radius) continue;
    if (pt.y+a_tile->y_start < ref_tile->y_start + radius) continue;
    if (pt.x+a_tile->x_start > ref_tile->x_finish - radius) continue;
    if (pt.y+a_tile->y_start > ref_tile->y_finish - radius) continue;
    atile_kps_in_overlap.push_back((*a_tile->p_kps_3d)[pt_idx]);
    atile_kps_desc_in_overlap_list.push_back(a_tile->p_kps_desc_3d->row(pt_idx).clone());
    atile_kps_tile_list.push_back(atile_id);
  }
/*
  for (size_t pt_idx = 0; pt_idx < b_tile->p_kps->size(); ++pt_idx) {
    cv::Point2f pt = (*b_tile->p_kps)[pt_idx].pt;
    btile_kps_in_overlap.push_back((*b_tile->p_kps)[pt_idx]);
    btile_kps_desc_in_overlap_list.push_back(b_tile->p_kps_desc->row(pt_idx).clone());
    btile_kps_tile_list.push_back(btile_id);
  }
*/
  //cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
}




 
int compare_two_tiles(tile_data_t* a_tile, tile_data_t* b_tile, int atile_id, int btile_id, Graph<vdata, edata>* graph) {
  //printf("Comparing the tiles a_tile %d b_tile %d\n", atile_id, btile_id);
  std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;
  atile_kps_in_overlap.reserve(a_tile->p_kps->size());
  btile_kps_in_overlap.reserve(b_tile->p_kps->size());
  std::vector< cv::Mat > atile_kps_desc_in_overlap_list;
  atile_kps_desc_in_overlap_list.reserve(a_tile->p_kps->size());
  std::vector< cv::Mat > btile_kps_desc_in_overlap_list;
  btile_kps_desc_in_overlap_list.reserve(b_tile->p_kps->size());

  cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
  // Filter the points in a_tile.
  for (size_t pt_idx = 0; pt_idx < a_tile->p_kps->size(); ++pt_idx) {
    cv::Point2f pt = (*a_tile->p_kps)[pt_idx].pt;
    atile_kps_in_overlap.push_back((*a_tile->p_kps)[pt_idx]);
    atile_kps_desc_in_overlap_list.push_back(a_tile->p_kps_desc->row(pt_idx).clone());
  }
  cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));

  // Filter the points in b_tile.
  for (size_t pt_idx = 0; pt_idx < b_tile->p_kps->size(); ++pt_idx) {
    cv::Point2f pt = (*b_tile->p_kps)[pt_idx].pt;
      btile_kps_in_overlap.push_back((*b_tile->p_kps)[pt_idx]);
      btile_kps_desc_in_overlap_list.push_back(b_tile->p_kps_desc->row(pt_idx).clone());
  }
  cv::vconcat(btile_kps_desc_in_overlap_list,
      (btile_kps_desc_in_overlap));


        if (atile_kps_in_overlap.size() < MIN_FEATURES_NUM) {
          printf("Not enough features.\n");
          return 0;
        }
        if (btile_kps_in_overlap.size() < MIN_FEATURES_NUM) {
          printf("Not enough features.\n");
          return 0;
        }

        // Match the features
        std::vector< cv::DMatch > matches;
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       0.95);
         

        // Filter the matches with RANSAC
        std::vector<cv::Point2f> match_points_a, match_points_b;
        for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
          match_points_a.push_back(
              atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
          match_points_b.push_back(
              btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
        }
        //void tfk_simple_ransac(std::vector<cv::Point2f>& match_points_a,
        //    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {
        bool* mask = (bool*)calloc(matches.size()+1, 1);
        if (match_points_a.size() >= 12) {
          tfk_simple_ransac_strict(match_points_a, match_points_b, 25.0, mask);
        } 
        std::vector< cv::Point2f > filtered_match_points_a(0);
        std::vector< cv::Point2f > filtered_match_points_b(0);
        int num_matches_filtered = 0;
        // Use the output mask to filter the matches

        for (size_t i = 0; i < matches.size(); ++i) {
          if (mask[i]) {
            num_matches_filtered++;
            filtered_match_points_a.push_back(
                atile_kps_in_overlap[matches[i].queryIdx].pt);
            filtered_match_points_b.push_back(
                btile_kps_in_overlap[matches[i].trainIdx].pt);
          }
        }
        free(mask);
        if (num_matches_filtered > match_points_a.size()*0.2 && num_matches_filtered > 4) {
          graph->insert_matches(atile_id, btile_id,
              filtered_match_points_a, filtered_match_points_b, 1.0);
          printf("Number of matches filtered are %d\n", num_matches_filtered);
        }
        return num_matches_filtered;
}


void updateVertex2DAlign(int vid, void* scheduler_void);

//#include "cilk_tools/scheduler.cpp"
//#include "cilk_tools/engine.cpp"

#ifndef SKIPJSON
#define SAVE_TILE_MATCHES(...) save_tile_matches(__VA_ARGS__)
#else
//#define SAVE_TILE_MATCHES(...)
#endif
int global_lock = 0;
void updateVertex2DAlign(int vid, void* scheduler_void) {

  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
  Graph<vdata, edata>* graph = reinterpret_cast<Graph<vdata,edata>*>(scheduler->graph_void);

  //printf("starting vertex %d\n", vid);
  vdata* vertex_data = graph->getVertexData(vid);
  std::vector<edata> edges = graph->edgeData[vid];


  double original_offset_x = vertex_data->offset_x;
  double original_offset_y = vertex_data->offset_y;

  //std::vector<cv::Point2d> filtered_match_points_a(0);
  //std::vector<cv::Point2d> filtered_match_points_b(0);
  //printf("current dx %f, current dy %f\n", current_dx, current_dy);
  // for each neighboring tile.
  //printf("edges size %d\n", edges.size());

  std::vector<cv::Point2f> source_points(0), dest_points(0);
  //std::vector<std::vector<cv::Point2f> > neighbor_vector_source;
  //std::vector<std::vector<cv::Point2f> > neighbor_vector_dest;

  if (edges.size() == 0) return;


    std::vector<cv::Point2f> original_points;
    std::vector<cv::Point2f> original_points2;
  std::vector<float> weights;
  for (int i = 0; i < edges.size(); i++) {
    //printf("in the edges loop doing %d of %d\n", i,edges.size());
    std::vector<cv::Point2f>* v_points = edges[i].v_points;
    std::vector<cv::Point2f>* n_points = edges[i].n_points;
    //if (graph->getVertexData(edges[i].neighbor_id)->z < graph->getVertexData(vid)->z) continue; // skip 
    // NOTE(TFK): Below might be needed.
    //neighbor_vector_source.push_back(std::vector<cv::Point2f>());
    //neighbor_vector_dest.push_back(std::vector<cv::Point2f>());

    //std::vector<cv::Point2f> source_points(v_points->size()), dest_points(n_points->size());

  //double current_dx = vertex_data->offset_x + vertex_data->start_x;
  //double current_dy = vertex_data->offset_y + vertex_data->start_y;

    vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
    //double neighbor_dx = neighbor_vertex->offset_x + neighbor_vertex->start_x;
    //double neighbor_dy = neighbor_vertex->offset_y + neighbor_vertex->start_y;
    double curr_weight = 1.0;//edges[i].weight;
    if (graph->getVertexData(edges[i].neighbor_id)->z != graph->getVertexData(vid)->z) {
      curr_weight = 3.0/40.0;
    }
    //printf("points size %d\n", v_points->size());
    for (int j = 0; j < v_points->size(); j++) {
      //cv::Point2d ptx1((*v_points)[j].x + current_dx,
      //    (*v_points)[j].y + current_dy);
      cv::Point2f ptx1 = transform_point(vertex_data, (*v_points)[j]);
      cv::Point2f ptx2 = transform_point(neighbor_vertex, (*n_points)[j]);

      //cv::Point2d ptx2((*n_points)[j].x + neighbor_dx,
      //    (*n_points)[j].y + neighbor_dy);
      
      source_points.push_back(ptx1);
      dest_points.push_back(ptx2);
      original_points.push_back((*v_points)[j]);
      original_points2.push_back((*n_points)[j]);
      weights.push_back(curr_weight);
      //neighbor_vector_source[i].push_back(ptx1);
      //neighbor_vector_dest[i].push_back(ptx2);
    }
}
  
  std::vector<cv::Point2f>& filtered_match_points_a = source_points;
  std::vector<cv::Point2f>& filtered_match_points_b = dest_points;

   


    double best_dx = 0.0;
    double best_dy = 0.0;
    double failures = 9.0;//vertex_data->last_radius_value;
    double maxMinInliers = 0.0;

    simple_mutex_t maxMinInliers_lock;
    simple_mutex_init(&maxMinInliers_lock);

    uint64_t rng_a = 16807;
    uint64_t rng_m = 2147483647;
    uint64_t rng_seed = vid + 17103*vertex_data->iteration_count;

    //while (maxMinInliers < 4.0 && failures < 1000) {
    //  failures += 1.0;
    //  cilk_for (int trycount = 0; trycount < source_points.size(); trycount++) {
    //    //if (trycount > 100) break;
    //    //rng_seed = (rng_a*rng_seed)%rng_m;
    //    //int index = (rng_seed%source_points.size());
    //    int index = trycount;
    //    cv::Point2d spoint = source_points[index];
    //    cv::Point2d dpoint = dest_points[index];
    //    double dx = dpoint.x-spoint.x;
    //    double dy = dpoint.y-spoint.y;

    //    double minInliers = 1000000;
    //    for (int neigh = 0; neigh < neighbor_vector_source.size(); neigh++) {
    //      int numinliers = 0;
    //      if (neighbor_vector_source[neigh].size() < 4) continue;
    //      for (int z = 0; z < neighbor_vector_source[neigh].size(); z++) {
    //        double newdx = neighbor_vector_source[neigh][z].x+dx - neighbor_vector_dest[neigh][z].x;
    //        double newdy = neighbor_vector_source[neigh][z].y+dy - neighbor_vector_dest[neigh][z].y;
    //        if (newdx*newdx + newdy*newdy < (1.0 + failures+90000)*(1.0+failures+90000)) {
    //          numinliers++;
    //          /*cv::Point2f v_tmp = (*(edges[neigh].v_points))[z];
    //          cv::Point2f n_tmp = (*(edges[neigh].n_points))[z];
    //          for (int hole = z-1; hole >=0; --hole) {
    //            (*(edges[neigh].v_points))[hole+1] = (*(edges[neigh].v_points))[hole];
    //            (*(edges[neigh].n_points))[hole+1] = (*(edges[neigh].n_points))[hole];
    //          }
    //          (*(edges[neigh].v_points))[0] = v_tmp;
    //          (*(edges[neigh].n_points))[0] = n_tmp;*/
    //        }
    //      }
    //      if (numinliers < minInliers) {
    //        minInliers = numinliers;
    //      }
    //    }

    //    if (minInliers > maxMinInliers) {
    //      simple_acquire(&maxMinInliers_lock);
    //      if (minInliers > maxMinInliers) {
    //        maxMinInliers = minInliers;
    //        best_dx = dx;
    //        best_dy = dy;
    //      }
    //      simple_release(&maxMinInliers_lock);
    //    }
    //  }
    //}

    /*if (maxMinInliers > 0 || true) {
      vertex_data->last_radius_value = failures - 2.0;
        if (vertex_data->last_radius_value < 0.0) {
          vertex_data->last_radius_value = 0.0;
        }
      for (int neigh = 0; neigh < neighbor_vector_source.size(); neigh++) {
        int numinliers = 0;
        for (int z = 0; z < neighbor_vector_source[neigh].size(); z++) {
          //double newdx = neighbor_vector_source[neigh][z].x+best_dx - neighbor_vector_dest[neigh][z].x;
          //double newdy = neighbor_vector_source[neigh][z].y+best_dy - neighbor_vector_dest[neigh][z].y;
          //if (true || newdx*newdx + newdy*newdy < (1.0 + failures+90000)*(1.0+failures+90000)) {
          if (true) {
            filtered_match_points_a.push_back(neighbor_vector_source[neigh][z]);
            filtered_match_points_b.push_back(neighbor_vector_dest[neigh][z]);
          }
        }
      }

    } else {
      vertex_data->last_radius_value = failures;
    }*/


    std::vector<cv::Point2d> match_points_a_fixed(0);
    if (filtered_match_points_a.size() > 0) {
          double grad_error_x = 0.0;
          double grad_error_y = 0.0;
          double zgrad_error_x = 0.0;
          double zgrad_error_y = 0.0;
          double grad_error_a00 = 0.0;
          double grad_error_a01 = 0.0;
          double grad_error_a10 = 0.0;
          double grad_error_a11 = 0.0;
          double zgrad_error_a00 = 0.0;
          double zgrad_error_a01 = 0.0;
          double zgrad_error_a10 = 0.0;
          double zgrad_error_a11 = 0.0;

          double weight_sum = 1.0;
          double zweight_sum = 1.0;
          for (int iter = 0; iter < filtered_match_points_a.size(); iter++) {
             double delta_x = filtered_match_points_b[iter].x - filtered_match_points_a[iter].x;
             double delta_y = filtered_match_points_b[iter].y - filtered_match_points_a[iter].y;


             double local_x = original_points[iter].x/4000;
             double local_y = original_points[iter].y/4000;

             //grad_error_x += 2*(delta_x)*weights[iter];
             //grad_error_y += 2*(delta_y)*weights[iter];
             if (weights[iter] > 0.5) {
               grad_error_x += 2*(delta_x);
               grad_error_y += 2*(delta_y);
	       //grad_error_a00 += 1e-1*2*(local_x)*(delta_x/4000);// - 30000*(1-vertex_data->a00));
               //grad_error_a11 += 1e-1*2*local_y*(delta_y/4000);// - 30000*(1-vertex_data->a11));
               weight_sum += 1.0;
             } else {
               zgrad_error_x += 2*(delta_x);
               zgrad_error_y += 2*(delta_y);
	       //zgrad_error_a00 += 1e-1*2*(local_x)*(delta_x/4000);// - 30000*(1-vertex_data->a00));
               //zgrad_error_a11 += 1e-1*2*local_y*(delta_y/4000);// - 30000*(1-vertex_data->a11));
               zweight_sum += 1.0;
             }
             //grad_error_a01 += 1e-2*2*(local_y)*(delta_x/4000);// + 30000*vertex_data->a01);

             //grad_error_a10 += 1e-2*2*local_x*(delta_y/4000);// + 30000*vertex_data->a10);
              
          }
          //double max_ratio = grad_error_x > grad_error_y ? 5.0/grad_error_x : 5.0/grad_error_y;
          //if (max_ratio < 1.0) {
          //  vertex_data->offset_x += max_ratio* grad_error_x*0.5/(filtered_match_points_a.size());
          //  vertex_data->offset_y += max_ratio* grad_error_y*0.5/(filtered_match_points_a.size());
          //} else {
            //vertex_data->offset_x += grad_error_x*0.49/(filtered_match_points_a.size());
            //vertex_data->offset_y += grad_error_y*0.49/(filtered_match_points_a.size());
            if (vid==1) {
              //printf("grad error a00 %f, a00 value is %f\n", grad_error_a00, vertex_data->a00);
            }

            //double delta = 1e-4; /// vertex_data->iteration_count;
            //double original_error = 0.0;
            //bool force_change = false;
            //for (int k = 0; k < filtered_match_points_b.size(); k++) {
            //  cv::Point2f modified_point = transform_point(vertex_data, original_points[k]);
            //  cv::Point2f neighbor_point = filtered_match_points_b[k];
            //  original_error += (modified_point.x-neighbor_point.x)*(modified_point.x-neighbor_point.x) + 
            //           (modified_point.y-neighbor_point.y)*(modified_point.y-neighbor_point.y) + (1-vertex_data->a00)*(1-vertex_data->a00)*100 + (1-vertex_data->a11)*(1-vertex_data->a11)*100;
            //}

            //{
            //vertex_data->a00 += delta;
            //double error_up = 0.0;
            //for (int k = 0; k < filtered_match_points_b.size(); k++) {
            //  cv::Point2f modified_point = transform_point(vertex_data, original_points[k]);
            //  cv::Point2f neighbor_point = filtered_match_points_b[k];
            //  error_up += (modified_point.x-neighbor_point.x)*(modified_point.x-neighbor_point.x) + 
            //           (modified_point.y-neighbor_point.y)*(modified_point.y-neighbor_point.y) + (1-vertex_data->a00)*(1-vertex_data->a00)*100 +(1-vertex_data->a11)*(1-vertex_data->a11)*100;
            //}
            //vertex_data->a00 -= delta;

            //vertex_data->a00 -= delta;
            //double error_down = 0.0;
            //for (int k = 0; k < filtered_match_points_b.size(); k++) {
            //  cv::Point2f modified_point = transform_point(vertex_data, original_points[k]);
            //  cv::Point2f neighbor_point = filtered_match_points_b[k];
            //  error_down += (modified_point.x-neighbor_point.x)*(modified_point.x-neighbor_point.x) + 
            //           (modified_point.y-neighbor_point.y)*(modified_point.y-neighbor_point.y)+ (1-vertex_data->a00)*(1-vertex_data->a00)*100 + (1-vertex_data->a11)*(1-vertex_data->a11)*100;

            //}
            //vertex_data->a00 += delta;

            //if (error_up < error_down) {
            //  if (error_up < original_error) {
            //    vertex_data->a00 += delta;
            //    if (std::abs(error_up-original_error) > 10.0) {
            //      force_change = true; 
            //    }
            //    if (vid == 1) {
            //      //printf("Error up is %f, original is %f\n", error_up, original_error);
            //    }
            //  }
            //} else {
            //  if (error_down < original_error) {
            //    vertex_data->a00 -= delta;
            //    if (std::abs(error_down-original_error) > 10.0) {
            //      force_change = true; 
            //    }
            //    if (vid == 1) {
            //      //printf("Error down is %f, original is %f\n", error_down, original_error);
            //    }
            //  }
            //}
            //}

           //// {
           //// vertex_data->a01 += delta;
           //// double error_up = 0.0;
           //// for (int k = 0; k < filtered_match_points_b.size(); k++) {
           ////   cv::Point2f modified_point = transform_point(vertex_data, original_points[k]);
           ////   cv::Point2f neighbor_point = filtered_match_points_b[k];
           ////   error_up += (modified_point.x-neighbor_point.x)*(modified_point.x-neighbor_point.x) + 
           ////            (modified_point.y-neighbor_point.y)*(modified_point.y-neighbor_point.y);
           //// }
           //// vertex_data->a01 -= delta;

           //// vertex_data->a01 -= delta;
           //// double error_down = 0.0;
           //// for (int k = 0; k < filtered_match_points_b.size(); k++) {
           ////   cv::Point2f modified_point = transform_point(vertex_data, original_points[k]);
           ////   cv::Point2f neighbor_point = filtered_match_points_b[k];
           ////   error_down += (modified_point.x-neighbor_point.x)*(modified_point.x-neighbor_point.x) + 
           ////            (modified_point.y-neighbor_point.y)*(modified_point.y-neighbor_point.y);
           //// }
           //// vertex_data->a01 += delta;

           //// if (error_up < error_down) {
           ////   if (error_up < original_error) {
           ////     vertex_data->a01 += delta;
           ////   }
           //// } else {
           ////   if (error_down < original_error) {
           ////     vertex_data->a01 -= delta;
           ////   }
           //// }
           //// }


           //// {
           //// vertex_data->a10 += delta;
           //// double error_up = 0.0;
           //// for (int k = 0; k < filtered_match_points_b.size(); k++) {
           ////   cv::Point2f modified_point = transform_point(vertex_data, original_points[k]);
           ////   cv::Point2f neighbor_point = filtered_match_points_b[k];
           ////   error_up += (modified_point.x-neighbor_point.x)*(modified_point.x-neighbor_point.x) + 
           ////            (modified_point.y-neighbor_point.y)*(modified_point.y-neighbor_point.y);
           //// }
           //// vertex_data->a10 -= delta;

           //// vertex_data->a10 -= delta;
           //// double error_down = 0.0;
           //// for (int k = 0; k < filtered_match_points_b.size(); k++) {
           ////   cv::Point2f modified_point = transform_point(vertex_data, original_points[k]);
           ////   cv::Point2f neighbor_point = filtered_match_points_b[k];
           ////   error_down += (modified_point.x-neighbor_point.x)*(modified_point.x-neighbor_point.x) + 
           ////            (modified_point.y-neighbor_point.y)*(modified_point.y-neighbor_point.y);
           //// }
           //// vertex_data->a10 += delta;

           //// if (error_up < error_down) {
           ////   if (error_up < original_error) {
           ////     vertex_data->a10 += delta;
           ////   }
           //// } else {
           ////   if (error_down < original_error) {
           ////     vertex_data->a10 -= delta;
           ////   }
           //// }
           //// }

            //{
            //vertex_data->a11 += delta;
            //double error_up = 0.0;
            //for (int k = 0; k < filtered_match_points_b.size(); k++) {
            //  cv::Point2f modified_point = transform_point(vertex_data, original_points[k]);
            //  cv::Point2f neighbor_point = filtered_match_points_b[k];
            //  error_up += (modified_point.x-neighbor_point.x)*(modified_point.x-neighbor_point.x) + 
            //           (modified_point.y-neighbor_point.y)*(modified_point.y-neighbor_point.y)+ (1-vertex_data->a11)*(1-vertex_data->a11)*100 + (1-vertex_data->a00)*(1-vertex_data->a00)*100;

            //}
            //vertex_data->a11 -= delta;

            //vertex_data->a11 -= delta;
            //double error_down = 0.0;
            //for (int k = 0; k < filtered_match_points_b.size(); k++) {
            //  cv::Point2f modified_point = transform_point(vertex_data, original_points[k]);
            //  cv::Point2f neighbor_point = filtered_match_points_b[k];
            //  error_down += (modified_point.x-neighbor_point.x)*(modified_point.x-neighbor_point.x) + 
            //           (modified_point.y-neighbor_point.y)*(modified_point.y-neighbor_point.y)+ (1-vertex_data->a11)*(1-vertex_data->a11)*100 + (1-vertex_data->a00)*(1-vertex_data->a00)*100;

            //}
            //vertex_data->a11 += delta;

            //if (error_up < error_down) {
            //  if (error_up < original_error) {
            //    vertex_data->a11 += delta;
            //    if (std::abs(error_up-original_error) > 0) {
            //      force_change = true; 
            //    }
            //  }
            //} else {
            //  if (error_down < original_error) {
            //    vertex_data->a11 -= delta;
            //    if (std::abs(error_down-original_error) > 0) {
            //      force_change = true; 
            //    }
            //  }
            //}
            //}

            vertex_data->offset_x += (1.0-2*4.0/30.0)*grad_error_x*0.49/(weight_sum+zweight_sum);
            vertex_data->offset_y += (1.0-2*4.0/30.0)*grad_error_y*0.49/(weight_sum+zweight_sum);
            vertex_data->offset_x += (2*4.0/30.0)*zgrad_error_x*0.49/(zweight_sum+weight_sum);
            vertex_data->offset_y += (2*4.0/30.0)*zgrad_error_y*0.49/(zweight_sum+weight_sum);



            //if (grad_error_a00 > 0.1) {
            //  grad_error_a00 = 0.1; 
            //}
            //if (grad_error_a00 < -0.1) {
            //  grad_error_a00 = -0.1; 
            //}
            //if (grad_error_a01 > 0.1) {
            //  grad_error_a01 = 0.1; 
            //}
            //if (grad_error_a01 < -0.1) {
            //  grad_error_a01 = -0.1; 
            //}

            //if (grad_error_a10 > 0.1) {
            //  grad_error_a10 = 0.1; 
            //}
            //if (grad_error_a10 < -0.1) {
            //  grad_error_a10 = -0.1; 
            //}

            //if (grad_error_a11 > 0.1) {
            //  grad_error_a11 = 0.1; 
            //}
            //if (grad_error_a11 < -0.1) {
            //  grad_error_a11 = -0.1; 
            //}



            //vertex_data->a00 += (1.0-4.0/30.0)*0.49*grad_error_a00/(zweight_sum+weight_sum);//*0.00149 / (weight_sum*grad_sum);
            //vertex_data->a11 += (1.0-4.0/30.0)*0.49*grad_error_a11/(zweight_sum+weight_sum);//*0.00149 / (weight_sum*grad_sum);
            //vertex_data->a00 += (4.0/30.0)*0.49*zgrad_error_a00/(weight_sum+zweight_sum);//*0.00149 / (weight_sum*grad_sum);
            //vertex_data->a11 += (4.0/30.0)*0.49*zgrad_error_a11/(weight_sum+zweight_sum);//*0.00149 / (weight_sum*grad_sum);
            
          //}
          


  if ( vertex_data->iteration_count < 20000000 && (
      std::abs(vertex_data->offset_x - original_offset_x /*x grad_error_x/weight_sum*/) +
      std::abs(vertex_data->offset_y - original_offset_y)/*grad_error_y/weight_sum)*/ + 10*(grad_error_a00 + grad_error_a01 + grad_error_a10 + grad_error_a11) > 1e-2)) {
    scheduler->add_task(vid, updateVertex2DAlign);
    //if (std::abs(original_offset_x - vertex_data->offset_x) +
    //  std::abs(original_offset_y - vertex_data->offset_y) > 1e-2) {
      for (int i = 0; i < edges.size(); i++) {
        scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
      }
    //}
  }

  //if ( vertex_data->iteration_count < 20000000 &&
  //    std::abs(original_offset_x - vertex_data->offset_x) +
  //    std::abs(original_offset_y - vertex_data->offset_y) > 1e-2) {

  //  scheduler->add_task(vid, updateVertex2DAlign);
  //  if (std::abs(original_offset_x - vertex_data->offset_x) +
  //    std::abs(original_offset_y - vertex_data->offset_y) > 1e-2) {
  //    for (int i = 0; i < edges.size(); i++) {
  //      scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
  //    }
  //  }
  //}

  }
  vertex_data->iteration_count++;
  //printf("updated vertex %d, %d\n", vid, vertex_data->iteration_count);
}


////////////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

// Helper method to check if a key point is inside a given bounding
// box.
__attribute__((const))
static bool bbox_contains(float pt_x, float pt_y,
                          int x_start, int x_finish,
                          int y_start, int y_finish) {
  // TRACE_1("  -- pt: (%f, %f)\n", pt.x, pt.y);
  // TRACE_1("  -- bbox: [(%d, %d), (%d, %d)]\n",
  //         x_start, y_start,
  //         x_finish, y_finish);
  return (pt_x >= x_start && pt_x <= x_finish) &&
    (pt_y >= y_start && pt_y <= y_finish);
}

// Helper method to match the features of two tiles.
static void match_features(std::vector< cv::DMatch > &matches,
                           cv::Mat &descs1, cv::Mat &descs2,
                           float rod) {
  std::vector< std::vector < cv::DMatch > > raw_matches;
  //cv::BFMatcher matcher(cv::NORM_L2, false);
  cv::FlannBasedMatcher matcher;
  matcher.knnMatch(descs1, descs2,
                   raw_matches,
                   2);

  matches.reserve(raw_matches.size());
  // Apply ratio test
  for (size_t i = 0; i < raw_matches.size(); i++) {
    if (raw_matches[i][0].distance <
        (ROD * raw_matches[i][1].distance)) {
      matches.push_back(raw_matches[i][0]);
    }
  }
}

__attribute__((const))
static double dist(const cv::Point2f a_pt, const cv::Point2f b_pt) {
  double x_delta = a_pt.x - b_pt.x;
  double y_delta = a_pt.y - b_pt.y;
  return std::sqrt((x_delta * x_delta) + (y_delta * y_delta));
}

const int INDENT_SPACES = 4;

static void save_tile_matches(size_t num_matches,
                      const char *out_filepath,
                      const tile_data_t *a_data,
                      const tile_data_t *b_data,
                      const std::vector< cv::Point2f > *match_points_a,
                      const std::vector< cv::Point2f > *match_points_b,
                      const std::vector< cv::Point2f > *match_points_a_fixed) {
#ifndef SKIPJSON
  static double totalTime = 0;
  FILE *fp;

  TRACE_1("save_tile_matches: start\n");

  TRACE_1("Writing %s\n", out_filepath);

  fasttime_t tstart = gettime();

  fp = fopen(out_filepath, "wb+");
  // Output prologue
  fprintf(fp, "[\n");
  fprintf(fp, "%*s%s\n", INDENT_SPACES, "", "{");

  // Output correspondence point pairs
  TRACE_1("  -- outputting %lu matches\n", num_matches);
  fprintf(fp, "%*s%s\n", 2*INDENT_SPACES, "",
          "\"correspondencePointPairs\": [");
  for (size_t i = 0; i < num_matches; ++i) {
    fprintf(fp, "%*s%s\n", 3*INDENT_SPACES, "", "{");

    // Emit distance
    fprintf(fp, "%*s%s %f, \n", 4*INDENT_SPACES, "",
            "\"dist_after_ransac\":",
            dist((*match_points_a_fixed)[i], (*match_points_b)[i]));
    // Emit first point
    fprintf(fp, "%*s%s\n", 4*INDENT_SPACES, "",
            "\"p1\": {");

    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "\"l\": [");
    fprintf(fp, "%*s%f, \n", 6*INDENT_SPACES, "",
            (*match_points_a)[i].x);
    fprintf(fp, "%*s%f\n", 6*INDENT_SPACES, "",
            (*match_points_a)[i].y);
    fprintf(fp, "%*s%s, \n", 5*INDENT_SPACES, "",
            "]");

    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "\"w\": [");
    fprintf(fp, "%*s%f, \n", 6*INDENT_SPACES, "",
            (*match_points_a)[i].x + a_data->x_start);
    fprintf(fp, "%*s%f\n", 6*INDENT_SPACES, "",
            (*match_points_a)[i].y + a_data->y_start);
    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "]");

    fprintf(fp, "%*s%s, \n", 4*INDENT_SPACES, "",
            "}");

    // Emit second point
    fprintf(fp, "%*s%s\n", 4*INDENT_SPACES, "",
            "\"p2\": {");

    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "\"l\": [");
    fprintf(fp, "%*s%f, \n", 6*INDENT_SPACES, "",
            (*match_points_b)[i].x);
    fprintf(fp, "%*s%f\n", 6*INDENT_SPACES, "",
            (*match_points_b)[i].y);
    fprintf(fp, "%*s%s, \n", 5*INDENT_SPACES, "",
            "]");

    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "\"w\": [");
    fprintf(fp, "%*s%f, \n", 6*INDENT_SPACES, "",
            (*match_points_b)[i].x + b_data->x_start);
    fprintf(fp, "%*s%f\n", 6*INDENT_SPACES, "",
            (*match_points_b)[i].y + b_data->y_start);
    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "]");

    fprintf(fp, "%*s%s\n", 4*INDENT_SPACES, "",
            "}");

    fprintf(fp, "%*s%s", 3*INDENT_SPACES, "", "}");
    if (i+1 < num_matches)
      fprintf(fp, ", ");
    fprintf(fp, "\n");
  }
  fprintf(fp, "%*s%s, \n", 2*INDENT_SPACES, "",
          "]");

  // Output mipmapLevel
  fprintf(fp, "%*s%s: %d, \n", 2*INDENT_SPACES, "",
          "\"mipmapLevel\"", 0);

  // TODO(TB): Output model
  //
  // TB: This field does not seem to be used by the optimization step.

  // Output input file images
  fprintf(fp, "%*s%s: \"file://%s\", \n", 2*INDENT_SPACES, "",
          "\"url1\"", a_data->filepath);
  fprintf(fp, "%*s%s: \"file://%s\"\n", 2*INDENT_SPACES, "",
          "\"url2\"", b_data->filepath);

  // Output epilogue
  fprintf(fp, "%*s%s\n", INDENT_SPACES, "", "}");
  fprintf(fp, "]");

  fclose(fp);

  fasttime_t tend = gettime();
  totalTime += tdiff(tstart, tend);
  TRACE_1("save_tile_matches cumulative time: %.6lf [sec]\n",
          totalTime);

  TRACE_1("save_tile_matches: finish\n");
#endif
}

////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

// thread_local std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;
// thread_local std::vector< cv::Mat > atile_kps_desc_in_overlap_list, btile_kps_desc_in_overlap_list;
// thread_local std::vector< cv::DMatch > matches;
// thread_local std::vector< cv::Point2f > filtered_match_points_a, filtered_match_points_b;
// thread_local std::vector< cv::Point2f > match_points_a, match_points_b;
// thread_local std::vector< cv::Point2f > match_points_a_fixed;

static std::vector<Graph<vdata, edata>* > graph_list;

void compute_tile_matches(align_data_t *p_align_data, int force_section_id) {
  TRACE_1("compute_tile_matches: start\n");




  //section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);

  // Iterate over all pairs of tiles
  //for (int sec_id = 0; sec_id < p_align_data->n_sections; sec_id++) {
  for (int sec_id = force_section_id; sec_id < force_section_id+1 && force_section_id != -1; sec_id++) {
    Graph<vdata, edata>* graph;
    Scheduler* scheduler;
    engine<vdata, edata>* e;
    section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);

    // Record the set of mfov ID's encountered.  Right now, this set
    // is used to limit the number of system calls performed.
    std::set<int> mfovs;
    simple_mutex_t mfovs_lock;
    simple_mutex_init(&mfovs_lock);


    printf("REsizing the graph to be size %d\n", p_sec_data->n_tiles);
    graph = new Graph<vdata, edata>();
    graph->resize(p_sec_data->n_tiles);
    graph_list.push_back(graph);

    cilk_for (int atile_id = 0; atile_id < p_sec_data->n_tiles; atile_id++) {
      //printf("The tile id is %d\n", atile_id);
      if (atile_id >= p_sec_data->n_tiles) {
        printf("Big error!\n");    
      }
      tile_data_t *a_tile = &(p_sec_data->tiles[atile_id]);
      int indices_to_check[50];
      int indices_to_check_len = 0;

      // get all close tiles.
      for (int i = atile_id+1; i < p_sec_data->n_tiles; i++) {
        tile_data_t *b_tile = &(p_sec_data->tiles[i]);
        // Skip tiles that don't overlap
        if (!is_tiles_overlap(a_tile, b_tile)) continue;
        indices_to_check[indices_to_check_len++] = i;
        if (indices_to_check_len > 49) {
           printf("Major error!!! wtf\n");
        }
        assert(indices_to_check_len <= 49);
      }

      for (int tmpindex = 0; tmpindex < indices_to_check_len; tmpindex++) {
         do {
        int btile_id = indices_to_check[tmpindex];
        tile_data_t *b_tile = &(p_sec_data->tiles[btile_id]);

        // Skip tiles that don't overlap
        if (!is_tiles_overlap(a_tile, b_tile)) continue;  // just in case.

        // Index pair is:
        // a_tile->mfov_id, a_tile->index
        // b_tile->mfov_id, b_tile->index
        //TRACE_1("    -- index_pair [%d_%d, %d_%d]\n",
        //        a_tile->mfov_id, a_tile->index,
        //        b_tile->mfov_id, b_tile->index);

        //TRACE_1("    -- %d_%d features_num: %lu\n",
        //        a_tile->mfov_id, a_tile->index,
        //        a_tile->p_kps->size());
        //TRACE_1("    -- %d_%d features_num: %lu\n",
        //        b_tile->mfov_id, b_tile->index,
        //        b_tile->p_kps->size());

        // Check that both tiles have enough features to match.
        if (a_tile->p_kps->size() < MIN_FEATURES_NUM) {
          //TRACE_1("Less than %d features in tile %d_%d, saving empty match file\n",
          //        MIN_FEATURES_NUM,
          //        a_tile->mfov_id, a_tile->index);
          continue;
        }
        if (b_tile->p_kps->size() < MIN_FEATURES_NUM) {
          //TRACE_1("Less than %d features in tile %d_%d, saving empty match file\n",
          //        MIN_FEATURES_NUM,
          //        b_tile->mfov_id, b_tile->index);
          continue;
        }

        // Filter the features, so that only features that are in the
        // overlapping tile will be matches.
        std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;
        atile_kps_in_overlap.reserve(a_tile->p_kps->size());
        btile_kps_in_overlap.reserve(b_tile->p_kps->size());
        // atile_kps_in_overlap.clear(); btile_kps_in_overlap.clear();
        cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
        {
          // Compute bounding box of overlap
          int overlap_x_start = a_tile->x_start > b_tile->x_start ?
                                    a_tile->x_start : b_tile->x_start;
          int overlap_x_finish = a_tile->x_finish < b_tile->x_finish ?
                                    a_tile->x_finish : b_tile->x_finish;
          int overlap_y_start = a_tile->y_start > b_tile->y_start ?
                                    a_tile->y_start : b_tile->y_start;
          int overlap_y_finish = a_tile->y_finish < b_tile->y_finish ?
                                    a_tile->y_finish : b_tile->y_finish;
          // Add 50-pixel offset
          const int OFFSET = 50;
          overlap_x_start -= OFFSET;
          overlap_x_finish += OFFSET;
          overlap_y_start -= OFFSET;
          overlap_y_finish += OFFSET;

          std::vector< cv::Mat > atile_kps_desc_in_overlap_list;
          atile_kps_desc_in_overlap_list.reserve(a_tile->p_kps->size());
          std::vector< cv::Mat > btile_kps_desc_in_overlap_list;
          btile_kps_desc_in_overlap_list.reserve(b_tile->p_kps->size());

          // Filter the points in a_tile.
          for (size_t pt_idx = 0; pt_idx < a_tile->p_kps->size(); ++pt_idx) {
            cv::Point2f pt = (*a_tile->p_kps)[pt_idx].pt;
            if (bbox_contains(pt.x + a_tile->x_start,
                              pt.y + a_tile->y_start,  // transformed_pt[0],
                              overlap_x_start, overlap_x_finish,
                              overlap_y_start, overlap_y_finish)) {
              atile_kps_in_overlap.push_back((*a_tile->p_kps)[pt_idx]);
              atile_kps_desc_in_overlap_list.push_back(
                  a_tile->p_kps_desc->row(pt_idx).clone());
            }
          }
          cv::vconcat(atile_kps_desc_in_overlap_list,
              (atile_kps_desc_in_overlap));

          // Filter the points in b_tile.
          for (size_t pt_idx = 0; pt_idx < b_tile->p_kps->size(); ++pt_idx) {
            cv::Point2f pt = (*b_tile->p_kps)[pt_idx].pt;
            if (bbox_contains(pt.x + b_tile->x_start,
                              pt.y + b_tile->y_start,  // transformed_pt[0],
                              overlap_x_start, overlap_x_finish,
                              overlap_y_start, overlap_y_finish)) {
              btile_kps_in_overlap.push_back((*b_tile->p_kps)[pt_idx]);
              btile_kps_desc_in_overlap_list.push_back(b_tile->p_kps_desc->row(pt_idx).clone());
            }
          }
          cv::vconcat(btile_kps_desc_in_overlap_list,
              (btile_kps_desc_in_overlap));
        }

        //TRACE_1("    -- %d_%d overlap_features_num: %lu\n",
        //        a_tile->mfov_id, a_tile->index,
        //        atile_kps_in_overlap.size());
        //TRACE_1("    -- %d_%d overlap_features_num: %lu\n",
        //        b_tile->mfov_id, b_tile->index,
        //        btile_kps_in_overlap.size());

        // TODO(TB): Deal with optionally filtering the maximal number of
        // features from one tile.
        //
        // TB: The corresponding code in the Python pipeline did not
        // appear to run at all, at least on the small data set, so
        // I'm skipping this part for now.

        // Check that both tiles have enough features in the overlap
        // to match.
        if (atile_kps_in_overlap.size() < MIN_FEATURES_NUM) {
          //TRACE_1("Less than %d features in the overlap in tile %d_%d, saving empty match file\n",
          //        MIN_FEATURES_NUM,
          //        a_tile->mfov_id, a_tile->index);
          continue;
        }
        if (btile_kps_in_overlap.size() < MIN_FEATURES_NUM) {
          //TRACE_1("Less than %d features in the overlap in tile %d_%d, saving empty match file\n",
          //        MIN_FEATURES_NUM,
          //        b_tile->mfov_id, b_tile->index);
          continue;
        }

        // Match the features
        std::vector< cv::DMatch > matches;
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       ROD);

        //TRACE_1("    -- [%d_%d, %d_%d] matches: %lu\n",
        //        a_tile->mfov_id, a_tile->index,
        //        b_tile->mfov_id, b_tile->index,
        //        matches.size());


        // Filter the matches with RANSAC
        std::vector<cv::Point2f> match_points_a, match_points_b;
        for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
          match_points_a.push_back(
              atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
          match_points_b.push_back(
              btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
        }

        // Use cv::findHomography to run RANSAC on the match points.
        //
        // TB: Using the maxEpsilon value (10) from
        // conf_example.json as the ransacReprojThreshold for
        // findHomography.
        //
        // TODO(TB): Read the appropriate RANSAC settings from the
        // configuration file.
        if (matches.size() < MIN_FEATURES_NUM) {
          //TRACE_1("Less than %d matched features, saving empty match file\n",
          //        MIN_FEATURES_NUM);
          if (matches.size() == 0) printf("There are zero matches.\n");
          continue;
        }

        bool* mask = (bool*) calloc(match_points_a.size(), 1);
        double thresh = 10.0;
        tfk_simple_ransac(match_points_a, match_points_b, thresh, mask);


        std::vector< cv::Point2f > filtered_match_points_a(0);
        std::vector< cv::Point2f > filtered_match_points_b(0);
        //filtered_match_points_a.reserve(matches.size());
        //filtered_match_points_b.reserve(matches.size());

        int num_matches_filtered = 0;
        // Use the output mask to filter the matches
        for (size_t i = 0; i < matches.size(); ++i) {
          if (mask[i]) {
            num_matches_filtered++;
            filtered_match_points_a.push_back(
                atile_kps_in_overlap[matches[i].queryIdx].pt);
            filtered_match_points_b.push_back(
                btile_kps_in_overlap[matches[i].trainIdx].pt);
          }
        }
        free(mask);
        if (num_matches_filtered > 0) {
          graph->insert_matches(atile_id, btile_id,
              filtered_match_points_a, filtered_match_points_b, 1.0);
        }

        if (filtered_match_points_a.size() < MIN_FEATURES_NUM) {
          //TRACE_1("Less than %d matched features, saving empty match file\n",
          //        MIN_FEATURES_NUM);
          continue;
        }

        // Write the JSON output
        {
          // Estimate the rigid transform from the matched points in
          // tile A to those in tile B.

          cv::Mat model = cv::estimateRigidTransform(filtered_match_points_a,
              filtered_match_points_b, false);

          //TRACE_1("    -- [%d_%d, %d_%d] estimated a %d by %d affine transform matrix.\n",
          //        a_tile->mfov_id, a_tile->index,
          //        b_tile->mfov_id, b_tile->index,
          //        model.rows, model.cols);

          if (0 == model.rows) {
            //TRACE_1("Could not estimate affine transform, saving empty match file\n");
            continue;
          }

          // Transform the matched points in tile A.  These
          // transformed points are used to estimate the distances
          // between matched points after alignment.
          std::vector< cv::Point2f > match_points_a_fixed;
          match_points_a_fixed.reserve(filtered_match_points_a.size());

          cv::transform(filtered_match_points_a, match_points_a_fixed, model);

          // Output the tile matches.
        }
      } while (false); // end the do while wrapper.
      }  // for (btile_id)
    }  // for (atile_id)

    // can free the matches. NOTE(TFK): Disabled temporarily.
    cilk_for (int atile_id = 0; atile_id < p_sec_data->n_tiles; atile_id++) {
      tile_data_t *a_tile = &(p_sec_data->tiles[atile_id]);
      a_tile->p_kps->clear();
      std::vector<cv::KeyPoint>().swap(*(a_tile->p_kps));
      ((a_tile->p_kps_desc))->release();
    }

  printf("Size of the graph is %d\n", graph->num_vertices());
  for (int i = 0; i < graph->num_vertices(); i++) {
    //printf("init vertex data for vertex %d\n", i);
    vdata* d = graph->getVertexData(i);
    _tile_data tdata = p_sec_data->tiles[i];
    d->vertex_id = i;
    d->mfov_id = tdata.mfov_id;
    d->tile_index = tdata.index;
    d->tile_id = i;
    d->start_x = tdata.x_start;
    d->end_x = tdata.x_finish;
    d->start_y = tdata.y_start;
    d->end_y = tdata.y_finish;
    d->offset_x = 0.0;
    d->offset_y = 0.0;
    d->iteration_count = 0;
    d->last_radius_value = 9.0;
    d->z = sec_id;
    d->a00 = 1.0;
    d->a01 = 0.0;
    d->a10 = 0.0;
    d->a11 = 1.0;
  }
  graph->section_id = sec_id;
  }  // for (sec_id)



  if (force_section_id != -1) return;
  printf("Done with all sections now doing graph computation");
 



  printf("First we need to merge the graphs.\n");
  int total_size = 0;
  for (int i = 0; i < graph_list.size(); i++) {
    total_size += graph_list[i]->num_vertices();
  }
  Graph<vdata, edata>* merged_graph = new Graph<vdata, edata>();
  merged_graph->resize(total_size);

  int vertex_id_offset = 0;
  for (int i = 0; i < graph_list.size(); i++) {
    for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
      vdata* d = merged_graph->getVertexData(j+vertex_id_offset);
      *d = *(graph_list[i]->getVertexData(j));
      d->vertex_id += vertex_id_offset;
      //_tile_data tdata = p_sec_data->tiles[i];
      //d->vertex_id = i;
      //d->mfov_id = tdata.mfov_id;
      //d->tile_index = tdata.index;
      //d->tile_id = i;
      //d->start_x = tdata.x_start;
      //d->end_x = tdata.x_finish;
      //d->start_y = tdata.y_start;
      //d->end_y = tdata.y_finish;
      //d->offset_x = 0.0;
      //d->offset_y = 0.0;
      //d->iteration_count = 0;
      //d->last_radius_value = 9.0;
    }
    vertex_id_offset += graph_list[i]->num_vertices();
  }

  vertex_id_offset = 0;
  // now insert the edges.
  for (int i = 0; i < graph_list.size(); i++) {
    for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
      for (int k = 0; k < graph_list[i]->edgeData[j].size(); k++) {
        edata edge = graph_list[i]->edgeData[j][k];
        edge.neighbor_id += vertex_id_offset;
        merged_graph->insertEdge(j+vertex_id_offset, edge);
      }
    }
    vertex_id_offset += graph_list[i]->num_vertices();
  }

    vertex_id_offset = 0;
    // now compute all of the matches between sections.
   
  double COMPARE_RADIUS = 0.0;
  if (ASSUME_PRE_ALIGNED) {

    for (int i = 1; i < graph_list.size(); i++) {
      for (int atile_id = 0; atile_id < graph_list[i-1]->num_vertices(); atile_id++) {
        _tile_data a_tile = p_align_data->sec_data[i-1].tiles[atile_id];

       
        // get all close tiles.
        int indices_to_check[500];
        int indices_to_check_len = 0;
        for (int tmpi = 0; tmpi < graph_list[i]->num_vertices(); tmpi++) {
          _tile_data tdata_b = p_align_data->sec_data[i].tiles[tmpi];
          tile_data_t b_tile = (p_align_data->sec_data[i].tiles[tmpi]);
          // Skip tiles that don't overlap
   
          if (!is_tiles_overlap_slack(&a_tile, &b_tile, COMPARE_RADIUS)) continue;
          indices_to_check[indices_to_check_len++] = tmpi;
          if (indices_to_check_len > 499) {
             printf("Major error!!! wtf\n");
          }
          assert(indices_to_check_len <= 499);
	}
        if (indices_to_check_len == 0) continue;


        std::vector <cv::KeyPoint > atile_kps_in_overlap;
        std::vector <cv::Mat > atile_kps_desc_in_overlap_list;
        std::vector<int> atile_kps_tile_list; 
        std::vector <cv::KeyPoint > btile_kps_in_overlap;
        std::vector <cv::Mat > btile_kps_desc_in_overlap_list;
        std::vector<int> btile_kps_tile_list;

        concat_two_tiles(&a_tile, atile_id+vertex_id_offset, atile_kps_in_overlap, atile_kps_desc_in_overlap_list, atile_kps_tile_list, &a_tile, COMPARE_RADIUS);


        for (int tmpi = 0; tmpi < indices_to_check_len; tmpi++) {
        int extra_offset = graph_list[i-1]->num_vertices();
          int btile_id = indices_to_check[tmpi];
          _tile_data tdata_b = p_align_data->sec_data[i].tiles[btile_id];
          tile_data_t b_tile = (p_align_data->sec_data[i].tiles[btile_id]);
          concat_two_tiles(&b_tile, btile_id+vertex_id_offset+extra_offset, btile_kps_in_overlap, btile_kps_desc_in_overlap_list, btile_kps_tile_list, &a_tile, COMPARE_RADIUS);
  
        }
        printf("Total size of a tile kps is %d\n", atile_kps_in_overlap.size());
        printf("Total size of b tile kps is %d\n", btile_kps_in_overlap.size());
  
        cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
        cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
        cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));

        std::vector< cv::DMatch > matches;
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       0.92);
            printf("Done with the matching. Num matches is %d\n", matches.size());
  
            // Filter the matches with RANSAC
            std::vector<cv::Point2f> match_points_a, match_points_b;
            for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
        //std::vector<int> atile_kps_tile_list;
              int atile_id = atile_kps_tile_list[matches[tmpi].queryIdx];
              int btile_id = btile_kps_tile_list[matches[tmpi].trainIdx];
              float x_start_a = merged_graph->getVertexData(atile_id)->start_x;
              float y_start_a = merged_graph->getVertexData(atile_id)->start_y;
  
              float x_start_b = merged_graph->getVertexData(btile_id)->start_x;
              float y_start_b = merged_graph->getVertexData(btile_id)->start_y;
  
              match_points_a.push_back(
                  atile_kps_in_overlap[matches[tmpi].queryIdx].pt + cv::Point2f(x_start_a, y_start_a));
              match_points_b.push_back(
                  btile_kps_in_overlap[matches[tmpi].trainIdx].pt + cv::Point2f(x_start_b, y_start_b));
              std::cout << (btile_kps_in_overlap[matches[tmpi].trainIdx].pt + cv::Point2f(x_start_b, y_start_b)) << std::endl;
            }
            //void tfk_simple_ransac(std::vector<cv::Point2f>& match_points_a,
            //    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {
            bool* mask = (bool*)calloc(matches.size()+1, 1);
            tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 25.0, mask);
        if (matches.size() > 4) { 
        for (int c = 0; c < matches.size(); c++) {
          if (mask[c]) {
            std::vector< cv::Point2f > filtered_match_points_a(0);
            std::vector< cv::Point2f > filtered_match_points_b(0);
            int atile_id = atile_kps_tile_list[matches[c].queryIdx];
            int btile_id = btile_kps_tile_list[matches[c].trainIdx];
            filtered_match_points_a.push_back(
                atile_kps_in_overlap[matches[c].queryIdx].pt);
            filtered_match_points_b.push_back(
                btile_kps_in_overlap[matches[c].trainIdx].pt);
            merged_graph->insert_matches(atile_id, btile_id,
                filtered_match_points_a, filtered_match_points_b, 1.0);
          }
        }
        }
        free(mask);
      }
      vertex_id_offset += graph_list[i-1]->num_vertices();
    }

  } else if (false) {  
    //void concat_two_tiles(tile_data_t* a_tile, int atile_id, std::vector< cv::KeyPoint >& atile_kps_in_overlap, std::vector < cv::Mat >& atile_kps_desc_in_overlap_list, std::vector<int>& atile_kps_tile_list) {
    vertex_id_offset = 0;

    // now insert the edges.
    for (int i = 1; i < graph_list.size(); i++) {
      std::vector <cv::KeyPoint > atile_kps_in_overlap;
      std::vector <cv::Mat > atile_kps_desc_in_overlap_list;
      std::vector<int> atile_kps_tile_list; 
      std::vector <cv::KeyPoint > btile_kps_in_overlap;
      std::vector <cv::Mat > btile_kps_desc_in_overlap_list;
      std::vector<int> btile_kps_tile_list;

      for (int j = 0; j < graph_list[i-1]->num_vertices(); j++) {
        int atile_id = j+vertex_id_offset;
        _tile_data tdata_a = p_align_data->sec_data[i-1].tiles[j];
        concat_two_tiles_all(&tdata_a, atile_id, atile_kps_in_overlap, atile_kps_desc_in_overlap_list, atile_kps_tile_list);
      }
      vertex_id_offset += graph_list[i-1]->num_vertices();
      for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
        int btile_id = j+vertex_id_offset;
        _tile_data tdata_b = p_align_data->sec_data[i].tiles[j];
        concat_two_tiles_all(&tdata_b, btile_id, btile_kps_in_overlap, btile_kps_desc_in_overlap_list, btile_kps_tile_list);
      }

      printf("Total size of a tile kps is %d\n", atile_kps_in_overlap.size());
      printf("Total size of b tile kps is %d\n", btile_kps_in_overlap.size());

      cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
      cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
      cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));

      std::vector< cv::DMatch > matches;
      match_features(matches,
                     atile_kps_desc_in_overlap,
                     btile_kps_desc_in_overlap,
                     0.92);
          printf("Done with the matching. Num matches is %d\n", matches.size());

          // Filter the matches with RANSAC
          std::vector<cv::Point2f> match_points_a, match_points_b;
          for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
      //std::vector<int> atile_kps_tile_list;
            int atile_id = atile_kps_tile_list[matches[tmpi].queryIdx];
            int btile_id = btile_kps_tile_list[matches[tmpi].trainIdx];
            int x_start_a = merged_graph->getVertexData(atile_id)->start_x + merged_graph->getVertexData(atile_id)->offset_x;
            int y_start_a = merged_graph->getVertexData(atile_id)->start_y + merged_graph->getVertexData(atile_id)->offset_y;

            int x_start_b = merged_graph->getVertexData(btile_id)->start_x + merged_graph->getVertexData(btile_id)->offset_x;
            int y_start_b = merged_graph->getVertexData(btile_id)->start_y + merged_graph->getVertexData(btile_id)->offset_y;

            match_points_a.push_back(
                atile_kps_in_overlap[matches[tmpi].queryIdx].pt + cv::Point2f(x_start_a, y_start_a));
            match_points_b.push_back(
                btile_kps_in_overlap[matches[tmpi].trainIdx].pt + cv::Point2f(x_start_b, y_start_b));
          }
          //void tfk_simple_ransac(std::vector<cv::Point2f>& match_points_a,
          //    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {
          bool* mask = (bool*)calloc(matches.size()+1, 1);
          std::pair<double,double> offset_pair = tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 10.0, mask);
      for (int c = 0; c < matches.size(); c++) {
        if (mask[c]) {
          std::vector< cv::Point2f > filtered_match_points_a(0);
          std::vector< cv::Point2f > filtered_match_points_b(0);
          int atile_id = atile_kps_tile_list[matches[c].queryIdx];
          int btile_id = btile_kps_tile_list[matches[c].trainIdx];
          filtered_match_points_a.push_back(
              atile_kps_in_overlap[matches[c].queryIdx].pt);
          filtered_match_points_b.push_back(
              btile_kps_in_overlap[matches[c].trainIdx].pt);
          merged_graph->insert_matches(atile_id, btile_id,
              filtered_match_points_a, filtered_match_points_b, 1.0);
        }
      }
      free(mask);
        //for (int v = vertex_id_offset - graph_list[i-1]->num_vertices(); v < vertex_id_offset; v++) {
        //  merged_graph->getVertexData(v)->offset_x = -1.0*offset_pair.first;
        //  merged_graph->getVertexData(v)->offset_y = -1.0*offset_pair.second;
        //}

      //exit(0);
    }
  }


  

  //    for (int i = 0; i <  


//    for (int i = 1; i < graph_list.size(); i++) {
//      int sec_id_a = i-1;
//      int sec_id_b = i;
//
//      cilk_for (int j = 0; j < graph_list[sec_id_a]->num_vertices(); j++) {
//        int tile_index_a = graph_list[sec_id_a]->getVertexData(j)->tile_index;
//        _tile_data tdata_a = p_align_data->sec_data[sec_id_a].tiles[j];
//        for (int k = 0; k < graph_list[sec_id_b]->num_vertices(); 
//             k++) {
//          int tile_index_b = graph_list[sec_id_b]->getVertexData(k)->tile_index;
//          _tile_data tdata_b = p_align_data->sec_data[sec_id_b].tiles[k];
//          compare_two_tiles(&tdata_a, &tdata_b,
//                            tile_index_a, tile_index_b+graph_list[sec_id_a]->num_vertices(), merged_graph);        
//        }
//      }
//    }
     //compare_two_tiles(tile_data_t* a_tile, tile_data_t* b_tile, int atile_id, int btile_id, Graph<vdata, edata>* graph) {

    int ncolors = merged_graph->compute_trivial_coloring();
    Scheduler* scheduler;
    engine<vdata, edata>* e;
    scheduler =
        new Scheduler(merged_graph->vertexColors, ncolors+1, merged_graph->num_vertices());
    scheduler->graph_void = (void*) merged_graph;
    e = new engine<vdata, edata>(merged_graph, scheduler);
    for (int i = 0; i < merged_graph->num_vertices(); i++) {
      scheduler->add_task(i, updateVertex2DAlign);
    }
    printf("starting run\n");
    e->run();
    printf("ending run\n");

    if (true) {
    vertex_id_offset = 0;



    for (int section = 1; section < p_align_data->n_sections; section++) {
      int section_a = section-1;
      int section_b = section;
      std::vector <cv::KeyPoint > atile_kps_in_overlap;
      std::vector <cv::Mat > atile_kps_desc_in_overlap_list;
      std::vector<int> atile_kps_tile_list; 
      std::vector <cv::KeyPoint > btile_kps_in_overlap;
      std::vector <cv::Mat > btile_kps_desc_in_overlap_list;
      std::vector<int> btile_kps_tile_list;

      for (int v = 1; v < merged_graph->num_vertices(); v++) {
        if (merged_graph->getVertexData(v)->z == section_a) {
          _tile_data tdata_a = p_align_data->sec_data[section_a].tiles[merged_graph->getVertexData(v)->tile_id];
          concat_two_tiles_all(&tdata_a, v, atile_kps_in_overlap, atile_kps_desc_in_overlap_list, atile_kps_tile_list);
        } else if (merged_graph->getVertexData(v)->z == section_b) {
          _tile_data tdata_b = p_align_data->sec_data[section_b].tiles[merged_graph->getVertexData(v)->tile_id];
          concat_two_tiles_all(&tdata_b, v, btile_kps_in_overlap, btile_kps_desc_in_overlap_list, btile_kps_tile_list);
        }
      }

      printf("Total size of a tile kps is %d\n", atile_kps_in_overlap.size());
      printf("Total size of b tile kps is %d\n", btile_kps_in_overlap.size());
 
      cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
      cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
      cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));

      std::vector< cv::DMatch > matches;
      match_features(matches,
                     atile_kps_desc_in_overlap,
                     btile_kps_desc_in_overlap,
                     0.92);
          printf("Done with the matching. Num matches is %d\n", matches.size());
          // Filter the matches with RANSAC
          std::vector<cv::Point2f> match_points_a, match_points_b;
          for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
          //std::vector<int> atile_kps_tile_list;
            int atile_id = atile_kps_tile_list[matches[tmpi].queryIdx];
            int btile_id = btile_kps_tile_list[matches[tmpi].trainIdx];

            int x_start_a = merged_graph->getVertexData(atile_id)->start_x + merged_graph->getVertexData(atile_id)->offset_x;
            int y_start_a = merged_graph->getVertexData(atile_id)->start_y + merged_graph->getVertexData(atile_id)->offset_y;

            int x_start_b = merged_graph->getVertexData(btile_id)->start_x + merged_graph->getVertexData(btile_id)->offset_x;
            int y_start_b = merged_graph->getVertexData(btile_id)->start_y + merged_graph->getVertexData(btile_id)->offset_y;

            //int x_start_a = merged_graph->getVertexData(atile_id)->start_x;
            //int y_start_a = merged_graph->getVertexData(atile_id)->start_y;

            //int x_start_b = merged_graph->getVertexData(btile_id)->start_x;
            //int y_start_b = merged_graph->getVertexData(btile_id)->start_y;

            match_points_a.push_back(
                atile_kps_in_overlap[matches[tmpi].queryIdx].pt + cv::Point2f(x_start_a, y_start_a));
            match_points_b.push_back(
                btile_kps_in_overlap[matches[tmpi].trainIdx].pt + cv::Point2f(x_start_b, y_start_b));
          }
          bool* mask = (bool*)calloc(matches.size()+1, 1);
          std::pair<double,double> offset_pair = tfk_simple_ransac_strict_ret(match_points_a, match_points_b, 20.0, mask);
          std::vector< cv::Point2f > filtered_match_points_a(0);
          std::vector< cv::Point2f > filtered_match_points_b(0);
      for (int c = 0; c < matches.size(); c++) {
        if (mask[c]) {
          int atile_id = atile_kps_tile_list[matches[c].queryIdx];
          int btile_id = btile_kps_tile_list[matches[c].trainIdx];
          //filtered_match_points_a.push_back(
          //    atile_kps_in_overlap[matches[c].queryIdx].pt);
          //filtered_match_points_b.push_back(
          //    btile_kps_in_overlap[matches[c].trainIdx].pt);

          filtered_match_points_a.push_back(
              match_points_a[matches[c].queryIdx]);
          filtered_match_points_b.push_back(
              match_points_b[matches[c].trainIdx]);

          //merged_graph->insert_matches(atile_id, btile_id,
          //    filtered_match_points_a, filtered_match_points_b, 1.0);
        }
      }

      int iteration_count = 0;
      double b_offset_x = offset_pair.first;
      double b_offset_y = offset_pair.second;
      while(true) {
        iteration_count++;
        double grad_x = 0.0;
        double grad_y = 0.0;
        for (int subid = 0; subid < filtered_match_points_a.size(); subid++) {
          cv::Point2f pt_a = filtered_match_points_a[subid];
          cv::Point2f pt_b = filtered_match_points_b[subid];
          grad_x += 2*(pt_b.x-b_offset_x-pt_a.x);
          grad_y += 2*(pt_b.y-b_offset_y-pt_a.y);
        }
        grad_x = (100.0 / (100.0+iteration_count)) *grad_x*0.49/(filtered_match_points_a.size()+1);
        grad_y = (100.0 / (100.0+iteration_count)) *grad_y*0.49/(filtered_match_points_a.size()+1);
        b_offset_x += grad_x;
        b_offset_y += grad_y;
        if (std::abs(grad_x)+std::abs(grad_y) < 1e-2) {
          break;
        }
        if (iteration_count % 100000 == 0) {
          printf("Delta is currently %f\n", std::abs(grad_x)+std::abs(grad_y));
        }
      }

      free(mask);
      for (int v = 0; v < merged_graph->num_vertices(); v++) {
        if (merged_graph->getVertexData(v)->z <= section_a) {
          merged_graph->getVertexData(v)->offset_x += b_offset_x;//offset_pair.first;
          merged_graph->getVertexData(v)->offset_y += b_offset_y;//offset_pair.second;
        }
      }

    }

    }






  vertex_id_offset = 0;
  for (int i = 0; i < graph_list.size(); i++) {
    for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
      vdata* d = merged_graph->getVertexData(j+vertex_id_offset);
      *(graph_list[i]->getVertexData(j)) = *d;
      (graph_list[i]->getVertexData(j))->vertex_id -= vertex_id_offset;
      //d->vertex_id += vertex_id_offset;
    }
    vertex_id_offset += graph_list[i]->num_vertices();
  }




  for (int i = 0; i < graph_list.size(); i++) {
    Graph<vdata, edata>* graph = graph_list[i];
    int sec_id = graph->section_id;
    //int ncolors = graph->compute_trivial_coloring();
    //Scheduler* scheduler;
    //engine<vdata, edata>* e;
    //scheduler =
    //    new Scheduler(graph->vertexColors, ncolors+1, graph->num_vertices());
    //scheduler->graph_void = (void*) graph;
    //e = new engine<vdata, edata>(graph, scheduler);
    //for (int i = 0; i < graph->num_vertices(); i++) {
    //  scheduler->add_task(i, updateVertex2DAlign);
    //}
    //printf("starting run\n");
    //e->run();
    //printf("ending run\n");

    //void compute_tile_matches(align_data_t *p_align_data) {
    std::string section_id_string =
        std::to_string(p_align_data->sec_data[sec_id].section_id +
        p_align_data->base_section+1);

    FILE* wafer_file = fopen((std::string(p_align_data->output_dirpath)+std::string("/W01_Sec") +
        matchPadTo(section_id_string, 3)+std::string("_montaged.json")).c_str(), "w+");
    fprintf(wafer_file, "[\n");
    for (int i = 0; i < graph->num_vertices(); i++) {
      printf("affine params %f %f %f %f\n", graph->getVertexData(i)->a00, graph->getVertexData(i)->a01, graph->getVertexData(i)->a10, graph->getVertexData(i)->a11);
      vdata* vd = graph->getVertexData(i);
      //if (vd->z == 1) {
      //vd->a00 = 1.1;
      //vd->a01 = 0.0;
      //vd->a10 = 0.0;
      //vd->a11 = 1.1;
      //}
      fprintf(wafer_file, "\t{\n");
      fprintf(wafer_file, "\t\t\"bbox\": [\n");
      fprintf(wafer_file,
          "\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f\n],",
          vd->start_x+vd->offset_x, (vd->end_x+vd->offset_x),
          vd->start_y+vd->offset_y, (vd->end_y+vd->offset_y));
      fprintf(wafer_file, "\t\t\"height\": %d,\n",2724);
      fprintf(wafer_file, "\t\t\"layer\": %d,\n",p_align_data->sec_data[sec_id].section_id + p_align_data->base_section+1);
      fprintf(wafer_file, "\t\t\"maxIntensity\": %f,\n",255.0);
      fprintf(wafer_file, "\t\t\"mfov\": %d,\n",
          graph->getVertexData(i)->mfov_id);
      fprintf(wafer_file, "\t\t\"minIntensity\": %f,\n",
          0.0);
      fprintf(wafer_file, "\t\t\"mipmapLevels\": {\n");
      fprintf(wafer_file, "\t\t\"0\": {\n");
      fprintf(wafer_file, "\t\t\t\"imageUrl\": \"%s\"\n", p_align_data->sec_data[sec_id].tiles[graph->getVertexData(i)->tile_id].filepath);
      fprintf(wafer_file, "\t\t\t}\n");
      fprintf(wafer_file, "\t\t},\n");
      fprintf(wafer_file, "\t\t\"tile_index\": %d,\n",
          graph->getVertexData(i)->tile_index);
      fprintf(wafer_file, "\t\t\"transforms\": [\n");
      fprintf(wafer_file, "\t\t\t{\n");



      // {'className': 'mpicbg.trakem2.transform.AffineModel2D', 'dataString': '0.1 0.0 0.0 0.1 0.0 0.0'}

      fprintf(wafer_file,
          "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.AffineModel2D\",\n");
      fprintf(wafer_file,
          "\t\t\t\t\"dataString\": \"%f %f %f %f %f %f\"\n", vd->a00,
          vd->a01, vd->a10, vd->a11, vd->start_x + vd->offset_x, vd->start_y + vd->offset_y);

      //fprintf(wafer_file,
      //    "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.RigidModel2D\",\n");
      //fprintf(wafer_file,
      //    "\t\t\t\t\"dataString\": \"%f %f %f\"\n", 0.0,
      //    vd->start_x+vd->offset_x, vd->start_y + vd->offset_y);
      fprintf(wafer_file,
          "\t\t\t}\n");
      fprintf(wafer_file,
          "\t\t],\n");
      fprintf(wafer_file,
          "\t\t\"width\":%d\n",3128);
      if (i != graph->num_vertices()-1) {
        fprintf(wafer_file,
            "\t},\n");
      } else {
        fprintf(wafer_file,
            "\t}\n]");
      }
    }
    fclose(wafer_file);
    delete graph;
    delete scheduler;
    delete e;
  }

  //TRACE_1("compute_tile_matches: finish\n");
}

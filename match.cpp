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
#include <algorithm>
#include "./common.h"
#include "./align.h"
#include "./fasttime.h"
#include "./mesh.h"
#include "./simple_mutex.h"
#include "cilk_tools/engine.h"
//static cv::Point2f invert_transform_point(vdata* vertex, cv::Point2f point_local) {
//  float new_x = point_local.x*vertex->ia00 + point_local.y * vertex->ia01 + vertex->ioffset_x;// + vertex->start_x;
//  float new_y = point_local.x*vertex->ia10 + point_local.y * vertex->ia11 + vertex->ioffset_y;// + vertex->start_y;
//  //float new_x = point_local.x+vertex->offset_x+vertex->start_x;
//  //float new_y = point_local.y+vertex->offset_y+vertex->start_y;
//  return cv::Point2f(new_x, new_y);
//}




static std::string matchPadTo(std::string str, const size_t num, const char paddingChar = '0')
{
    if(num > str.size())
        str.insert(0, num - str.size(), paddingChar);
    return str;
}



static cv::Point2d transform_point_double(vdata* vertex, cv::Point2f point_local) {
  double new_x = point_local.x*vertex->a00 + point_local.y * vertex->a01 + vertex->offset_x + vertex->start_x;
  double new_y = point_local.x*vertex->a10 + point_local.y * vertex->a11 + vertex->offset_y + vertex->start_y;
  return cv::Point2d(new_x, new_y);
}

static cv::Point2f transform_point(vdata* vertex, cv::Point2f point_local) {
  float new_x = point_local.x*vertex->a00 + point_local.y * vertex->a01 + vertex->offset_x + vertex->start_x;
  float new_y = point_local.x*vertex->a10 + point_local.y * vertex->a11 + vertex->offset_y + vertex->start_y;
  return cv::Point2f(new_x, new_y);
}



std::string get_point_transform_string(Graph<vdata, edata>* merged_graph, vdata* vd) {
  std::string ret = "";
  for (int i = 0; i < vd->my_mesh_points->size(); i++) {
      cv::Point2f my_point = (*(vd->section_data->mesh_orig))[(*(vd->my_mesh_points))[i]];
      cv::Point2f n_point = (*(vd->section_data->mesh))[(*(vd->my_mesh_points))[i]];
      if (i == 0) {
      ret += std::to_string(my_point.x) + " " + std::to_string(my_point.y) + " " +
             std::to_string(n_point.x) + " " + std::to_string(n_point.y) + " " + std::to_string(1.0);
      } else {
      ret += " " + std::to_string(my_point.x) + " " + std::to_string(my_point.y) + " " +
             std::to_string(n_point.x) + " " + std::to_string(n_point.y) + " " + std::to_string(1.0);
      }
  }
  return ret;
}


void concat_two_tiles_all(vdata* vertex_data, tile_data_t* a_tile, int atile_id, std::vector< cv::KeyPoint >& atile_kps_in_overlap, std::vector < cv::Mat >& atile_kps_desc_in_overlap_list, std::vector<int>& atile_kps_tile_list) {
  if (a_tile->p_kps_3d->size() <= 0) {
     //printf("Skipping because p_kps_3d is zero.\n");
     return;
  }

  for (size_t pt_idx = 0; pt_idx < a_tile->p_kps_3d->size(); ++pt_idx) {
    if (a_tile->ignore[pt_idx]) continue;
    cv::Point2f pt = transform_point(vertex_data, (*a_tile->p_kps_3d)[pt_idx].pt);
    cv::KeyPoint kpt = (*a_tile->p_kps_3d)[pt_idx];
    kpt.pt = pt;
    atile_kps_in_overlap.push_back(kpt);
    atile_kps_desc_in_overlap_list.push_back(a_tile->p_kps_desc_3d->row(pt_idx).clone());
    atile_kps_tile_list.push_back(atile_id);
  }
}

void concat_two_tiles_all_filter(vdata* vertex_data, tile_data_t* a_tile, int atile_id, std::vector< cv::KeyPoint >& atile_kps_in_overlap, std::vector < cv::Mat >& atile_kps_desc_in_overlap_list, std::vector<int>& atile_kps_tile_list,
double _min_x, double _min_y, double _max_x, double _max_y) {
  for (size_t pt_idx = 0; pt_idx < a_tile->p_kps_3d->size(); ++pt_idx) {
    if (a_tile->ignore[pt_idx]) continue;
    cv::Point2f pt = transform_point(vertex_data, (*a_tile->p_kps_3d)[pt_idx].pt);
    cv::KeyPoint kpt = (*a_tile->p_kps_3d)[pt_idx];
    kpt.pt = pt;
    if (pt.x < _min_x || pt.x > _max_x || pt.y < _min_y || pt.y > _max_y) {
      continue;
    } 
    atile_kps_in_overlap.push_back(kpt);
    atile_kps_desc_in_overlap_list.push_back(a_tile->p_kps_desc_3d->row(pt_idx).clone());
    atile_kps_tile_list.push_back(atile_id);
  }
}

void concat_two_tiles(tile_data_t* a_tile, int atile_id, std::vector< cv::KeyPoint >& atile_kps_in_overlap, std::vector < cv::Mat >& atile_kps_desc_in_overlap_list, std::vector<int>& atile_kps_tile_list, tile_data_t* ref_tile, double radius) {
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
}


#include "ransac.h"

static void match_features(std::vector< cv::DMatch > &matches,
                           cv::Mat &descs1, cv::Mat &descs2,
                           float rod);
void updateVertex2DAlign(int vid, void* scheduler_void);
#include "meshoptimize.h"
#include "elastic_mesh.h"

std::vector<Graph<vdata, edata>* > graph_list;

#include "./match_helpers.h"
void set_graph_list(std::vector<Graph<vdata,edata>* > _graph_list, bool startEmpty) {
  if (startEmpty) graph_list.clear();

  for (int i = 0; i < _graph_list.size(); i++) {
    graph_list.push_back(_graph_list[i]);
  }
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
static void match_features(std::vector< cv::DMatch > &matches, cv::Mat &descs1, cv::Mat &descs2, float rod) {
  std::vector< std::vector < cv::DMatch > > raw_matches;
  if (true || descs1.rows + descs1.cols > descs2.rows + descs2.cols) {
    //cv::BFMatcher matcher(cv::NORM_L2, false);
    cv::FlannBasedMatcher matcher;
    matcher.knnMatch(descs1, descs2,
                     raw_matches,
                     2);

    matches.reserve(raw_matches.size());
    // Apply ratio test
    for (size_t i = 0; i < raw_matches.size(); i++) {
      if (raw_matches[i][0].distance <
          (rod * raw_matches[i][1].distance)) {
        matches.push_back(raw_matches[i][0]);
      }
    }
  } else {
    //cv::BFMatcher matcher(cv::NORM_L2, false);
    cv::FlannBasedMatcher matcher;
    matcher.knnMatch(descs2, descs1,
                     raw_matches,
                     2);

    matches.reserve(raw_matches.size());
    // Apply ratio test
    for (size_t i = 0; i < raw_matches.size(); i++) {
      if (raw_matches[i][0].distance <
          (rod * raw_matches[i][1].distance)) {
        matches.push_back(cv::DMatch(raw_matches[i][0].trainIdx, raw_matches[i][0].queryIdx, raw_matches[i][0].distance));
      }
    }
   
  } 
}

//__attribute__((const))
//static double dist(const cv::Point2f a_pt, const cv::Point2f b_pt) {
//  double x_delta = a_pt.x - b_pt.x;
//  double y_delta = a_pt.y - b_pt.y;
//  return std::sqrt((x_delta * x_delta) + (y_delta * y_delta));
//}


////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
/*
  gets all of the close tiles with an ID more than atile_id
  and puts the list into the array indices to check
  assuures that their are less than 50 close tiles
*/
int get_all_close_tiles(int atile_id, section_data_t *p_sec_data, int* indices_to_check) {
  int indices_to_check_len = 0;
  tile_data_t *a_tile = &(p_sec_data->tiles[atile_id]);
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
  return indices_to_check_len;
}


void compute_alignment_3d(align_data_t *p_align_data,
    Graph<vdata, edata>* merged_graph, bool construct_tri) {


  // NOTE(TFK): These transformations are kind-of silly, but are an artifact of
  //   a past implementation in which we relied on unpacked representation more.
  // Unpack the graphs within the merged graph.
  //if (construct_tri) {
    int vertex_id_offset = 0;
    for (int i = 0; i < graph_list.size(); i++) {
      for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
        vdata* d = merged_graph->getVertexData(j+vertex_id_offset);
        *(graph_list[i]->getVertexData(j)) = *d;
        (graph_list[i]->getVertexData(j))->vertex_id -= vertex_id_offset;
      }
      vertex_id_offset += graph_list[i]->num_vertices();
    }

    // These functions, for some reason, still require that we operate on a per-section graph.
    //   that's why we did the unpack and repack.
    for (int i = 0; i < graph_list.size(); i++) {
      construct_triangles(graph_list[i], 3500.0);
      filter_overlap_points_3d(graph_list[i], p_align_data);
    }

    // repack.
    vertex_id_offset = 0;
    for (int i = 0; i < graph_list.size(); i++) {
      for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
        vdata* d = merged_graph->getVertexData(j+vertex_id_offset);
        *d = *(graph_list[i]->getVertexData(j));
        d->vertex_id += vertex_id_offset;
      }
      vertex_id_offset += graph_list[i]->num_vertices();
    }

  // Now we do the actual 3d alignment stuff.

  // Coarse alignment 3D:
  //   This function makes *no* assumptions about how well the sections are aligned.
  //   It looks at sift features throughout the image and tries to find a good affine transformation.
  //   This function is important because the preliminary alignment it discovers allows later algorithms
  //    like fine_alignment_3d to assume that sections are "roughly" aligned --- allowing one to project 
  //    a neighborhood of one section into a larger neighborhood of another.
  coarse_alignment_3d(merged_graph, p_align_data, 64.0);

  // Fine alignment 3D:
  fine_alignment_3d(merged_graph, p_align_data);

  // Elastic mesh optimize.
  elastic_mesh_optimize(merged_graph, p_align_data);
  //}
}


void compute_alignment_2d(align_data_t *p_align_data,
    Graph<vdata, edata>* merged_graph) {
  TRACE_1("compute_alignment_2d: start\n");

  int ncolors = merged_graph->compute_trivial_coloring();
  Scheduler* scheduler;
  engine<vdata, edata>* e;
  scheduler =
      new Scheduler(merged_graph->vertexColors, ncolors+1, merged_graph->num_vertices());
  scheduler->graph_void = (void*) merged_graph;
  scheduler->roundNum = 0;
  e = new engine<vdata, edata>(merged_graph, scheduler);

  for (int trial = 0; trial < 5; trial++) {
    global_error_sq = 0.0;
    //global_learning_rate = 0.6/(trial+1);
    global_learning_rate = 0.49;
    std::vector<int> vertex_ids;
    for (int i = 0; i < merged_graph->num_vertices(); i++) {
      vertex_ids.push_back(i);
    } 
    std::random_shuffle(vertex_ids.begin(), vertex_ids.end());
    // pick one section to be "converged"
    std::set<int> section_list;
    for (int _i = 0; _i < merged_graph->num_vertices(); _i++) {
      int i = _i;//vertex_ids[_i];
      //merged_graph->getVertexData(i)->offset_x += (20.0-20.0*(1.0*(rand()%256)/256))/(trial*trial+1);
      //merged_graph->getVertexData(i)->offset_y += (20.0-20.0*(1.0*(rand()%256)/256))/(trial*trial+1);
      int z = merged_graph->getVertexData(i)->z;
      //merged_graph->getVertexData(i)->converged = 0;
      merged_graph->getVertexData(i)->iteration_count = 0;
      if (section_list.find(z) == section_list.end()) {
        if (merged_graph->edgeData[i].size() > 4) {
          section_list.insert(z);
          //merged_graph->getVertexData(i)->converged = 1;
        }
      }
    }

    scheduler->isStatic = false;
    for (int i = 0; i < merged_graph->num_vertices(); i++) {
      scheduler->add_task_static(i, updateVertex2DAlignFULLFast);
    }
    scheduler->isStatic = true;

    printf("starting run\n");
    e->run();

    printf("ending run\n");

    for (int i = 0; i < merged_graph->num_vertices(); i++) {
      merged_graph->getVertexData(i)->iteration_count = 0;
      computeError2DAlign(i, (void*) scheduler);
    }
    printf("Global error sq2 on iter %d is %f\n", trial, global_error_sq);
    break;
  }
}

void compute_tile_matches_active_set(align_data_t *p_align_data, int sec_id, std::set<int> active_set, Graph<vdata, edata>* graph) {
  TRACE_1("compute_tile_matches: start\n");

  //section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);

  // Iterate over all pairs of tiles

  section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);

  int active_set_array [active_set.size()];
  int i = 0;
  for (auto it = active_set.begin(); it != active_set.end(); ++it) {
    active_set_array[i] = *it;
    i++;
  }

  cilk_for (int j = 0; j < i; j++) {
    int atile_id = active_set_array[j];
    //printf("The tile id is %d\n", atile_id);
    if (atile_id >= p_sec_data->n_tiles) {
      printf("Big error!\n");    
    }
    tile_data_t *a_tile = &(p_sec_data->tiles[atile_id]);

    // get all close tiles.
    int indices_to_check[50];
    int indices_to_check_len = get_all_close_tiles(atile_id, p_sec_data, indices_to_check);


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


      float trial_rod;
      for (int trial = 0; trial < 4; trial++) {
        if (trial == 0) trial_rod = 0.7;
        if (trial == 1) trial_rod = 0.8;
        if (trial == 2) trial_rod = 0.92;
        if (trial == 3) trial_rod = 0.96;
        // Match the features
        std::vector< cv::DMatch > matches;
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       trial_rod);

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
          //if (matches.size() == 0) printf("There are zero matches.\n");
          continue;
        }

        bool* mask = (bool*) calloc(match_points_a.size(), 1);
        double thresh = 5.0;
        tfk_simple_ransac(match_points_a, match_points_b, thresh, mask);


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
        if (num_matches_filtered > 12) {
          graph->insert_matches(atile_id, btile_id,
              filtered_match_points_a, filtered_match_points_b, 1.0);
          break;
        }
      }

      // NOTE(TFK): Previously we only output features if we had at least MIN_FEATURES_NUM because
      //              the code reading the match files would add "fake" matches between adjacent tiles
      //              which tended to do better than less than MIN_FEATURES_NUM real matches.
      //if (filtered_match_points_a.size() < MIN_FEATURES_NUM) {
      //  //TRACE_1("Less than %d matched features, saving empty match file\n",
      //  //        MIN_FEATURES_NUM);
      //  continue;
      //}

    } while (false); // end the do while wrapper.
    }  // for (btile_id)
  }  // for (atile_id)
}


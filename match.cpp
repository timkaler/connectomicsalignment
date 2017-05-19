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



static cv::Point2d transform_point_double(vdata* vertex, cv::Point2f point_local) {
  double new_x = point_local.x*vertex->a00 + point_local.y * vertex->a01 + vertex->offset_x + vertex->start_x;
  double new_y = point_local.x*vertex->a10 + point_local.y * vertex->a11 + vertex->offset_y + vertex->start_y;
  //float new_x = point_local.x+vertex->offset_x+vertex->start_x;
  //float new_y = point_local.y+vertex->offset_y+vertex->start_y;
  return cv::Point2d(new_x, new_y);
}

static cv::Point2f transform_point(vdata* vertex, cv::Point2f point_local) {
  float new_x = point_local.x*vertex->a00 + point_local.y * vertex->a01 + vertex->offset_x + vertex->start_x;
  float new_y = point_local.x*vertex->a10 + point_local.y * vertex->a11 + vertex->offset_y + vertex->start_y;
  //float new_x = point_local.x+vertex->offset_x+vertex->start_x;
  //float new_y = point_local.y+vertex->offset_y+vertex->start_y;
  return cv::Point2f(new_x, new_y);
}



std::string get_point_transform_string(Graph<vdata, edata>* merged_graph, vdata* vd) {

  std::string ret = "";  
  int vid = vd->vertex_id;
  int count = 0;
  for (int i = 0; i < vd->my_mesh_points->size(); i++) {
      cv::Point2f my_point = (*(vd->section_data->mesh_orig))[(*(vd->my_mesh_points))[i]];
      cv::Point2f n_point = (*(vd->section_data->mesh))[(*(vd->my_mesh_points))[i]];
      //cv::Point2f n_point = ;
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

  for (size_t pt_idx = 0; pt_idx < a_tile->p_kps_3d->size(); ++pt_idx) {
    //cv::Point2f pt = (*a_tile->p_kps_3d)[pt_idx].pt;
    //if (pt.x < ref_tile->x_start + radius) continue;
    //if (pt.y < ref_tile->y_start + radius) continue;
    //if (pt.x > ref_tile->x_end - radius) continue;
    //if (pt.y > ref_tile->y_end - radius) continue;
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

  //printf("Bounding box is %f %f %f %f\n", _min_x, _min_y, _max_x, _max_y);
  for (size_t pt_idx = 0; pt_idx < a_tile->p_kps_3d->size(); ++pt_idx) {
    cv::Point2f pt = transform_point(vertex_data, (*a_tile->p_kps_3d)[pt_idx].pt);
    cv::KeyPoint kpt = (*a_tile->p_kps_3d)[pt_idx];
    kpt.pt = pt;
    //if (pt.x < _min_x || pt.x > _max_x || pt.y < _min_y || pt.y > _max_y) {
    //  //printf("Filtered: %f,%f\n", pt.x, pt.y);
    //  continue;
    //} else {
    //  //printf("Unfiltered: %f,%f\n", pt.x, pt.y);
    //}
    atile_kps_in_overlap.push_back(kpt);
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


#include "ransac.h" 

static void match_features(std::vector< cv::DMatch > &matches,
                           cv::Mat &descs1, cv::Mat &descs2,
                           float rod);


void updateVertex2DAlign(int vid, void* scheduler_void);
#include "meshoptimize.h"

std::vector<Graph<vdata, edata>* > graph_list;

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
    //NOTE(TFK): I am not sure if this switching of A->B to B->A is correct.
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

__attribute__((const))
static double dist(const cv::Point2f a_pt, const cv::Point2f b_pt) {
  double x_delta = a_pt.x - b_pt.x;
  double y_delta = a_pt.y - b_pt.y;
  return std::sqrt((x_delta * x_delta) + (y_delta * y_delta));
}

#include "match_output.h"

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

void compute_tile_matches(align_data_t *p_align_data, int force_section_id) {
  TRACE_1("compute_tile_matches: start\n");

  //section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);

  // Iterate over all pairs of tiles
  //for (int sec_id = 0; sec_id < p_align_data->n_sections; sec_id++) {
  for (int sec_id = force_section_id; sec_id < force_section_id+1 && force_section_id != -1; sec_id++) {
    Graph<vdata, edata>* graph;
    //Scheduler* scheduler;
    //engine<vdata, edata>* e;
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
          //if (matches.size() == 0) printf("There are zero matches.\n");
          continue;
        }

        bool* mask = (bool*) calloc(match_points_a.size(), 1);
        double thresh = 10.0;
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
        if (num_matches_filtered > 0) {
          graph->insert_matches(atile_id, btile_id,
              filtered_match_points_a, filtered_match_points_b, 1.0);
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


    // Release the memory for keypoints after they've been filtered via matching.
    cilk_for (int atile_id = 0; atile_id < p_sec_data->n_tiles; atile_id++) {
      tile_data_t *a_tile = &(p_sec_data->tiles[atile_id]);
      a_tile->p_kps->clear();
      std::vector<cv::KeyPoint>().swap(*(a_tile->p_kps));
      ((a_tile->p_kps_desc))->release();
    }

    // Initialize data in the graph representation.
    printf("Size of the graph is %d\n", graph->num_vertices());
    for (int i = 0; i < graph->num_vertices(); i++) {
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
      //d->last_radius_value = 9.0;
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


  // Merging the graphs in graph_list into a single merged graph.
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
    //std::priority_queue<std::pair<double, int> > queue;

    //// perform initial enqueue.
    //for (int i = 0; i < merged_graph->num_vertices(); i++) {
    //  serialUpdateVertex2DAlign(i, -1.0, (void*)scheduler, &queue);
    //}

    //while (!queue.empty()) {
    //  std::pair<double, int> vid = queue.top();
    //  queue.pop();
    //  if (vid.first < 1e-3) {
    //    printf("converged\n");
    //    break;
    //  }
    //  serialUpdateVertex2DAlign(vid.second, vid.first, (void*)scheduler, &queue);
    //}
    ////int update_count = 0;
    ////while (!queue.empty()) {
    ////  std::pair<double, int> vid = queue.top();
    ////  queue.pop();
    ////  if (update_count % 100*merged_graph->num_vertices() == 0) {
    ////    global_learning_rate *= 0.99;
    ////    printf("Learning rate %f worst value is %f\n", global_learning_rate, vid.first);
    ////  }
    ////  update_count++; 
    ////  if (vid.first < 1e-3) {
    ////    printf("converged\n");
    ////    break;
    ////  }
    ////  serialUpdateVertex2DAlign(vid.second, vid.first, (void*)scheduler, &queue);
    ////}

    printf("ending run\n");

    //mfov_alignment_3d(merged_graph, p_align_data);
    //scheduler =
    //    new Scheduler(merged_graph->vertexColors, ncolors+1, merged_graph->num_vertices());
    //e = new engine<vdata, edata>(merged_graph, scheduler);
    //scheduler->graph_void = (void*) merged_graph;
    //scheduler->roundNum = 0;
    //scheduler->isStatic = true;
    //for (int i = 0; i < merged_graph->num_vertices(); i++) {
    //  merged_graph->getVertexData(i)->iteration_count = 0;
    //  scheduler->add_task_static(i, updateVertex2DAlignFULL);
    //}
    //printf("starting run\n");
    //e->run();

    for (int i = 0; i < merged_graph->num_vertices(); i++) {
      merged_graph->getVertexData(i)->iteration_count = 0;
      computeError2DAlign(i, (void*) scheduler);
      //scheduler->add_task(i, computeError2DAlign);
    }
    printf("Global error sq2 on iter %d is %f\n", trial, global_error_sq);
    //if (global_error_sq < 2.0*p_align_data->n_sections) break;
    break;
  }
  //e->run();
  #ifdef ALIGN3D


  //for (int i = 0; i < merged_graph->num_vertices(); i++) {
  //  merged_graph->edgeData[i].clear();
  //}
  //// before we align lets compute overlap correctly.
  //cilk_for (int i = 0; i < merged_graph->num_vertices(); i++) {
  //  vdata* vd = merged_graph->getVertexData(i);
  //  double start_x = vd->start_x+vd->offset_x;
  //  double start_y = vd->start_y+vd->offset_y;
  //  double end_x = vd->end_x+vd->offset_x;
  //  double end_y = vd->end_y+vd->offset_y;
  //  for (int j = i+1; j < merged_graph->num_vertices(); j++) {
  //  vdata* vd2 = merged_graph->getVertexData(j);
  //  double start_x2 = vd2->start_x+vd2->offset_x;
  //  double start_y2 = vd2->start_y+vd2->offset_y;
  //  double end_x2 = vd2->end_x+vd2->offset_x;
  //  double end_y2 = vd2->end_y+vd2->offset_y;

  //  double overlap_x_start = std::max(start_x, start_x2);
  //  double overlap_x_end = std::min(end_x, end_x2);
  //  double overlap_y_start = std::max(start_y, start_y2);
  //  double overlap_y_end = std::min(end_y, end_y2);

  //  if (overlap_y_start >= overlap_y_end || overlap_x_start >= overlap_x_end) continue;
  //  if (vd->z != vd2->z) continue;
  //  // otherwise add evenly spaced matches.
  //  std::vector<cv::Point2f> points_a(0), points_b(0);
  //  for (int k = 0; k < 5; k += 1) {
  //    for (int m = 0; m < 5; m += 1) {
  //      float fake_x = overlap_x_start + (overlap_x_end-overlap_x_start)*(k*1.0/5.0 + m*1.0/25.0);
  //      float fake_y = overlap_y_start + (overlap_y_end-overlap_y_start)*(m*1.0/5.0 + k*1.0/25.0);
  //      points_a.push_back(cv::Point2f(fake_x - start_x, fake_y - start_y)); 
  //      points_b.push_back(cv::Point2f(fake_x - start_x2, fake_y - start_y2));
  //      assert(fake_x-start_x>=0 && fake_y-start_y >=0 && fake_x - start_x2 >=0 && fake_y - start_y2 >= 0);
  //    }
  //  }
  //  merged_graph->insert_matches(i,j,points_a, points_b, 1.0);
  //  }
  //}

  //for (int v = 0; v < merged_graph->num_vertices(); v++) {
  //  vdata* vd = merged_graph->getVertexData(v);
  //  vd->corner_points[0] = cv::Point2f(0.0, 0.0);
  //  vd->corner_points[1] = cv::Point2f(0.0, 2724.0);
  //  vd->corner_points[2] = cv::Point2f(3128.0, 0.0);
  //  vd->corner_points[3] = cv::Point2f(3128.0, 2724.0);
  //}

  // Unpack the graphs within the merged graph.
  vertex_id_offset = 0;
  for (int i = 0; i < graph_list.size(); i++) {
    for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
      vdata* d = merged_graph->getVertexData(j+vertex_id_offset);
      *(graph_list[i]->getVertexData(j)) = *d;
      (graph_list[i]->getVertexData(j))->vertex_id -= vertex_id_offset;
    }
    vertex_id_offset += graph_list[i]->num_vertices();
  }
  for (int i = 0; i < graph_list.size(); i++) {
    construct_triangles(graph_list[i], 1500.0);
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

  coarse_alignment_3d(merged_graph, p_align_data, 64.0);
  fine_alignment_3d(merged_graph, p_align_data);

  std::set<int> sections_done;
  sections_done.clear();
  // now transform the mesh points using section transforms.
  for (int v = 0; v < merged_graph->num_vertices(); v++) {
    vdata* vd = merged_graph->getVertexData(v);
    if (sections_done.find(vd->z) == sections_done.end()) {
      printf("Doing an update for section %d\n", vd->z);
      sections_done.insert(vd->z);
      graph_section_data* section_data = vd->section_data;
      std::vector<cv::Point2f>* mesh = section_data->mesh;
      std::vector<cv::Point2f>* mesh_orig = section_data->mesh_orig;
      cv::Mat warp_mat = *(section_data->transform);
      vdata tmp;
      tmp.a00 = warp_mat.at<double>(0, 0); 
      tmp.a01 = warp_mat.at<double>(0, 1);
      tmp.offset_x = warp_mat.at<double>(0, 2);
      tmp.a10 = warp_mat.at<double>(1, 0); 
      tmp.a11 = warp_mat.at<double>(1, 1); 
      tmp.offset_y = warp_mat.at<double>(1, 2);
      tmp.start_x = 0.0;
      tmp.start_y = 0.0;

      for (int mesh_index = 0; mesh_index < mesh->size(); mesh_index++) {
        (*mesh)[mesh_index] = transform_point(&tmp, (*mesh)[mesh_index]);
        (*mesh_orig)[mesh_index] = transform_point(&tmp, (*mesh_orig)[mesh_index]);
      }
    }
  }

  // this is going to give us the matches.
  std::vector<tfkMatch> mesh_matches;
  fine_alignment_3d_2(merged_graph, p_align_data,64.0, mesh_matches);
  printf("The number of mesh matches is %d\n", mesh_matches.size());


  {
    double cross_slice_weight = 1.0;
    double cross_slice_winsor = 20.0;
    double intra_slice_weight = 1.0;
    double intra_slice_winsor = 200.0;
    int max_iterations = 10000;
    double min_stepsize = 1e-20;
    double stepsize = 0.0001;
    std::map<int, graph_section_data> section_data_map;
    section_data_map.clear();
    for (int i = 0; i < mesh_matches.size(); i++) {
      int az = mesh_matches[i].my_section_data.z;
      int bz = mesh_matches[i].n_section_data.z;
      if (section_data_map.find(az) == section_data_map.end()) {
        section_data_map[az] = mesh_matches[i].my_section_data;
      }
      if (section_data_map.find(bz) == section_data_map.end()) {
        section_data_map[bz] = mesh_matches[i].n_section_data;
      }
    }

      for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
           it != section_data_map.end(); ++it) {
        it->second.gradients = new cv::Point2f[it->second.mesh->size()];
        it->second.rest_lengths = new double[it->second.triangle_edges->size()];
        it->second.rest_areas = new double[it->second.triangles->size()];
        it->second.mesh_old = new std::vector<cv::Point2f>();

        for (int j = 0; j < it->second.mesh->size(); j++) {
          it->second.mesh_old->push_back((*(it->second.mesh))[j]);
        }

        // init
        for (int j = 0; j < it->second.mesh->size(); j++) {
          it->second.gradients[j] = cv::Point2f(0.0,0.0);  
        }

        for (int j = 0; j < it->second.triangle_edges->size(); j++) {
          cv::Point2f p1 = (*(it->second.mesh))[(*(it->second.triangle_edges))[j].first];
          cv::Point2f p2 = (*(it->second.mesh))[(*(it->second.triangle_edges))[j].second];
          double dx = p1.x-p2.x;
          double dy = p1.y-p2.y;
          double len = std::sqrt(dx*dx+dy*dy);
          it->second.rest_lengths[j] = len;
          printf("Rest length is %f\n", len);
        }

        // now triangle areas.
        for (int j = 0; j < it->second.triangles->size(); j++) {
          tfkTriangle tri = (*(it->second.triangles))[j];
          cv::Point2f p1 = (*(it->second.mesh))[tri.index1];
          cv::Point2f p2 = (*(it->second.mesh))[tri.index2];
          cv::Point2f p3 = (*(it->second.mesh))[tri.index3];
          it->second.rest_areas[j] = computeTriangleArea(p1,p2,p3);
          printf("Rest area is %f\n", it->second.rest_areas[j]);
        }
      }
    double prev_cost = 0.0; 
    for (int iter = 0; iter < max_iterations; iter++) {
      double cost = 0.0;

      for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
           it != section_data_map.end(); ++it) {
        // clear old gradients.
        for (int j = 0; j < it->second.mesh->size(); j++) {
          it->second.gradients[j] = cv::Point2f(0.0,0.0);  
        }
      }

      // compute internal gradients.
      for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
           it != section_data_map.end(); ++it) {
        // internal_mesh_derivs
        double all_weight = intra_slice_weight; 
        double sigma = intra_slice_winsor; 
        std::vector<cv::Point2f>* mesh = it->second.mesh;
        cv::Point2f* gradients = it->second.gradients;
        std::vector<std::pair<int, int> >* triangle_edges = it->second.triangle_edges;
        std::vector<tfkTriangle >* triangles = it->second.triangles;
        double* rest_lengths = it->second.rest_lengths;
        double* rest_areas = it->second.rest_areas;

        //// update all edges
        for (int j = 0; j < triangle_edges->size(); j++) {
          cost += internal_mesh_derivs(mesh, gradients, (*triangle_edges)[j], rest_lengths[j],
                                       all_weight, sigma);
        }

        //// update all triangles
        for (int j = 0; j < triangles->size(); j++) {
          int triangle_indices[3] = {(*triangles)[j].index1,
                                     (*triangles)[j].index2,
                                     (*triangles)[j].index3};
          cost += area_mesh_derivs(mesh, gradients, triangle_indices, rest_areas[j], all_weight);
        }
      }

      for (int j = 0; j < mesh_matches.size(); j++) {
        std::vector<cv::Point2f>* mesh1 = mesh_matches[j].my_section_data.mesh;
        std::vector<cv::Point2f>* mesh2 = mesh_matches[j].n_section_data.mesh;

        int myz = mesh_matches[j].my_section_data.z;
        int nz = mesh_matches[j].n_section_data.z;

        cv::Point2f* gradients1 = section_data_map[myz].gradients;
        cv::Point2f* gradients2 = section_data_map[nz].gradients;
        double* barys1 = mesh_matches[j].my_barys;
        double* barys2 = mesh_matches[j].n_barys;
        double all_weight = cross_slice_weight;
        double sigma = cross_slice_winsor;

        int indices1[3] = {mesh_matches[j].my_tri.index1,
                           mesh_matches[j].my_tri.index2,
                           mesh_matches[j].my_tri.index3};
        int indices2[3] = {mesh_matches[j].n_tri.index1,
                           mesh_matches[j].n_tri.index2,
                           mesh_matches[j].n_tri.index3};

        cost += crosslink_mesh_derivs(mesh1, mesh2,
                                      gradients1, gradients2,
                                      indices1, indices2,
                                      barys1, barys2,
                                      all_weight, sigma);

        cost += crosslink_mesh_derivs(mesh2, mesh1,
                                      gradients2, gradients1,
                                      indices2, indices1,
                                      barys2, barys1,
                                      all_weight, sigma);
      }

      if (iter == 0) prev_cost = cost+10.0;

      if (cost <= prev_cost) {
        stepsize *= 1.1;
        if (stepsize > 1.0) {
          stepsize = 1.0;
        }
        // TODO(TFK): momentum.

        for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
             it != section_data_map.end(); ++it) {
          std::vector<cv::Point2f>* mesh = it->second.mesh;
          std::vector<cv::Point2f>* mesh_old = it->second.mesh_old;
          cv::Point2f* gradients = it->second.gradients;
          for (int j = 0; j < mesh->size(); j++) {
            (*mesh_old)[j] = ((*mesh)[j]);
          }
          for (int j = 0; j < mesh->size(); j++) {
            (*mesh)[j].x -= (float)(stepsize * (gradients)[j].x); 
            (*mesh)[j].y -= (float)(stepsize * (gradients)[j].y); 
          }
        }
        printf("Good step old cost %f, new cost %f, iteration %d\n", prev_cost, cost, iter);
        prev_cost = cost;
      } else {
        stepsize *= 0.5;
        // bad step undo.
        for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
             it != section_data_map.end(); ++it) {
          std::vector<cv::Point2f>* mesh = it->second.mesh;
          std::vector<cv::Point2f>* mesh_old = it->second.mesh_old;
          //if (mesh_old->size() != mesh->size()) continue;
          for (int j = 0; j < mesh->size(); j++) {
            (*mesh)[j] = (*mesh_old)[j];
          }
        }
        //printf("Bad step old cost %f, new cost %f\n", prev_cost, cost);
      }
    }
      
  } // END BLOCK.
    
    

  //}
//  exit(0);

  //fine_alignment_3d_2(merged_graph, p_align_data,128.0);
  //fine_alignment_3d_2(merged_graph, p_align_data,32.0);
  //fine_alignment_3d_2(merged_graph, p_align_data,16.0);

  //// add matches above and below.
  //cilk_for (int va = 0; va < merged_graph->num_vertices(); va++) {
  //  vdata* vda = merged_graph->getVertexData(va);
  //  cv::Point2f centera = transform_point(vda, vda->original_center_point);


  //  for (int vb = va+1; vb < merged_graph->num_vertices(); vb++) {
  //    vdata* vdb = merged_graph->getVertexData(vb);
  //    if (vdb->z != vda->z-1 && vdb->z != vda->z+1) {
  //      continue; 
  //    }
  //    cv::Point2f centerb = transform_point(vdb, vdb->original_center_point);
  //    std::vector<cv::Point2f> points_b(0);
  //    std::vector<cv::Point2f> points_a(0);

  //    // center of point a relative to the btile.
  //    cv::Point2f centera_rb = invert_transform_point(vdb, centera);
  //    cv::Point2f centerb_ra = invert_transform_point(vda, centerb);

  //    float dx = (centera_rb.x - vdb->original_center_point.x);
  //    float dy = (centera_rb.y - vdb->original_center_point.y);
  //    float dist = dx*dx+dy*dy;
  //    if (dist > 1500.0*1500.0) continue;
  //    dx = (centerb_ra.x - vda->original_center_point.x);
  //    dy = (centerb_ra.y - vda->original_center_point.y);
  //    dist = dx*dx+dy*dy;
  //    if (dist > 1500.0*1500.0) continue;

  //    //if (isnan(centera_rb.x) || isnan(centera_rb.y)) continue;
  //    //if (isnan(centerb_ra.x) || isnan(centerb_ra.y)) continue;

  //    //if ((centera_rb.x >0 && centera_rb.x < 2700) &&
  //    //     (centera_rb.y>0 && centera_rb.y < 00)) {
  //    //   points_a.push_back(vda->original_center_point);
  //    //   points_b.push_back(centera_rb);
  //    //}

  //    points_a.push_back(centerb_ra);
  //    points_b.push_back(vdb->original_center_point);

  //    merged_graph->insert_matches(va,vb,points_a,points_b, 1.0);
  //  }
  //}

  //ncolors = merged_graph->compute_trivial_coloring();
  //scheduler =
  //    new Scheduler(merged_graph->vertexColors, ncolors+1, merged_graph->num_vertices());
  //scheduler->graph_void = (void*) merged_graph;
  //e = new engine<vdata, edata>(merged_graph, scheduler);

  //// undo the affine transforms.
  //for (int v = 0; v < merged_graph->num_vertices(); v++) {
  //  merged_graph->getVertexData(v)->a11 = 1.0;
  //  merged_graph->getVertexData(v)->a00 = 1.0;
  //  merged_graph->getVertexData(v)->a01 = 0.0;
  //  merged_graph->getVertexData(v)->a10 = 0.0;
  //}

  //for (int trial = 0; trial < 2; trial++) {
  //  global_error_sq = 0.0; 
  //  //global_learning_rate = 0.6/(trial+1);
  //  global_learning_rate = 0.9;
  //  std::vector<int> vertex_ids;
  //  for (int i = 0; i < merged_graph->num_vertices(); i++) {
  //    vertex_ids.push_back(i);
  //  } 
  //  std::random_shuffle(vertex_ids.begin(), vertex_ids.end());
  //  // pick one section to be "converged"
  //  std::set<int> section_list;
  //  for (int _i = 0; _i < merged_graph->num_vertices(); _i++) {
  //    int i = _i;//vertex_ids[_i];
  //    //merged_graph->getVertexData(i)->offset_x += (20.0-20.0*(1.0*(rand()%256)/256))/(trial*trial+1);
  //    //merged_graph->getVertexData(i)->offset_y += (20.0-20.0*(1.0*(rand()%256)/256))/(trial*trial+1);
  //    int z = merged_graph->getVertexData(i)->z;
  //    merged_graph->getVertexData(i)->converged = 0;
  //    merged_graph->getVertexData(i)->iteration_count = 0;
  //    if (section_list.find(z) == section_list.end()) {
  //      if (merged_graph->edgeData[i].size() > 4) {
  //        section_list.insert(z);
  //        merged_graph->getVertexData(i)->converged = 1;
  //      }
  //    }
  //  }
  //  for (int i = 0; i < merged_graph->num_vertices(); i++) {
  //    scheduler->add_task(i, updateVertex2DAlignFULL);
  //  }
  //  printf("starting run\n");
  //  e->run();
  //  printf("ending run\n");

  //  for (int i = 0; i < merged_graph->num_vertices(); i++) {
  //    merged_graph->getVertexData(i)->iteration_count = 0;
  //    computeError2DAlign(i, (void*) scheduler);
  //    //scheduler->add_task(i, computeError2DAlign);
  //  }
  //  printf("Global error sq2 on iter %d is %f\n", trial, global_error_sq);
  //  if (global_error_sq < 2.0*p_align_data->n_sections) break;
  ////  fine_alignment_3d(merged_graph, p_align_data);
  //}
  ////coarse_alignment_3d(merged_graph, p_align_data, 64.0);
  ////fine_alignment_3d(merged_graph, p_align_data);
  ////fine_alignment_3d_dampen(merged_graph, p_align_data);
  #endif






  // Unpack the graphs within the merged graph.
  vertex_id_offset = 0;
  for (int i = 0; i < graph_list.size(); i++) {
    for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
      vdata* d = merged_graph->getVertexData(j+vertex_id_offset);
      *(graph_list[i]->getVertexData(j)) = *d;
      (graph_list[i]->getVertexData(j))->vertex_id -= vertex_id_offset;
    }
    vertex_id_offset += graph_list[i]->num_vertices();
  }


  for (int _i = 0; _i < graph_list.size(); _i++) {
    Graph<vdata, edata>* graph = graph_list[_i];
    int sec_id = graph->section_id;
    std::string section_id_string =
        std::to_string(p_align_data->sec_data[sec_id].section_id +
        p_align_data->base_section+1);


    //std::string transform_string = "";
    //for (int i = 0; i < graph->num_vertices(); i++) {
    //  if (i == 0) {
    //    transform_string += get_point_transform_string(graph, graph->getVertexData(i));
    //  } else {
    //    transform_string += " " + get_point_transform_string(graph, graph->getVertexData(i));
    //  }
    //  
    //}

    FILE* wafer_file = fopen((std::string(p_align_data->output_dirpath)+std::string("/W01_Sec") +
        matchPadTo(section_id_string, 3)+std::string("_montaged.json")).c_str(), "w+");
    fprintf(wafer_file, "[\n");

    

    for (int i = 0; i < graph->num_vertices(); i++) {
      //printf("affine params %f %f %f %f\n", graph->getVertexData(i)->a00, graph->getVertexData(i)->a01, graph->getVertexData(i)->a10, graph->getVertexData(i)->a11);
      vdata* vd = graph->getVertexData(i);
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].x_start = vd->start_x+vd->offset_x;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].x_finish = vd->end_x + vd->offset_x;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].y_start = vd->start_y+vd->offset_y;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].y_finish = vd->end_y + vd->offset_y;
      //if (vd->z == 1) {
      //vd->a00 = 1.1;
      //vd->a01 = 0.0;
      //vd->a10 = 0.0;
      //vd->a11 = 1.1;
      //}

      //cv::Point2f corner_points_transformed[4];
      //for (int t = 0; t < 4; t++) {
      //  corner_points_transformed[t] = vd->corner_points[t];//transform_point(vd, vd->corner_points[t]);
      //}
      //double min_x = corner_points_transformed[0].x; 
      //double max_x = corner_points_transformed[0].x; 
      //double min_y = corner_points_transformed[0].y; 
      //double max_y = corner_points_transformed[0].y;
      //for (int t = 1; t < 4; t++) {
      //  if (corner_points_transformed[t].x < min_x) {
      //    min_x = corner_points_transformed[t].x;
      //  }
      //  if (corner_points_transformed[t].y < min_y) {
      //    min_y = corner_points_transformed[t].y;
      //  }
      //  if (corner_points_transformed[t].x > max_x) {
      //    max_x = corner_points_transformed[t].x;
      //  }
      //  if (corner_points_transformed[t].y > max_y) {
      //    max_y = corner_points_transformed[t].y;
      //  }
      //}

      fprintf(wafer_file, "\t{\n");
      fprintf(wafer_file, "\t\t\"bbox\": [\n");

      fprintf(wafer_file,
          "\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f\n],",
          vd->start_x+vd->offset_x, (vd->end_x+vd->offset_x),
          vd->start_y+vd->offset_y, (vd->end_y+vd->offset_y));
      //fprintf(wafer_file,
      //    "\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f\n],",
      //    min_x, max_x,
      //    min_y, max_y);

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
      // {'className': 'mpicbg.trakem2.transform.AffineModel2D', 'dataString': '0.1 0.0 0.0 0.1 0.0 0.0'}

      fprintf(wafer_file, "\t\t\t{\n");
      fprintf(wafer_file,
          "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.AffineModel2D\",\n");
      fprintf(wafer_file,
          "\t\t\t\t\"dataString\": \"%f %f %f %f %f %f\"\n", vd->a00,
          vd->a10, vd->a01, vd->a11, vd->start_x+vd->offset_x, vd->start_y+vd->offset_y);
      //fprintf(wafer_file,
      //    "\t\t\t\t\"dataString\": \"%f %f %f %f %f %f\"\n", 1.0,
      //    0.0, 0.0, 1.0, 0.0, 0.0);

      //fprintf(wafer_file,
      //    "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.RigidModel2D\",\n");
      //fprintf(wafer_file,
      //    "\t\t\t\t\"dataString\": \"%f %f %f\"\n", 0.0,
      //    vd->start_x+vd->offset_x, vd->start_y + vd->offset_y);
      //if (false && (vd->boundary || true)) {
      if (true) {
      fprintf(wafer_file,
          "\t\t\t},\n");

      fprintf(wafer_file, "\t\t\t{\n");
      fprintf(wafer_file,
          "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.PointsTransformModel\",\n");
      fprintf(wafer_file,
          "\t\t\t\t\"dataString\": \"%s\"\n", get_point_transform_string(graph, vd).c_str());
      //fprintf(wafer_file,
      //    "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.RigidModel2D\",\n");
      //fprintf(wafer_file,
      //    "\t\t\t\t\"dataString\": \"%f %f %f\"\n", 0.0,
      //    vd->start_x+vd->offset_x, vd->start_y + vd->offset_y);
      fprintf(wafer_file,
          "\t\t\t}\n");
      } else {
       fprintf(wafer_file,
          "\t\t\t}\n");
      }


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
    //delete graph;
    //delete scheduler;
    //delete e;
  }

  //TRACE_1("compute_tile_matches: finish\n");
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
//<<<<<<< HEAD


//  std::set<int> active_and_neighbors;
//  active_and_neighbors.insert(active_set.begin(), active_set.end());
//  active_and_neighbors.insert(neighbor_set.begin(), neighbor_set.end());
//  // Release the memory for keypoints after they've been filtered via matching.
//  for (auto it = active_and_neighbors.begin(); it != active_and_neighbors.end(); ++it) {
//    int atile_id = *it;
//    tile_data_t *a_tile = &(p_sec_data->tiles[atile_id]);
//    a_tile->p_kps->clear();
//    std::vector<cv::KeyPoint>().swap(*(a_tile->p_kps));
//    ((a_tile->p_kps_desc))->release();
//  }
}
//=======
//}
//>>>>>>> c14d91436faca55e27ea0ec2883fcfc36ad79a19

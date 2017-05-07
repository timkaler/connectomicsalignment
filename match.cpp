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
#include "./simple_mutex.h"
#include "cilk_tools/engine.h"

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
      merged_graph->getVertexData(i)->converged = 0;
      merged_graph->getVertexData(i)->iteration_count = 0;
      if (section_list.find(z) == section_list.end()) {
        if (merged_graph->edgeData[i].size() > 4) {
          section_list.insert(z);
          merged_graph->getVertexData(i)->converged = 1;
        }
      }
    }

    for (int i = 0; i < merged_graph->num_vertices(); i++) {
      scheduler->add_task(i, updateVertex2DAlign);
    }
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

    mfov_alignment_3d(merged_graph, p_align_data);

    for (int i = 0; i < merged_graph->num_vertices(); i++) {
      merged_graph->getVertexData(i)->iteration_count = 0;
      scheduler->add_task(i, updateVertex2DAlignFULL);
    }
    printf("starting run\n");
    e->run();

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
  coarse_alignment_3d(merged_graph, p_align_data, 64.0);
  fine_alignment_3d_mfov(merged_graph, p_align_data);
  coarse_alignment_3d(merged_graph, p_align_data, 64.0);
  fine_alignment_3d_mfov(merged_graph, p_align_data);
  //fine_alignment_3d_mfov(merged_graph, p_align_data);
  //fine_alignment_3d_mfov(merged_graph, p_align_data);
  //fine_alignment_3d_mfov(merged_graph, p_align_data);
  //fine_alignment_3d_mfov(merged_graph, p_align_data);
  //mfov_alignment_3d(merged_graph, p_align_data);
  //mfov_alignment_3d(merged_graph, p_align_data);
  //coarse_alignment_3d(merged_graph, p_align_data, 64.0);
  //fine_alignment_3d_mfov(merged_graph, p_align_data);

  //fine_alignment_3d(merged_graph, p_align_data);
  //for (int trial = 0; trial < 5; trial++) {
  //  //fine_alignment_3d(merged_graph, p_align_data);
  //  //merged_graph->getVertexData(i)->iteration_count = 0;
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
  //    scheduler->add_task(i, updateVertex2DAlign);
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
  //fine_alignment_3d_dampen(merged_graph, p_align_data);
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
          vd->a10, vd->a01, vd->a11, vd->start_x+vd->offset_x, vd->start_y+vd->offset_y);

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

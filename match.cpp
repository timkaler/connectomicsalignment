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
#include "./common.h"
#include "./align.h"
#include "./fasttime.h"
#include "./simple_mutex.h"
#include "./Graph.cpp"


void updateVertex2DAlign(int vid, void* scheduler_void);

#include "cilk_tools/scheduler.cpp"
#include "cilk_tools/engine.cpp"

#ifndef SKIPJSON
#define SAVE_TILE_MATCHES(...) save_tile_matches(__VA_ARGS__)
#else
#define SAVE_TILE_MATCHES(...)
#endif

static Graph<vdata, edata>* graph;
static Scheduler* scheduler;
static engine<vdata, edata>* e;
int global_lock = 0;
void updateVertex2DAlign(int vid, void* scheduler_void) {

  //Scheduler* scheduler2 = reinterpret_cast<Scheduler*>(scheduler_void);

  //printf("starting vertex %d\n", vid);
  vdata* vertex_data = graph->getVertexData(vid);
  std::vector<edata> edges = graph->edgeData[vid];


  float original_offset_x = vertex_data->offset_x;
  float original_offset_y = vertex_data->offset_y;

  std::vector<cv::Point2f> filtered_match_points_a;
  std::vector<cv::Point2f> filtered_match_points_b;
  //printf("current dx %f, current dy %f\n", current_dx, current_dy);
  // for each neighboring tile.
  //printf("edges size %d\n", edges.size());
  for (int i = 0; i < edges.size(); i++) {
    std::vector<cv::Point2f> source_points(0), dest_points(0);
    std::vector<cv::Point2f>* v_points = edges[i].v_points;
    std::vector<cv::Point2f>* n_points = edges[i].n_points;

  float current_dx = vertex_data->offset_x + vertex_data->start_x;
  float current_dy = vertex_data->offset_y + vertex_data->start_y;
    vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);

    float neighbor_dx = neighbor_vertex->offset_x + neighbor_vertex->start_x;
    float neighbor_dy = neighbor_vertex->offset_y + neighbor_vertex->start_y;

    //printf("points size %d\n", v_points->size());
    for (int j = 0; j < v_points->size(); j++) {
      cv::Point2f ptx1((*v_points)[j].x + current_dx,
          (*v_points)[j].y + current_dy);
      cv::Point2f ptx2((*n_points)[j].x + neighbor_dx,
          (*n_points)[j].y + neighbor_dy);
      source_points.push_back(ptx1);
      dest_points.push_back(ptx2);
    }

  if (source_points.size() == 0) {
    //printf("edges size is 0.\n");
    continue;
  }

  int failures = 0;
  int count = 0;

 
  cv::Size cv_size(1,1);
    cv::videostab::TranslationBasedLocalOutlierRejector outReject;
    // NOTE(TFK): eps and prob don't appear to be used in outReject.
    // cv::RansacParams rParams =
    //     cv::RansacParams(int size, float thresh, float eps, float prob)
  while (count < 4&&failures<4000) {
    count = 0;
    cv::Mat mask(1,source_points.size(), CV_8U);
    //cv::videostab::TranslationBasedLocalOutlierRejector* outReject =
    //  new cv::videostab::TranslationBasedLocalOutlierRejector();
    failures++;
    cv::videostab::RansacParams rParams =
        cv::videostab::RansacParams(4,
        1.0 + failures*0.5, 0.3, 0.99);

    outReject.setRansacParams(rParams);

    outReject.process(cv_size, source_points, dest_points, mask);

    float average_dx = 0;
    float average_dy = 0;
    // Use the output mask to filter the matches
    for (int k = 0; k < source_points.size(); ++k) {
      if (mask.at<bool>(0,k)) {
        average_dx += dest_points[k].x - source_points[k].x;
        average_dy += dest_points[k].y - source_points[k].y;
        count++;
      }
    }

    if (count >= 4) {

      for (int k = 0; k < source_points.size(); ++k) {
        if (mask.at<bool>(0,k)) {
          filtered_match_points_a.push_back(source_points[k]);
          filtered_match_points_b.push_back(dest_points[k]);
        }
      }

      //average_dx = average_dx / count;
      //average_dy = average_dy / count;
      //vertex_data->offset_x += average_dx*0.1;
      //vertex_data->offset_y += average_dy*0.1;
      //printf("inliers for vertex %d count is %d\n", vid, count);
    } else {
      //printf("No more inliers for vertex %d\n", vid);
    }
  }
}

    if (filtered_match_points_a.size() == 0) {
      return;
    }
          cv::Mat model = cv::estimateRigidTransform(filtered_match_points_a,
              filtered_match_points_b, false);

          if (!model.empty()) {
          std::vector<cv::Point2f> match_points_a_fixed;
          match_points_a_fixed.reserve(filtered_match_points_a.size());
          cv::transform(filtered_match_points_a, match_points_a_fixed, model);

          double delta_x = 0.0;
          double delta_y = 0.0;
          int tempcount = 0.0;
          for (int i = 0; i < filtered_match_points_a.size(); i++) {
            delta_x += match_points_a_fixed[i].x - filtered_match_points_a[i].x;
            delta_y += match_points_a_fixed[i].y - filtered_match_points_a[i].y;
            tempcount++;
          }
          if (tempcount > 0) {
            delta_x = delta_x/tempcount;
            delta_y = delta_y/tempcount;
            vertex_data->offset_x += delta_x;
            vertex_data->offset_y += delta_y;
          }
          }


  if (std::abs(original_offset_x - vertex_data->offset_x) > 1.0 ||
      std::abs(original_offset_y - vertex_data->offset_y) > 1.0){
    //printf("current error at vertex %d is %f, %f\n", (int)vid,
    //    (float)std::abs(original_offset_x - vertex_data->offset_x),
    //    (float)std::abs(original_offset_y - vertex_data->offset_y));
  }
  //if (vid==17){
  //if (vid==17){
    //simple_acquire(&global_lock);
    ////printf("current error at vertex %d is %f, %f\n", (int)vid,
    ////    (float)std::abs(original_offset_x - vertex_data->offset_x),
    ////    (float)std::abs(original_offset_y - vertex_data->offset_y));
    //simple_release(&global_lock);
  //}

  if (
      std::abs(original_offset_x - vertex_data->offset_x) > 1e-2 ||
      std::abs(original_offset_y - vertex_data->offset_y) > 1e-2) {
    scheduler->add_task(vid, updateVertex2DAlign);
    for (int i = 0; i < edges.size(); i++) {
      scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
    }
    vertex_data->iteration_count++;
  }

  //printf("updating vertex %d, %d\n", vid, vertex_data->iteration_count);
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
  cv::BFMatcher matcher(cv::NORM_L2, false);
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

void compute_tile_matches(align_data_t *p_align_data) {
  TRACE_1("compute_tile_matches: start\n");

#ifndef SKIPJSON
  char out_filepath_base[strlen(p_align_data->output_dirpath) +
                         strlen("/matched_sifts/W01_Sec") + 1];
  sprintf(out_filepath_base, "%s/matched_sifts/W01_Sec",
          p_align_data->output_dirpath);
#endif
  // Iterate over all pairs of tiles
  for (int sec_id = 0; sec_id < p_align_data->n_sections; ++sec_id) {
    section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
#ifndef SKIPJSON
    char out_filepath_start[MAX_FILEPATH] = "\0";
    // Get the section id from the first tile.
    const char *tmp_filepath = (p_sec_data->tiles[0]).filepath;
    const char *section_id_start = strrchr(tmp_filepath, '/') + 1;
    const char *section_id_end = strchr(section_id_start, '_');
    size_t section_id_size = (section_id_end - section_id_start)/sizeof(char);
    char section_id[MAX_FILEPATH];
    strncpy(section_id, section_id_start, section_id_size);
    section_id[section_id_size] = '\0';

    // Create the inter-mfov output directory
    sprintf(out_filepath_start, "%s%s/", out_filepath_base, section_id);
    {
      char mkdir_cmd[MAX_FILEPATH];
      sprintf(mkdir_cmd, "mkdir -p %sinter/", out_filepath_start);
      system(mkdir_cmd);
    }
    // Create a new vector of output file names.
    cilk::reducer_list_append< const char* > output_files_reducer;
#endif

    // Record the set of mfov ID's encountered.  Right now, this set
    // is used to limit the number of system calls performed.
    std::set<int> mfovs;
    simple_mutex_t mfovs_lock;
    simple_mutex_init(&mfovs_lock);


    graph = new Graph<vdata, edata>();
    graph->resize(p_sec_data->n_tiles);

    cilk_for (int atile_id = 0; atile_id < p_sec_data->n_tiles; ++atile_id) {
      tile_data_t *a_tile = &(p_sec_data->tiles[atile_id]);
      simple_acquire(&mfovs_lock);
      if (mfovs.insert(a_tile->mfov_id).second) {
        // Encountering a brand new mfov.
#ifndef SKIPJSON
        // Create the output directory.
        char mkdir_cmd[MAX_FILEPATH];
        sprintf(mkdir_cmd, "mkdir -p %sintra/%d/",
                out_filepath_start, a_tile->mfov_id);
        system(mkdir_cmd);
#endif
      }
      simple_release(&mfovs_lock);
#ifndef SKIPJSON
      // Compute starts of output file path and output file name.
      const char *a_filepath = a_tile->filepath;
      const char *a_imagename_start = strrchr(a_filepath, '/') + 1;
      const char *a_imagename_end = strrchr(a_imagename_start, '.');
      size_t a_imagename_size =
          (a_imagename_end - a_imagename_start)/sizeof(char);
      char a_imagename[MAX_FILEPATH];
      strncpy(a_imagename, a_imagename_start, a_imagename_size);
      a_imagename[a_imagename_size] = '\0';
      char out_filename_start[MAX_FILEPATH];
      sprintf(out_filename_start, "W01_Sec%s_sift_matches_%s_",
              section_id, a_imagename);
#endif
      int indices_to_check[50];
      int indices_to_check_len = 0;

      // get all close tiles.
      for (int i = atile_id+1; i < p_sec_data->n_tiles; i++) {
        tile_data_t *b_tile = &(p_sec_data->tiles[i]);
        // Skip tiles that don't overlap
        if (!is_tiles_overlap(a_tile, b_tile)) continue;
        indices_to_check[indices_to_check_len++] = i;
        assert(indices_to_check_len <= 49);
      }

      cilk_for (int tmpindex = 0; tmpindex < indices_to_check_len; tmpindex++) {
        // do {
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
#ifndef SKIPJSON
        const char *b_filepath = b_tile->filepath;
        const char *b_imagename_start = strrchr(b_filepath, '/') + 1;
        const char *b_imagename_end = strrchr(b_imagename_start, '.');
        size_t b_imagename_size =
            (b_imagename_end - b_imagename_start)/sizeof(char);
        char b_imagename[MAX_FILEPATH];
        strncpy(b_imagename, b_imagename_start, b_imagename_size);
        b_imagename[b_imagename_size] = '\0';
        char out_filepath[MAX_FILEPATH];
        if (a_tile->mfov_id == b_tile->mfov_id) {
          // Intra-mfov job
          sprintf(out_filepath, "%sintra/%d/%s%s.json",
                  out_filepath_start, a_tile->mfov_id, out_filename_start,
                  b_imagename);
        } else {
          // Inter-mfov job
          sprintf(out_filepath, "%sinter/%s%s.json",
                  out_filepath_start, out_filename_start, b_imagename);
        }

        char *output_file_buf = new char[strlen(out_filepath) + 7];
        sprintf(output_file_buf, "file://%s", out_filepath);
        output_files_reducer.push_back(output_file_buf);
#endif

        // Check that both tiles have enough features to match.
        if (a_tile->p_kps->size() < MIN_FEATURES_NUM) {
          //TRACE_1("Less than %d features in tile %d_%d, saving empty match file\n",
          //        MIN_FEATURES_NUM,
          //        a_tile->mfov_id, a_tile->index);
          #ifndef SKIPJSON
          SAVE_TILE_MATCHES(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          #endif
          continue;
        }
        if (b_tile->p_kps->size() < MIN_FEATURES_NUM) {
          //TRACE_1("Less than %d features in tile %d_%d, saving empty match file\n",
          //        MIN_FEATURES_NUM,
          //        b_tile->mfov_id, b_tile->index);
          #ifndef SKIPJSON
          SAVE_TILE_MATCHES(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          #endif
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
          #ifndef SKIPJSON
          SAVE_TILE_MATCHES(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          #endif
          continue;
        }
        if (btile_kps_in_overlap.size() < MIN_FEATURES_NUM) {
          //TRACE_1("Less than %d features in the overlap in tile %d_%d, saving empty match file\n",
          //        MIN_FEATURES_NUM,
          //        b_tile->mfov_id, b_tile->index);
          #ifndef SKIPJSON
          SAVE_TILE_MATCHES(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          #endif
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
          #ifndef SKIPJSON
          SAVE_TILE_MATCHES(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          #endif
          continue;
        }

          cv::Mat mask(matches.size(), 1, CV_8U);
          //cv::Mat mask(source_points.size(),1, CV_8U);
          // cv::Mat mask;
          // cv::Mat H = cv::findHomography(match_points_a2, match_points_b2, cv::RANSAC,
          //                               MAX_EPSILON, mask);

          cv::videostab::TranslationBasedLocalOutlierRejector outReject;

          // NOTE(TFK): eps and prob don't appear to be used in outReject.
          // cv::RansacParams rParams =
          //     cv::RansacParams(int size, float thresh, float eps, float prob)
          cv::videostab::RansacParams rParams =
              cv::videostab::RansacParams(MIN_FEATURES_NUM, 50.0, 0.3, 0.99);
          outReject.setRansacParams(rParams);
          outReject.process(cv::Size(1, 1),  // One cell.
                            match_points_a, match_points_b, mask);

        std::vector< cv::Point2f > filtered_match_points_a;
        std::vector< cv::Point2f > filtered_match_points_b;
        filtered_match_points_a.reserve(matches.size());
        filtered_match_points_b.reserve(matches.size());

        // Use the output mask to filter the matches
        for (size_t i = 0; i < matches.size(); ++i) {
          if (mask.at<bool>(0,i)) {
            filtered_match_points_a.push_back(
                atile_kps_in_overlap[matches[i].queryIdx].pt);
            filtered_match_points_b.push_back(
                btile_kps_in_overlap[matches[i].trainIdx].pt);
          }
        }

        graph->insert_matches(atile_id, btile_id,
            filtered_match_points_a, filtered_match_points_b);

        //TRACE_1("    -- [%d_%d, %d_%d] filtered_matches: %lu\n",
        //        a_tile->mfov_id, a_tile->index,
        //        b_tile->mfov_id, b_tile->index,
        //        filtered_match_points_a.size());

        if (filtered_match_points_a.size() < MIN_FEATURES_NUM) {
          //TRACE_1("Less than %d matched features, saving empty match file\n",
          //        MIN_FEATURES_NUM);
          SAVE_TILE_MATCHES(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
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
            #ifndef SKIPJSON
            SAVE_TILE_MATCHES(0, out_filepath,
                              a_tile, b_tile,
                              nullptr, nullptr, nullptr);
            #endif
            continue;
          }

          // Transform the matched points in tile A.  These
          // transformed points are used to estimate the distances
          // between matched points after alignment.
          std::vector< cv::Point2f > match_points_a_fixed;
          match_points_a_fixed.reserve(filtered_match_points_a.size());

          cv::transform(filtered_match_points_a, match_points_a_fixed, model);

          // Output the tile matches.
          #ifndef SKIPJSON
          SAVE_TILE_MATCHES(filtered_match_points_a.size(), out_filepath,
                            a_tile, b_tile,
                            &filtered_match_points_a, &filtered_match_points_b,
                            &match_points_a_fixed);
          #endif
        }
      }  // for (btile_id)
    }  // for (atile_id)

#ifndef SKIPJSON
    // Output list of matched sift files.
    {
      const std::string matched_sifts_files =
        std::string(p_align_data->output_dirpath) +
        "/W01_Sec" + section_id + "_matched_sifts_files.txt";
      //TRACE_1("Recording matched sifts files in %s\n",
      //        matched_sifts_files.c_str());

      FILE *fp = fopen(matched_sifts_files.c_str(), "wb+");
      const std::list< const char* > &output_files =
          output_files_reducer.get_value();
      for (const char *output_file : output_files)
        fprintf(fp, "%s\n", output_file);
      fclose(fp);
    }
#endif

  printf("Size of the graph is %d\n", graph->num_vertices());
  int ncolors = graph->compute_trivial_coloring();
  //for (int i = 0; i < graph->num_vertices(); i++) {
  //  printf("Vertex %d has edges %d\n", i, graph->edgeData[i].size());
  //}
  for (int i = 0; i < graph->num_vertices(); i++) {
    //printf("init vertex data for vertex %d\n", i);
    vdata* d = graph->getVertexData(i);
    _tile_data tdata = p_sec_data->tiles[i];
    d->vertex_id = i;
    d->mfov_id = tdata.mfov_id;
    d->tile_index = tdata.index;
    d->start_x = tdata.x_start;
    d->end_x = tdata.x_finish;
    d->start_y = tdata.y_start;
    d->end_y = tdata.y_finish;
    d->offset_x = 0.0;
    d->offset_y = 0.0;
    d->iteration_count = 0;
  }

  scheduler =
      new Scheduler(graph->vertexColors, ncolors+1, graph->num_vertices());
  e = new engine<vdata, edata>(graph, scheduler);
  for (int i = 0; i < graph->num_vertices(); i++) {
    scheduler->add_task(i, updateVertex2DAlign);
  }
  printf("starting run\n");
  e->run();
  printf("ending run\n");

  //void compute_tile_matches(align_data_t *p_align_data) {
  std::string section_id_string =
      std::to_string(p_align_data->sec_data[sec_id].section_id +
      p_align_data->base_section+1);

  FILE* wafer_file = fopen((std::string("W01_Sec0") +
      section_id_string+std::string("_montaged.json")).c_str(), "w+");
  fprintf(wafer_file, "[\n");
  for (int i = 0; i < graph->num_vertices(); i++) {
    vdata* vd = graph->getVertexData(i);
    fprintf(wafer_file, "\t{\n");
    fprintf(wafer_file, "\t\t\"bbox\": [\n");
    fprintf(wafer_file,
        "\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f\n],",
        vd->start_x+vd->offset_x, vd->end_x+vd->offset_x,
        vd->start_y+vd->offset_y, vd->end_y+vd->offset_y);
    fprintf(wafer_file, "\t\t\"height\": %d,\n",2724);
    fprintf(wafer_file, "\t\t\"layer\": %d,\n",10);
    fprintf(wafer_file, "\t\t\"maxIntensity\": %f,\n",255.0);
    fprintf(wafer_file, "\t\t\"mfov\": %d,\n",
        graph->getVertexData(i)->mfov_id);
    fprintf(wafer_file, "\t\t\"minIntensity\": %f,\n",
        0.0);
    fprintf(wafer_file, "\t\t\"tile_index\": %d,\n",
        graph->getVertexData(i)->tile_index);
    fprintf(wafer_file, "\t\t\"transforms\": [\n");
    fprintf(wafer_file, "\t\t\t{\n");
    fprintf(wafer_file,
        "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.RigidModel2D\",\n");
    fprintf(wafer_file,
        "\t\t\t\t\"dataString\": \"%f %f %f\"\n", 0.0,
        vd->start_x+vd->offset_x, vd->start_y + vd->offset_y);
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

  }  // for (sec_id)

  //TRACE_1("compute_tile_matches: finish\n");
}

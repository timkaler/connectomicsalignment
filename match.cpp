/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include "match.h"
#include "common.h"
#include "align.h"
#include "fasttime.h"
#include "simple_mutex.h"

#include <set>
#include <mutex>
#include <cilk/cilk.h>
#include <cilk/reducer_list.h>

/////////////////////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////

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
  // cv::BFMatcher matcher(cv::NORM_L2, false);
  cv::BFMatcher matcher;
  // cv::FlannBasedMatcher matcher(new cv::flann::AutotunedIndexParams(
  //                 0.9,
  //                 0.01,
  //                 0,
  //                 0.1));
  // cv::FlannBasedMatcher matcher;
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
  thread_local FILE *fp;

  TRACE_1("save_tile_matches: start\n");

  TRACE_1("Writing %s\n", out_filepath);

  fasttime_t tstart = gettime();

  fp = fopen(out_filepath, "wb");
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

  // TODO: Output model
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

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////

// thread_local std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;
// thread_local std::vector< cv::Mat > atile_kps_desc_in_overlap_list, btile_kps_desc_in_overlap_list;
// thread_local std::vector< cv::DMatch > matches;
// thread_local std::vector< cv::Point2f > filtered_match_points_a, filtered_match_points_b;
// thread_local std::vector< cv::Point2f > match_points_a, match_points_b;
// thread_local std::vector< cv::Point2f > match_points_a_fixed;

void compute_tile_matches(align_data_t *p_align_data) {

  TRACE_1("compute_tile_matches: start\n");

#ifndef SKIPJSON
  // const std::string out_filepath_base(std::string(p_align_data->output_dirpath) +
  //                                     "/matched_sifts/W01_Sec");
  char out_filepath_base[strlen(p_align_data->output_dirpath) +
                         strlen("/matched_sifts/W01_Sec") + 1];
  sprintf(out_filepath_base, "%s/matched_sifts/W01_Sec",
          p_align_data->output_dirpath);
#endif
  // Iterate over all pairs of tiles
  for (int sec_id = 0; sec_id < p_align_data->n_sections; ++sec_id) {
    section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
    char out_filepath_start[MAX_FILEPATH] = "\0";
#ifndef SKIPJSON
    // Get the section id from the first tile.

    // const std::string tmp_filepath((p_sec_data->tiles[0]).filepath);
    // const std::string tmp_imagename(tmp_filepath.substr(tmp_filepath.find_last_of("/") + 1));
    // const std::string section_id(tmp_imagename.substr(0, tmp_imagename.find("_")));

    const char *tmp_filepath = (p_sec_data->tiles[0]).filepath;
    const char *section_id_start = strrchr(tmp_filepath, '/') + 1;
    const char *section_id_end = strchr(section_id_start, '_');
    size_t section_id_size = (section_id_end - section_id_start)/sizeof(char);
    char section_id[section_id_size + 1];
    strncpy(section_id, section_id_start, section_id_size);
    section_id[section_id_size] = '\0';
    // Create the inter-mfov output directory
    // const std::string out_filepath_start(out_filepath_base + section_id + "/");
    // system((std::string("mkdir -p ") + out_filepath_start + "inter/").c_str());
    sprintf(out_filepath_start, "%s%s/", out_filepath_base, section_id);
    {
      char mkdir_cmd[MAX_FILEPATH];
      sprintf(mkdir_cmd, "mkdir -p %sinter/", out_filepath_start);
      system(mkdir_cmd);
    }
    // Create a new vector of output file names.
    // std::vector< std::string > output_files;
    // simple_mutex_t output_files_lock;
    // simple_mutex_init(&output_files_lock);
    cilk::reducer_list_append< const char* > output_files_reducer;
#endif
    // Record the set of mfov ID's encountered.  Right now, this set
    // is used to limit the number of system calls performed.
    std::set<int> mfovs;
    simple_mutex_t mfovs_lock;
    simple_mutex_init(&mfovs_lock);

    cilk_for (int atile_id = 0; atile_id < p_sec_data->n_tiles; ++atile_id) {
      tile_data_t *a_tile = &(p_sec_data->tiles[atile_id]);
      simple_acquire(&mfovs_lock);
      if (mfovs.insert(a_tile->mfov_id).second) {
        // Encountering a brand new mfov.
#ifndef SKIPJSON
        // Create the output directory.
        // system((std::string("mkdir -p ") + out_filepath_start + "intra/" +
        //         std::to_string(a_tile->mfov_id) + "/").c_str());
        char mkdir_cmd[MAX_FILEPATH];
        sprintf(mkdir_cmd, "mkdir -p %sintra/%d/",
                out_filepath_start, a_tile->mfov_id);
        system(mkdir_cmd);
#endif
      }
      simple_release(&mfovs_lock);
#ifndef SKIPJSON
      // Compute starts of output file path and output file name.
      // const std::string a_filepath(a_tile->filepath);
      // const std::string a_timagename(a_filepath.substr(a_filepath.find_last_of("/") + 1));
      // const std::string out_filename_start("W01_Sec" +
      //                                      section_id + "_sift_matches_" +
      //                                      a_timagename.substr(0, a_timagename.find_last_of(".")) + "_");
      const char *a_filepath = a_tile->filepath;
      const char *a_imagename_start = strrchr(a_filepath, '/') + 1;
      const char *a_imagename_end = strrchr(a_imagename_start, '.');
      size_t a_imagename_size = (a_imagename_end - a_imagename_start)/sizeof(char);
      char a_imagename[a_imagename_size + 1];
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
      }

      //for (int btile_id = atile_id + 1; btile_id < p_sec_data->n_tiles; ++btile_id) {
      for (int i = 0; i < indices_to_check_len; i++) {
        int btile_id = indices_to_check[i];
        // if (atile_id == btile_id) continue;
        tile_data_t *b_tile = &(p_sec_data->tiles[btile_id]);

        // Skip tiles that don't overlap
        if (!is_tiles_overlap(a_tile, b_tile)) continue;            

        // Index pair is:
        // a_tile->mfov_id, a_tile->index
        // b_tile->mfov_id, b_tile->index
        TRACE_1("    -- index_pair [%d_%d, %d_%d]\n",
                a_tile->mfov_id, a_tile->index,
                b_tile->mfov_id, b_tile->index);

        TRACE_1("    -- %d_%d features_num: %lu\n",
                a_tile->mfov_id, a_tile->index,
                a_tile->p_kps->size());
        TRACE_1("    -- %d_%d features_num: %lu\n",
                b_tile->mfov_id, b_tile->index,
                b_tile->p_kps->size());
#ifndef SKIPJSON
        // Compute output file path and output file name.

        // const std::string b_filepath(b_tile->filepath);
        // const std::string b_timagename(b_filepath.substr(b_filepath.find_last_of("/")+1));
        // std::string out_filepath(out_filepath_start);
        // if (a_tile->mfov_id == b_tile->mfov_id) {
        //   // Intra-mfov job
        //   out_filepath += "intra/" + std::to_string(a_tile->mfov_id) + "/";
        // } else {
        //   // Inter-mfov job
        //   out_filepath += "inter/";
        // }
        // const std::string out_filename = out_filename_start +
        //   b_timagename.substr(0, b_timagename.find_last_of(".")) +
        //   std::string(".json");
        // out_filepath += out_filename;

        const char *b_filepath = b_tile->filepath;
        const char *b_imagename_start = strrchr(b_filepath, '/') + 1;
        const char *b_imagename_end = strrchr(b_imagename_start, '.');
        size_t b_imagename_size = (b_imagename_end - b_imagename_start)/sizeof(char);
        char b_imagename[b_imagename_size + 1];
        strncpy(b_imagename, b_imagename_start, b_imagename_size);
        b_imagename[b_imagename_size] = '\0';
        char out_filepath[MAX_FILEPATH];
        if (a_tile->mfov_id == b_tile->mfov_id) {
          // Intra-mfov job
          sprintf(out_filepath, "%sintra/%d/%s%s.json",
                  out_filepath_start, a_tile->mfov_id, out_filename_start, b_imagename);
        } else {
          // Inter-mfov job
          sprintf(out_filepath, "%sinter/%s%s.json",
                  out_filepath_start, out_filename_start, b_imagename);
        }

        // Record the output file name.
        // simple_acquire(&output_files_lock);
        // output_files.push_back("file://" + out_filepath);
        // simple_release(&output_files_lock);
        // output_files_reducer->push_back(("file://" + out_filepath).c_str());
        char *output_file_buf = new char[strlen(out_filepath) + 7];
        sprintf(output_file_buf, "file://%s", out_filepath);
        output_files_reducer.push_back(output_file_buf);
#endif

        // Check that both tiles have enough features to match.
        if (a_tile->p_kps->size() < MIN_FEATURES_NUM) {
          TRACE_1("Less than %d features in tile %d_%d, saving empty match file\n",
                  MIN_FEATURES_NUM,
                  a_tile->mfov_id, a_tile->index);
          save_tile_matches(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          continue;
        }
        if (b_tile->p_kps->size() < MIN_FEATURES_NUM) {
          TRACE_1("Less than %d features in tile %d_%d, saving empty match file\n",
                  MIN_FEATURES_NUM,
                  b_tile->mfov_id, b_tile->index);
          save_tile_matches(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          continue;
        }

        // Filter the features, os that only features that are in the
        // overlapping tile will be matches.
        std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;
        atile_kps_in_overlap.reserve(a_tile->p_kps->size());
        btile_kps_in_overlap.reserve(b_tile->p_kps->size());
        // atile_kps_in_overlap.clear(); btile_kps_in_overlap.clear();
        cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
        {
          // Compute bounding box of overlap
          int overlap_x_start = a_tile->x_start > b_tile->x_start ? a_tile->x_start : b_tile->x_start;
          int overlap_x_finish = a_tile->x_finish < b_tile->x_finish ? a_tile->x_finish : b_tile->x_finish;
          int overlap_y_start = a_tile->y_start > b_tile->y_start ? a_tile->y_start : b_tile->y_start;
          int overlap_y_finish = a_tile->y_finish < b_tile->y_finish ? a_tile->y_finish : b_tile->y_finish;
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
          // atile_kps_desc_in_overlap_list.clear(); btile_kps_desc_in_overlap_list.clear();
          
          // Filter the points in a_tile.
          for (size_t pt_idx = 0; pt_idx < a_tile->p_kps->size(); ++pt_idx) {
            // std::vector<cv::Point2f> transformed_pt;
            // cv::transform(std::vector<cv::KeyPoint>({(*a_tile->p_kps)[pt_idx]}),
            //               transformed_pt,
            //               *(p_sec_data->p_transforms)[atile_id]);
            cv::Point2f pt = (*a_tile->p_kps)[pt_idx].pt;
            if (bbox_contains(pt.x+a_tile->x_start, pt.y+a_tile->y_start, // transformed_pt[0],
                              overlap_x_start, overlap_x_finish,
                              overlap_y_start, overlap_y_finish)) {
              atile_kps_in_overlap.push_back((*a_tile->p_kps)[pt_idx]);
              // TB: Not sure if the following works.
              //atile_kps_desc_in_overlap.push_back(a_tile->p_kps_desc->row(pt_idx).clone());
              // NOTE(TFK): Fixed - I don't think it was correct (segfaults in clang & gcc4.8 in parallel).
              atile_kps_desc_in_overlap_list.push_back(a_tile->p_kps_desc->row(pt_idx).clone());
            }
          }
          cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));

          // Filter the points in b_tile.
          for (size_t pt_idx = 0; pt_idx < b_tile->p_kps->size(); ++pt_idx) {
            // std::vector<cv::Point2f> transformed_pt;
            // cv::transform(std::vector<cv::KeyPoint>({(*b_tile->p_kps)[pt_idx]}),
            //               transformed_pt,
            //               *(p_sec_data->p_transforms)[atile_id]);
            cv::Point2f pt = (*b_tile->p_kps)[pt_idx].pt;
            if (bbox_contains(pt.x+b_tile->x_start, pt.y+b_tile->y_start, // transformed_pt[0],
                              overlap_x_start, overlap_x_finish,
                              overlap_y_start, overlap_y_finish)) {
              btile_kps_in_overlap.push_back((*b_tile->p_kps)[pt_idx]);
              // TB: Not sure if the following works.
              //btile_kps_desc_in_overlap.push_back(b_tile->p_kps_desc->row(pt_idx).clone());
              // NOTE(TFK): Fixed - I don't think it was correct (segfaults in clang & gcc4.8 in parallel).
              btile_kps_desc_in_overlap_list.push_back(b_tile->p_kps_desc->row(pt_idx).clone());
            }
          }
          cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));
        }

        TRACE_1("    -- %d_%d overlap_features_num: %lu\n",
                a_tile->mfov_id, a_tile->index,
                atile_kps_in_overlap.size());
        TRACE_1("    -- %d_%d overlap_features_num: %lu\n",
                b_tile->mfov_id, b_tile->index,
                btile_kps_in_overlap.size());

        // TODO: Deal with optionally filtering the maximal number of
        // features from one tile.
        //
        // TB: The corresponding code in the Python pipeline did not
        // appear to run at all, at least on the small data set, so
        // I'm skipping this part for now.

        // Check that both tiles have enough features in the overlap
        // to match.
        if (atile_kps_in_overlap.size() < MIN_FEATURES_NUM) {
          TRACE_1("Less than %d features in the overlap in tile %d_%d, saving empty match file\n",
                  MIN_FEATURES_NUM,
                  a_tile->mfov_id, a_tile->index);
          save_tile_matches(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          continue;
        }
        if (btile_kps_in_overlap.size() < MIN_FEATURES_NUM) {
          TRACE_1("Less than %d features in the overlap in tile %d_%d, saving empty match file\n",
                  MIN_FEATURES_NUM,
                  b_tile->mfov_id, b_tile->index);
          save_tile_matches(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          continue;
        }

        // Match the features
        std::vector< cv::DMatch > matches;
        // matches.clear();
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       ROD);

        TRACE_1("    -- [%d_%d, %d_%d] matches: %lu\n",
                a_tile->mfov_id, a_tile->index,
                b_tile->mfov_id, b_tile->index,
                matches.size());


        // Filter the matches with RANSAC
        // std::vector< cv::DMatch > filtered_matches;
        std::vector< cv::Point2f > filtered_match_points_a, filtered_match_points_b;
        filtered_match_points_a.reserve(matches.size());
        filtered_match_points_b.reserve(matches.size());
        // filtered_match_points_a.clear(); filtered_match_points_b.clear();
        // cv::Mat model;
        // {
          // Extract the match points
          std::vector< cv::Point2f > match_points_a, match_points_b;
          // match_points_a.clear(); match_points_b.clear();
          match_points_a.reserve(matches.size());
          match_points_b.reserve(matches.size());
          for (size_t i = 0; i < matches.size(); ++i) {
            match_points_a.push_back(atile_kps_in_overlap[matches[i].queryIdx].pt);
            match_points_b.push_back(btile_kps_in_overlap[matches[i].trainIdx].pt);
          }

          // // Use cv::findHomography to run RANSAC on the match points.
          // //
          // // TB: Using the maxEpsilon value (10) from
          // // conf_example.json as the ransacReprojThreshold for
          // // findHomography.
          // //
          // // TODO: Read the appropriate RANSAC settings from the
          // // configuration file.
          // cv::Mat mask(matches.size(), 1, CV_8UC1);
          // cv::Mat H = cv::findHomography(match_points_a, match_points_b, cv::RANSAC,
          //                                MAX_EPSILON, mask);

          cv::Mat *mask = new cv::Mat(matches.size(), 1, CV_8UC1);
          cv::Mat model = cv::estimateAffinePartial2D(match_points_a, match_points_b,
                                                      *mask, cv::RANSAC, MAX_EPSILON);

          TRACE_1("    -- [%d_%d, %d_%d] estimated a %d by %d affine transform matrix.\n",
                  a_tile->mfov_id, a_tile->index,
                  b_tile->mfov_id, b_tile->index,
                  model.rows, model.cols);
          if (model.empty()) {
            TRACE_1("Could not estimate affine transform, saving empty match file\n");
            save_tile_matches(0, out_filepath,
                              a_tile, b_tile,
                              nullptr, nullptr, nullptr);
            continue;
          }

          // Use the output mask to filter the matches
          for (size_t i = 0; i < matches.size(); ++i) {
            if (mask->at<bool>(i)) {
              // filtered_matches.push_back(matches[i]);
              filtered_match_points_a.push_back(atile_kps_in_overlap[matches[i].queryIdx].pt);
              filtered_match_points_b.push_back(btile_kps_in_overlap[matches[i].trainIdx].pt);
            }
          }

          delete mask;
          // }

        TRACE_1("    -- [%d_%d, %d_%d] filtered_matches: %lu\n",
                a_tile->mfov_id, a_tile->index,
                b_tile->mfov_id, b_tile->index,
                filtered_match_points_a.size());

        if (filtered_match_points_a.size() < MIN_FEATURES_NUM) {
          TRACE_1("Less than %d matched features, saving empty match file\n",
                  MIN_FEATURES_NUM);
          save_tile_matches(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          continue;
        }

        // Write the JSON output
        {
          // // Extract the match points
          // std::vector<cv::Point2f> filtered_match_points_a, filtered_match_points_b;
          // for (size_t i = 0; i < filtered_matches.size(); ++i) {
          //   filtered_match_points_a.push_back(atile_kps_in_overlap[filtered_matches[i].queryIdx].pt);
          //   filtered_match_points_b.push_back(btile_kps_in_overlap[filtered_matches[i].trainIdx].pt);
          // }

          // // Estimate the rigid transform from the matched points in
          // // tile A to those in tile B.
          // cv::Mat model = cv::estimateRigidTransform(filtered_match_points_a,
          //                                            filtered_match_points_b, false);
          // TRACE_1("    -- [%d_%d, %d_%d] estimated a %d by %d affine transform matrix.\n",
          //         a_tile->mfov_id, a_tile->index,
          //         b_tile->mfov_id, b_tile->index,
          //         model.rows, model.cols);
          // if (0 == model.rows) {
          //   TRACE_1("Could not estimate affine transform, saving empty match file\n");
          //   save_tile_matches(0, out_filepath,
          //                     a_tile, b_tile,
          //                     nullptr, nullptr, nullptr);
          //   continue;
          // }

          // Transform the matched points in tile A.  These
          // transformed points are used to estimate the distances
          // between matched points after alignment.
          std::vector< cv::Point2f > match_points_a_fixed;
          match_points_a_fixed.reserve(filtered_match_points_a.size());
          // match_points_a_fixed.clear();
          // std::vector< cv::Point2f > match_points_b_fixed;
          cv::transform(filtered_match_points_a, match_points_a_fixed, model);
          // cv::transform(match_points_b, match_points_b_fixed, model);

          // Output the tile matches.
          save_tile_matches(filtered_match_points_a.size(), out_filepath,
                            a_tile, b_tile,
                            &filtered_match_points_a, &filtered_match_points_b,
                            &match_points_a_fixed);
        }
      }  // for (btile_id)
    }  // for (atile_id)

#ifndef SKIPJSON
    // Output list of matched sift files.
    {
      const std::string matched_sifts_files =
        std::string(p_align_data->output_dirpath) +
        "/W01_Sec" + section_id + "_matched_sifts_files.txt";
      TRACE_1("Recording matched sifts files in %s\n",
              matched_sifts_files.c_str());

      thread_local FILE *fp = fopen(matched_sifts_files.c_str(), "wb");
      // for (size_t i = 0; i < output_files.size(); ++i)
      //   fprintf(fp, "%s\n", output_files[i].c_str());
      const std::list< const char* > &output_files = output_files_reducer.get_value();
      for (const char *output_file : output_files)
        fprintf(fp, "%s\n", output_file);
      fclose(fp);
    }
#endif
  }  // for (sec_id)

  TRACE_1("compute_tile_matches: finish\n");
}

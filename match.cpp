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

#include <set>
#include <mutex>
#include <cilk/cilk.h>

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
                              const std::string &out_filepath,
                              const tile_data_t *a_data,
                              const tile_data_t *b_data,
                              const std::vector< cv::Point2f > *match_points_a,
                              const std::vector< cv::Point2f > *match_points_b,
                              const std::vector< cv::Point2f > *match_points_a_fixed) {
#ifndef SKIPJSON
  static double totalTime = 0;
  FILE *fp;

  TRACE_1("save_tile_matches: start\n");

  TRACE_1("Writing %s\n", out_filepath.c_str());

  fasttime_t tstart = gettime();

  fp = fopen(out_filepath.c_str(), "wb");
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

void compute_tile_matches(align_data_t *p_align_data) {

  TRACE_1("compute_tile_matches: start\n");

#ifndef SKIPJSON
  const std::string out_filepath_base =
    std::string(p_align_data->output_dirpath) + "/matched_sifts/W01_Sec";
#endif
  // Iterate over all pairs of tiles
  for (int sec_id = 0; sec_id < p_align_data->n_sections; ++sec_id) {
    section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
#ifndef SKIPJSON
    // Get the section id from the first tile.
    const std::string tmp_filepath = std::string((p_sec_data->tiles[0]).filepath);
    const std::string tmp_imagename = tmp_filepath.substr(tmp_filepath.find_last_of("/") + 1);
    const std::string real_section_id = tmp_imagename.substr(0, tmp_imagename.find("_"));
    // Create the inter-mfov output directory
    const std::string out_filepath_start = out_filepath_base + real_section_id + "/";
    system((std::string("mkdir -p ") + out_filepath_start + "inter/").c_str());
    // Create a new vector of output file names.
    std::vector< std::string > output_files;
#endif
    // Record the set of mfov ID's encountered.  Right now, this set
    // is used to limit the number of system calls performed.
    std::set<int> mfovs;
    for (int atile_id = 0; atile_id < p_sec_data->n_tiles; ++atile_id) {
      tile_data_t *a_tile = &(p_sec_data->tiles[atile_id]);
      if (mfovs.insert(a_tile->mfov_id).second) {
        // Encountering a brand new mfov.
#ifndef SKIPJSON
        // Create the output directory.
        system((std::string("mkdir -p ") + out_filepath_start + "intra/" +
                std::to_string(a_tile->mfov_id) + "/").c_str());
#endif
      }
#ifndef SKIPJSON
      // Compute starts of output file path and output file name.
      const std::string a_filepath = std::string(a_tile->filepath);
      const std::string a_timagename = a_filepath.substr(a_filepath.find_last_of("/")+1);
      const std::string out_filename_start = std::string("W01_Sec") +
        real_section_id + std::string("_sift_matches_") +
        a_timagename.substr(0, a_timagename.find_last_of(".")) + "_"; 
#endif
      for (int btile_id = atile_id + 1; btile_id < p_sec_data->n_tiles; ++btile_id) {
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
        const std::string b_filepath = std::string(b_tile->filepath);
        const std::string b_timagename = b_filepath.substr(b_filepath.find_last_of("/")+1);
        std::string out_filepath = out_filepath_start;
        if (a_tile->mfov_id == b_tile->mfov_id) {
          // Intra mfov job
          out_filepath += "intra/" + std::to_string(a_tile->mfov_id) + "/";
        } else {
          // Inter mfov job
          out_filepath += "inter/";
        }
        const std::string out_filename = out_filename_start +
          b_timagename.substr(0, b_timagename.find_last_of(".")) +
          std::string(".json");
        out_filepath += out_filename;
        // Record the output file name.
        output_files.push_back("file://" + out_filepath);
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
              atile_kps_desc_in_overlap.push_back(a_tile->p_kps_desc->row(pt_idx));
            }
          }

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
              btile_kps_desc_in_overlap.push_back(b_tile->p_kps_desc->row(pt_idx));
            }
          }
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
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       ROD);

        TRACE_1("    -- [%d_%d, %d_%d] matches: %lu\n",
                a_tile->mfov_id, a_tile->index,
                b_tile->mfov_id, b_tile->index,
                matches.size());


        // Filter the matches with RANSAC
        std::vector< cv::DMatch > filtered_matches;
        {
          // Extract the match points
          std::vector<cv::Point2f> match_points_a, match_points_b;
          for (size_t i = 0; i < matches.size(); ++i) {
            match_points_a.push_back(atile_kps_in_overlap[matches[i].queryIdx].pt);
            match_points_b.push_back(btile_kps_in_overlap[matches[i].trainIdx].pt);
          }

          // Use cv::findHomography to run RANSAC on the match points.
          //
          // TB: Using the maxEpsilon value (10) from
          // conf_example.json as the ransacReprojThreshold for
          // findHomography.
          //
          // TODO: Read the appropriate RANSAC settings from the
          // configuration file.
          cv::Mat mask(matches.size(), 1, CV_8UC1);
          cv::Mat H = cv::findHomography(match_points_a, match_points_b, cv::RANSAC,
                                         MAX_EPSILON, mask);

          // Use the output mask from findHomography to filter the matches
          for (size_t i = 0; i < matches.size(); ++i) {
            if (mask.at<bool>(0, i))
              filtered_matches.push_back(matches[i]);
          }
        }

        TRACE_1("    -- [%d_%d, %d_%d] filtered_matches: %lu\n",
                a_tile->mfov_id, a_tile->index,
                b_tile->mfov_id, b_tile->index,
                filtered_matches.size());

        if (filtered_matches.size() < MIN_FEATURES_NUM) {
          TRACE_1("Less than %d matched features, saving empty match file\n",
                  MIN_FEATURES_NUM);
          save_tile_matches(0, out_filepath,
                            a_tile, b_tile,
                            nullptr, nullptr, nullptr);
          continue;
        }

        // Write the JSON output
        {
          // Extract the match points
          std::vector<cv::Point2f> match_points_a, match_points_b;
          for (size_t i = 0; i < filtered_matches.size(); ++i) {
            match_points_a.push_back(atile_kps_in_overlap[filtered_matches[i].queryIdx].pt);
            match_points_b.push_back(btile_kps_in_overlap[filtered_matches[i].trainIdx].pt);
          }

          // Estimate the rigid transform from the matched points in
          // tile A to those in tile B.
          cv::Mat model = cv::estimateRigidTransform(match_points_a, match_points_b, false);
          TRACE_1("    -- [%d_%d, %d_%d] estimated a %d by %d affine transform matrix.\n",
                  a_tile->mfov_id, a_tile->index,
                  b_tile->mfov_id, b_tile->index,
                  model.rows, model.cols);
          if (0 == model.rows) {
            TRACE_1("Could not estimate affine transform, saving empty match file\n");
            save_tile_matches(0, out_filepath,
                              a_tile, b_tile,
                              nullptr, nullptr, nullptr);
            continue;
          }

          // Transform the matched points in tile A.  These
          // transformed points are used to estimate the distances
          // between matched points after alignment.
          std::vector< cv::Point2f > match_points_a_fixed;
          // std::vector< cv::Point2f > match_points_b_fixed;
          cv::transform(match_points_a, match_points_a_fixed, model);
          // cv::transform(match_points_b, match_points_b_fixed, model);

          // Output the tile matches.
          save_tile_matches(filtered_matches.size(), out_filepath,
                            a_tile, b_tile,
                            &match_points_a, &match_points_b,
                            &match_points_a_fixed);
        }
      }  // for (btile_id)
    }  // for (atile_id)

#ifndef SKIPJSON
    // Output list of matched sift files.
    {
      const std::string matched_sifts_files =
        std::string(p_align_data->output_dirpath) +
        "/W01_Sec" + real_section_id + "_matched_sifts_files.txt";
      TRACE_1("Recording matched sifts files in %s\n",
              matched_sifts_files.c_str());

      FILE *fp = fopen(matched_sifts_files.c_str(), "wb");
      for (size_t i = 0; i < output_files.size(); ++i)
        fprintf(fp, "%s\n", output_files[i].c_str());
      fclose(fp);
    }
#endif
  }  // for (sec_id)

  TRACE_1("compute_tile_matches: finish\n");
}

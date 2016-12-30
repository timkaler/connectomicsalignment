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

  // TRACE_1("    -- [%d_%d, %d_%d] n_raw_matches: %lu\n",
  //         a_tile->mfov_id, a_tile->index,
  //         b_tile->mfov_id, b_tile->index,
  //         raw_matches.size());

  // Apply ratio test
  for (size_t i = 0; i < raw_matches.size(); i++) {
    if (raw_matches[i][0].distance <
        (ROD * raw_matches[i][1].distance)) {
      matches.push_back(raw_matches[i][0]);
    }
  }
}

void get_range(
               cv::Point2f pt,
               int d1_size,
               int d2_size,
               int patch_leg,
               int *p_d1_start,
               int *p_d1_finish,
               int *p_d2_start,
               int *p_d2_finish) {

  int pt_d2 = pt.x;
  int pt_d1 = pt.y;

  int d1_start = pt_d1 - patch_leg;
  int d1_finish = pt_d1 + patch_leg;

  int d2_start = pt_d2 - patch_leg;
  int d2_finish = pt_d2 + patch_leg;

  if (d1_start < 0) {
    d1_start = 0;
  }

  if (d1_finish > d1_size) {
    d1_finish = d1_size;
  }

  if (d2_start < 0) {
    d2_start = 0;
  }

  if (d2_finish > d2_size) {
    d2_finish = d2_size;
  }

  *p_d1_start = d1_start;
  *p_d1_finish = d1_finish;

  *p_d2_start = d2_start;
  *p_d2_finish = d2_finish;
}

float compare_patches(
                      tile_data_t *p_tile_dst,
                      tile_data_t *p_tile_src,
                      cv::Point2f dst_pt,
                      cv::Point2f src_pt) {

  int rows = p_tile_src->p_image->rows;
  int cols = p_tile_src->p_image->cols;

  int patch_leg = 30;

  ASSERT(rows == p_tile_dst->p_image->rows);
  ASSERT(cols == p_tile_dst->p_image->cols);

  int d1_start;
  int d1_finish;
  int d2_start;
  int d2_finish;

  get_range(
            dst_pt,
            rows,
            cols,
            patch_leg,
            &d1_start,
            &d1_finish,
            &d2_start,
            &d2_finish);

  int dst_d1_len = d1_finish - d1_start;
  int dst_d2_len = d2_finish - d2_start;

  cv::Mat patch_dst = (*p_tile_dst->p_image)(cv::Rect(
                                                      d2_start,
                                                      d1_start,
                                                      dst_d2_len,
                                                      dst_d1_len));

  get_range(
            src_pt,
            rows,
            cols,
            patch_leg,
            &d1_start,
            &d1_finish,
            &d2_start,
            &d2_finish);

  int src_d1_len = d1_finish - d1_start;
  int src_d2_len = d2_finish - d2_start;

  cv::Mat patch_src = (*p_tile_src->p_image)(cv::Rect(
                                                      d2_start,
                                                      d1_start,
                                                      src_d2_len,
                                                      src_d1_len));

  if ((dst_d1_len != src_d1_len) || (dst_d2_len != src_d2_len)) {
    return 0;
  }

  cv::Mat m_res = cv::Mat::zeros(1, 1, CV_32FC1);

  cv::matchTemplate(patch_dst, patch_src, m_res, CV_TM_CCORR_NORMED);

  float f_res = m_res.at<float>(0, 0);

  return f_res;
}

double get_angle(cv::Point2f p1, cv::Point2f p2) {
  double angle = atan2(p2.y - p1.y, p2.x - p1.x) * 180.0 / CV_PI;

  if (angle < 0) {
    angle = angle + 360;
  }

  //printf("get_angle: %f\n", angle);

  return angle;
}

static double dist(cv::Point2f a_pt, cv::Point2f b_pt) {
  double x_delta = a_pt.x - b_pt.x;
  double y_delta = a_pt.y - b_pt.y;
  return std::sqrt((x_delta * x_delta) + (y_delta * y_delta));
}

static void save_tile_matches(tile_data_t *a_data, tile_data_t *b_data,
                              std::vector< cv::Point2f > &match_points_a,
                              std::vector< cv::Point2f > &match_points_b,
                              std::vector< cv::Point2f > &match_points_a_fixed,
                              // std::vector< cv::Point2f > &match_points_b_fixed,
                              size_t num_matches,
                              std::string &out_filepath) {

  FILE *fp;

  TRACE_1("save_tile_matches: start\n");

  TRACE_1("Writing %s\n", out_filepath.c_str());

  const int INDENT_SPACES = 4;
  fp = fopen(out_filepath.c_str(), "wb");
  // Output prologue
  fprintf(fp, "[\n");
  fprintf(fp, "%*s%s\n", INDENT_SPACES, "", "{");

  // Output correspondence point pairs
  TRACE_1("  -- outputting %lu matches\n", num_matches);
  fprintf(fp, "%*s %s\n", 2*INDENT_SPACES, "",
          "\"correspondencePointPairs\": [");
  for (size_t i = 0; i < num_matches; ++i) {
    fprintf(fp, "%*s %s\n", 3*INDENT_SPACES, "", "{");

    // Emit distance
    fprintf(fp, "%*s %s %f,\n", 4*INDENT_SPACES, "",
            "\"dist_after_ransac\":",
            dist(match_points_a_fixed[i], match_points_b[i]));
    // Emit first point
    fprintf(fp, "%*s %s\n", 4*INDENT_SPACES, "",
            "\"p1\": {");

    fprintf(fp, "%*s %s\n", 5*INDENT_SPACES, "",
            "\"l\": [");
    fprintf(fp, "%*s %f,\n", 6*INDENT_SPACES, "",
            match_points_a[i].x);
    fprintf(fp, "%*s %f\n", 6*INDENT_SPACES, "",
            match_points_a[i].y);
    fprintf(fp, "%*s %s\n", 5*INDENT_SPACES, "",
            "],");

    fprintf(fp, "%*s %s\n", 5*INDENT_SPACES, "",
            "\"w\": [");
    fprintf(fp, "%*s %f,\n", 6*INDENT_SPACES, "",
            match_points_a[i].x + a_data->x_start);
    fprintf(fp, "%*s %f\n", 6*INDENT_SPACES, "",
            match_points_a[i].y + a_data->y_start);
    fprintf(fp, "%*s %s\n", 5*INDENT_SPACES, "",
            "],");

    fprintf(fp, "%*s %s\n", 4*INDENT_SPACES, "",
            "},");

    // Emit second point
    fprintf(fp, "%*s %s\n", 4*INDENT_SPACES, "",
            "\"p2\": {");

    fprintf(fp, "%*s %s\n", 5*INDENT_SPACES, "",
            "\"l\": [");
    fprintf(fp, "%*s %f,\n", 6*INDENT_SPACES, "",
            match_points_b[i].x);
    fprintf(fp, "%*s %f\n", 6*INDENT_SPACES, "",
            match_points_b[i].y);
    fprintf(fp, "%*s %s\n", 5*INDENT_SPACES, "",
            "],");

    fprintf(fp, "%*s %s\n", 5*INDENT_SPACES, "",
            "\"w\": [");
    fprintf(fp, "%*s %f,\n", 6*INDENT_SPACES, "",
            match_points_b[i].x + b_data->x_start);
    fprintf(fp, "%*s %f\n", 6*INDENT_SPACES, "",
            match_points_b[i].y + b_data->y_start);
    fprintf(fp, "%*s %s\n", 5*INDENT_SPACES, "",
            "]");

    fprintf(fp, "%*s %s\n", 4*INDENT_SPACES, "",
            "}");

    fprintf(fp, "%*s %s\n", 3*INDENT_SPACES, "", "},");
  }
  fprintf(fp, "%*s %s\n", 2*INDENT_SPACES, "",
          "],");

  // Output mipmapLevel
  fprintf(fp, "%*s %s: %d,\n", 2*INDENT_SPACES, "",
          "\"mipmapLevel\"", 0);

  // TODO: Output model

  // Output input file images
  fprintf(fp, "%*s %s: \"file://%s\",\n", 2*INDENT_SPACES, "",
          "\"url1\"", a_data->filepath);
  fprintf(fp, "%*s %s: \"file://%s\"\n", 2*INDENT_SPACES, "",
          "\"url2\"", b_data->filepath);
  
  // Output epilogue
  fprintf(fp, "%*s %s\n", INDENT_SPACES, "", "}");
  fprintf(fp, "]");

  fclose(fp);
  TRACE_1("save_tile_matches: finish\n");
}

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////

void compute_tile_matches(align_data_t *p_align_data) {

  TRACE_1("compute_tile_matches: start\n");

  // std::vector<cv::Point2f> sec_pts_dst[MAX_SECTIONS][MAX_TILES];
  // std::vector<cv::Point2f> sec_pts_src[MAX_SECTIONS][MAX_TILES];

  // Iterate over all pairs of tiles
  for (int sec_id = 0; sec_id < p_align_data->n_sections; ++sec_id) {
    section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
    for (int atile_id = 0; atile_id < p_sec_data->n_tiles; ++atile_id) {
      for (int btile_id = atile_id + 1; btile_id < p_sec_data->n_tiles; ++btile_id) {
        if (atile_id == btile_id) continue;

        tile_data_t *a_tile = &(p_sec_data->tiles[atile_id]);
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

        std::string a_filepath = std::string(a_tile->filepath);
        std::string a_timagename = a_filepath.substr(a_filepath.find_last_of("/")+1);
        std::string a_real_section_id = a_timagename.substr(0, a_timagename.find("_"));
        std::string b_filepath = std::string(b_tile->filepath);
        std::string b_timagename = b_filepath.substr(b_filepath.find_last_of("/")+1);
        // std::string b_real_section_id = b_timagename.substr(0, b_timagename.find("_"));
        std::string out_filepath = std::string(p_align_data->output_dirpath) +
          "/matched_sifts/W01_Sec" + a_real_section_id + "/";
        if (a_tile->mfov_id == b_tile->mfov_id) {
          // Intra mfov job
          out_filepath = out_filepath + "intra/" + std::to_string(a_tile->mfov_id);
        } else {
          // Inter mfov job
          out_filepath = out_filepath + "inter";
        }
        system((std::string("mkdir -p ") + out_filepath).c_str());
        out_filepath = out_filepath + std::string("/W01_Sec") +
          a_real_section_id + std::string("_sift_matches_") +
          a_timagename.substr(0, a_timagename.find_last_of(".")) +
          b_timagename.substr(0, b_timagename.find_last_of(".")) +
          std::string(".json");

        // Check that both tiles have enough features to match.
        if (a_tile->p_kps->size() < MIN_FEATURES_NUM) {
          TRACE_1("Less than %d features in tile %d_%d, saving empty match file\n",
                  MIN_FEATURES_NUM,
                  a_tile->mfov_id, a_tile->index);
          continue;
        }
        if (b_tile->p_kps->size() < MIN_FEATURES_NUM) {
          TRACE_1("Less than %d features in tile %d_%d, saving empty match file\n",
                  MIN_FEATURES_NUM,
                  b_tile->mfov_id, b_tile->index);
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
          continue;
        }
        if (btile_kps_in_overlap.size() < MIN_FEATURES_NUM) {
          TRACE_1("Less than %d features in the overlap in tile %d_%d, saving empty match file\n",
                  MIN_FEATURES_NUM,
                  b_tile->mfov_id, b_tile->index);
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
          continue;
        }

        // Write the json output
        {
          // Extract the match points
          std::vector<cv::Point2f> match_points_a, match_points_b;
          for (size_t i = 0; i < filtered_matches.size(); ++i) {
            match_points_a.push_back(atile_kps_in_overlap[filtered_matches[i].queryIdx].pt);
            match_points_b.push_back(btile_kps_in_overlap[filtered_matches[i].trainIdx].pt);
          }

          cv::Mat model = cv::estimateRigidTransform(match_points_a, match_points_b, false);
          TRACE_1("    -- [%d_%d, %d_%d] rigid transform matrix: %d by %d\n",
                  a_tile->mfov_id, a_tile->index,
                  b_tile->mfov_id, b_tile->index,
                  model.rows, model.cols);
          if (0 == model.rows) {
            TRACE_1("Could not estimate rigid transform, saving empty match file\n");
            continue;
          }

          // Extract the match points
          // std::vector< cv::Point2f > match_points_a, match_points_b;
          // for (size_t i = 0; i < filtered_matches.size(); ++i) {
          //   match_points_a.push_back(atile_kps_in_overlap[filtered_matches[i].queryIdx].pt);
          //   match_points_b.push_back(btile_kps_in_overlap[filtered_matches[i].trainIdx].pt);
          // }
          std::vector< cv::Point2f > match_points_a_fixed;
          // std::vector< cv::Point2f > match_points_b_fixed;
          cv::transform(match_points_a, match_points_a_fixed, model);
          // cv::transform(match_points_b, match_points_b_fixed, model);

          save_tile_matches(a_tile, b_tile,
                            match_points_a, match_points_b,
                            match_points_a_fixed,
                            filtered_matches.size(), out_filepath);
        }
      }  // for (btile_id)
    }  // for (atile_id)
  }  // for (sec_id)

  TRACE_1("compute_tile_matches: finish\n");

}

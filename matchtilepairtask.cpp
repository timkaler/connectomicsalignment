#include "matchtilepairtask.hpp"
namespace tfk {
    bool MatchTilePairTask::bbox_contains(float pt_x, float pt_y,
                              int x_start, int x_finish,
                              int y_start, int y_finish) {
      // TRACE_1("  -- pt: (%f, %f)\n", pt.x, pt.y);
      // TRACE_1("  -- bbox: [(%d, %d), (%d, %d)]\n",
      //         x_start, y_start,
      //         x_finish, y_finish);
      return (pt_x >= x_start && pt_x <= x_finish) &&
        (pt_y >= y_start && pt_y <= y_finish);
    }

    MatchTilePairTask::~MatchTilePairTask() {
    }

    std::vector<float> MatchTilePairTask::get_feature_vector () {
      return feature_vector;
    }

    MatchTilePairTask::MatchTilePairTask (Tile* a_tile, Tile* b_tile, bool train) {
      this->a_tile = a_tile;
      this->b_tile = b_tile;
      this->task_type_id = MATCH_TILE_PAIR_TASK_ID;
      this->min_features_num = MIN_FEATURES_NUM;
      if (!train) {
        this->paramDB = a_tile->paramdbs[this->task_type_id];
        this->model = a_tile->ml_models[this->task_type_id];
      }
      FORCE_FAST_PASS_SUCCEED = false;
    }

    MatchTilePairTask::MatchTilePairTask (Tile* a_tile, Tile* b_tile) {
      this->a_tile = a_tile;
      this->b_tile = b_tile;
      this->task_type_id = MATCH_TILE_PAIR_TASK_ID;
      this->paramDB = a_tile->paramdbs[this->task_type_id];
      this->model = a_tile->ml_models[this->task_type_id];
      this->min_features_num = MIN_FEATURES_NUM;
      FORCE_FAST_PASS_SUCCEED = false;
      align_data = a_tile->paramdbs[this->task_type_id]->align_data;
    }

    //MatchTilePairTask::~MatchTilePairTask () {}



    cv::Point2f MatchTilePairTask::compute_quick(Tile* a_tile, Tile* b_tile) {
      std::pair<cv::Point2f, cv::Point2f> overlap_offsets;
      std::pair<cv::Mat, cv::Mat> images = a_tile->get_overlap_matrix(b_tile, 0.5, overlap_offsets);
      printf("overlap offsets %f %f %f %f\n", overlap_offsets.first.x, overlap_offsets.first.y, overlap_offsets.second.x, overlap_offsets.second.y);
      int rows = images.first.rows;
      int cols = images.first.cols;
      if (rows < 20 || cols < 20) return cv::Point2f(0,0);

      int start_r = 0;
      int start_c = 0;

      int start_r2 = rows - 10;
      int start_c2 = cols - 10;

      int start_r3 = rows - 5;
      int start_c3 = cols - 5;

      cv::Mat tmplate(10,10,CV_8UC1);
      cv::Mat tmplate2(10,10,CV_8UC1);
      cv::Mat tmplate3(10,10,CV_8UC1);
      for (int r = 0; r < 10; r++) {
        for (int c = 0; c < 10; c++) {
          tmplate.at<unsigned char>(r,c) = images.first.at<unsigned char>(r+start_r,c+start_c);
          tmplate2.at<unsigned char>(r,c) = images.first.at<unsigned char>(r+start_r2,c+start_c2);
          tmplate3.at<unsigned char>(r,c) = images.first.at<unsigned char>(r+start_r3,c+start_c3);
        }
      }

      cv::Mat result,result2, result3;
      cv::matchTemplate(images.second, tmplate, result, CV_TM_CCOEFF);
      cv::matchTemplate(images.second, tmplate2, result2, CV_TM_CCOEFF);
      cv::matchTemplate(images.second, tmplate2, result3, CV_TM_CCOEFF);


      cv::Point2f best = cv::Point2f(0.0,0.0);
      cv::Point2f best_offset = cv::Point2f(0.0,0.0);
      float best_val = -1000.0;
      for (int r = 0 ; r < result.rows; r++) {
        for (int c = 0; c < result.cols; c++) {
          float res1 = result.at<float>(r,c);
          float res2 = result2.at<float>(r,c);
          float res3 = result3.at<float>(r,c);
          float val = std::min(res1,res2);
          val = std::min(val,res3);
          if (val > best_val) {
            best_val = val;
            best_offset = cv::Point2f(1.0*c,1.0*r);

            best = cv::Point2f(1.0*overlap_offsets.second.x-1.0*overlap_offsets.first.x, 1.0*overlap_offsets.second.y - 1.0*overlap_offsets.first.y);
            printf("best offset is %f %f val %f\n", best_offset.x-start_c, best_offset.y-start_r, val);
          }
        }
      }
      printf("best offset is %f %f\n", best_offset.x-start_c, best_offset.y-start_r);
      return best;
    }

    void MatchTilePairTask::compute_tile_matches_pair(Tile* a_tile, Tile* b_tile,
      std::vector< cv::KeyPoint >& a_tile_keypoints, std::vector< cv::KeyPoint >& b_tile_keypoints,
      cv::Mat& a_tile_desc, cv::Mat& b_tile_desc,
      std::vector< cv::Point2f > &filtered_match_points_a,
      std::vector< cv::Point2f > &filtered_match_points_b, float ransac_thresh) {

      this->avg_response_a = 0;
      this->avg_response_b = 0;
      this->avg_size_a = 0;
      this->avg_size_b = 0;
      this->avg_octave_a = 0;
      this->avg_octave_b = 0;
      this->successful_rod = 0;

      //std::vector<int> neighbors = get_all_close_tiles(a_tile->tile_id);

      if (a_tile_keypoints.size() < min_features_num) return;
      if (b_tile_keypoints.size() < min_features_num) return;

      // Filter the features, so that only features that are in the
      //   overlapping tile will be matches.
      std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;

      //printf("preatile kps %d, prebtile kps %d, secondpass:%d\n", a_tile_keypoints.size(), b_tile_keypoints.size(), second_pass); 
      atile_kps_in_overlap.reserve(a_tile_keypoints.size());
      btile_kps_in_overlap.reserve(b_tile_keypoints.size());

      // atile_kps_in_overlap.clear(); btile_kps_in_overlap.clear();
      cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;

      { // Begin scoped block A.
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
        atile_kps_desc_in_overlap_list.reserve(a_tile_keypoints.size());
        std::vector< cv::Mat > btile_kps_desc_in_overlap_list;
        btile_kps_desc_in_overlap_list.reserve(b_tile_keypoints.size());

        // Filter the points in a_tile.
        for (size_t pt_idx = 0; pt_idx < a_tile_keypoints.size(); ++pt_idx) {
          cv::Point2f pt = a_tile_keypoints[pt_idx].pt;
          if (bbox_contains(pt.x + a_tile->x_start,
                            pt.y + a_tile->y_start,  // transformed_pt[0],
                            overlap_x_start, overlap_x_finish,
                            overlap_y_start, overlap_y_finish)) {
            atile_kps_in_overlap.push_back(a_tile_keypoints[pt_idx]);
            atile_kps_desc_in_overlap_list.push_back(
                a_tile_desc.row(pt_idx).clone());
          }
        }
        cv::vconcat(atile_kps_desc_in_overlap_list,
            (atile_kps_desc_in_overlap));

        // Filter the points in b_tile.
        for (size_t pt_idx = 0; pt_idx < b_tile_keypoints.size(); ++pt_idx) {
          cv::Point2f pt = b_tile_keypoints[pt_idx].pt;
          if (bbox_contains(pt.x + b_tile->x_start,
                            pt.y + b_tile->y_start,  // transformed_pt[0],
                            overlap_x_start, overlap_x_finish,
                            overlap_y_start, overlap_y_finish)) {
            btile_kps_in_overlap.push_back(b_tile_keypoints[pt_idx]);
            btile_kps_desc_in_overlap_list.push_back(b_tile_desc.row(pt_idx).clone());
          }
        }
        cv::vconcat(btile_kps_desc_in_overlap_list,
            (btile_kps_desc_in_overlap));
      } // End scoped block A


      //printf("atile kps %d, btilekps %d, secondpass:%d\n", atile_kps_in_overlap.size(), btile_kps_in_overlap.size(), second_pass); 
      if (atile_kps_in_overlap.size() < min_features_num) return;
      if (btile_kps_in_overlap.size() < min_features_num) return;

      a_tile->keypoints_in_overlap[b_tile] = atile_kps_in_overlap.size();
      b_tile->keypoints_in_overlap[a_tile] = btile_kps_in_overlap.size();

      float trial_rod;
      for (int trial = 0; trial < 1; trial++) {
        if (trial == 0) trial_rod = 0.7;
        if (trial == 1) trial_rod = 0.8;
        if (trial == 2) trial_rod = 0.92;
        if (trial == 3) trial_rod = 0.96;
        trial_rod = 0.92;
        // Match the features
        std::vector< cv::DMatch > matches;
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       trial_rod, /*second_pass*/ true); // always do deterministic brute.

        // Filter the matches with RANSAC
        std::vector<cv::Point2f> match_points_a, match_points_b;
        for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
          match_points_a.push_back(
              a_tile->rigid_transform(atile_kps_in_overlap[matches[tmpi].queryIdx].pt));
          match_points_b.push_back(
              b_tile->rigid_transform(btile_kps_in_overlap[matches[tmpi].trainIdx].pt));
        }

        if (matches.size() < min_features_num) {
          continue;
        }
        a_tile->matched_keypoints_in_overlap[b_tile] = matches.size();
        b_tile->matched_keypoints_in_overlap[a_tile] = matches.size();

        bool* mask = (bool*) calloc(match_points_a.size(), 1);
        double thresh = ransac_thresh;//5.0;
        cv::Point2f best_offset = tfk_simple_ransac(match_points_a, match_points_b, thresh, mask);


        filtered_match_points_a.clear();
        filtered_match_points_b.clear();

        int num_matches_filtered = 0;
        // Use the output mask to filter the matches
        std::vector<float> responses;
        std::vector<float> sizes;
        std::vector<float> octaves;
        float response_sum_a = 0.0;
        float size_sum_a = 0.0;
        float octave_sum_a = 0.0;
        float response_sum_b = 0.0;
        float size_sum_b = 0.0;
        float octave_sum_b = 0.0;
        for (size_t i = 0; i < matches.size(); ++i) {
          if (mask[i]) {
            response_sum_a += atile_kps_in_overlap[matches[i].queryIdx].response;
            response_sum_b += btile_kps_in_overlap[matches[i].trainIdx].response;

            size_sum_a += atile_kps_in_overlap[matches[i].queryIdx].size;
            size_sum_b += btile_kps_in_overlap[matches[i].trainIdx].size;

            octave_sum_a += atile_kps_in_overlap[matches[i].queryIdx].octave;
            octave_sum_b += btile_kps_in_overlap[matches[i].trainIdx].octave;

            num_matches_filtered++;
            filtered_match_points_a.push_back(
                atile_kps_in_overlap[matches[i].queryIdx].pt);
            filtered_match_points_b.push_back(
                btile_kps_in_overlap[matches[i].trainIdx].pt);
          }
        }
        free(mask);
        if (num_matches_filtered >= min_features_num && filtered_match_points_a.size() >= 0.05*matches.size()) {
          this->avg_response_a = response_sum_a/num_matches_filtered;
          this->avg_response_b = response_sum_b/num_matches_filtered;
          this->avg_size_a = size_sum_a/num_matches_filtered;
          this->avg_size_b = size_sum_b/num_matches_filtered;
          this->avg_octave_a = octave_sum_a/num_matches_filtered;
          this->avg_octave_b = octave_sum_b/num_matches_filtered;

          //a_tile->insert_matches(b_tile, filtered_match_points_a, filtered_match_points_b);
          this->best_offset = best_offset;
          this->successful_rod = trial_rod;
          break;
        } else {
          filtered_match_points_a.clear();
          filtered_match_points_b.clear();
        }
      }
    }

    void MatchTilePairTask::compute_with_params(MRParams* mr_params_local) {
      std::vector<cv::KeyPoint> a_tile_keypoints;
      cv::Mat a_tile_desc;
      std::vector<cv::KeyPoint> a_tile_alt_keypoints;
      cv::Mat a_tile_alt_desc;

      this->mr_params = mr_params_local;



  params best_params;
  best_params.num_features = 1000;
  best_params.num_octaves = 6;
  best_params.contrast_threshold = CONTRAST_THRESH;//.015;//CONTRAST_THRESH;
  best_params.edge_threshold = EDGE_THRESH_2D;//10;//EDGE_THRESH_2D;
  best_params.sigma = 1.6;//1.2;//1.6;
  best_params.scale_x = 1.0;
  best_params.scale_y = 1.0;
  best_params.res = FULL;

  if (align_data->use_params) {
    if (align_data->skip_octave_slow) {
      best_params.num_features = 1001;
    } else {
      best_params.num_features = 1000;
    }
  }

  //params trial_params;
  //trial_params.num_features = 2;
  //trial_params.num_octaves = 12;
  //trial_params.contrast_threshold = .015;
  //trial_params.edge_threshold = 10;
  //trial_params.sigma = 1.2;
  //trial_params.scale_x = 0.15;
  //trial_params.scale_y = 0.15;
  //trial_params.res = FULL;

  params trial_params;
  trial_params.num_features = 1001;
  trial_params.num_octaves = 6;
  trial_params.contrast_threshold = CONTRAST_THRESH;////0.015;//.015;
  trial_params.edge_threshold = EDGE_THRESH_2D;//10;//10;
  trial_params.sigma = 1.6; //+ 0.2*0.2;//1.05;//1.05;//1.05;
  trial_params.scale_x = 0.3;
  trial_params.scale_y = 0.3;
  trial_params.res = FULL;

  if (align_data->use_params) {
    if (align_data->skip_octave_fast) {
      trial_params.num_features = 1001;
    } else {
      trial_params.num_features = 1000;
    }
    trial_params.scale_x = align_data->scale_fast;
    trial_params.scale_y = align_data->scale_fast;
    if (!align_data->use_fsj) {
      FORCE_FAST_PASS_SUCCEED = true;
    } else {
      FORCE_FAST_PASS_SUCCEED = false;
    }
  }

      second_pass = false;


      //TODO(wheatman) doing extra work here, but makes it the same as the cached version
      if (dependencies.find(a_tile->tile_id) == dependencies.end()) {
        min_features_num = 5;
        a_tile->compute_sift_keypoints2d_params(best_params, a_tile_keypoints,
                                                a_tile_desc, b_tile);
        second_pass = true;
        //printf("computing A keypoints\n");
      } else {
        min_features_num = 5;
        if (dependencies[a_tile->tile_id]->computed) {
          a_tile_desc = dependencies[a_tile->tile_id]->tile_desc;
          a_tile_keypoints = dependencies[a_tile->tile_id]->tile_keypoints;
        } else {
          a_tile->compute_sift_keypoints2d_params(trial_params, a_tile_keypoints,
                                                  a_tile_desc, b_tile);
        }
      }


      //int neighbor_success_count = 0;

      std::vector<cv::KeyPoint> b_tile_keypoints;
      cv::Mat b_tile_desc;
      std::vector<cv::KeyPoint> b_tile_alt_keypoints;
      cv::Mat b_tile_alt_desc;


      if (dependencies.find(b_tile->tile_id) == dependencies.end()) {
        min_features_num = 5;
        b_tile->compute_sift_keypoints2d_params(best_params, b_tile_keypoints,
                                                b_tile_desc, a_tile);
        second_pass = true;
        //printf("computing B keypoints\n");
      } else {
        min_features_num = 5;
        if (dependencies[b_tile->tile_id]->computed) {
          b_tile_desc = dependencies[b_tile->tile_id]->tile_desc;
          b_tile_keypoints = dependencies[b_tile->tile_id]->tile_keypoints;
        } else {
          b_tile->compute_sift_keypoints2d_params(trial_params, b_tile_keypoints,
                                                  b_tile_desc, a_tile);
        }
      }
      //if (a_tile_keypoints.size() < min_features_num) return; // failure.
      //if (b_tile_keypoints.size() < min_features_num) return;
      std::vector< cv::Point2f > filtered_match_points_a(0);
      std::vector< cv::Point2f > filtered_match_points_b(0);
      std::vector< cv::Point2f > alt_filtered_match_points_a(0);
      std::vector< cv::Point2f > alt_filtered_match_points_b(0);


      this->compute_tile_matches_pair(a_tile, b_tile,
        a_tile_keypoints, b_tile_keypoints,
        a_tile_desc, b_tile_desc,
        filtered_match_points_a,
        filtered_match_points_b, 5.0); // TFKNOTE JUST CHANGED FROM 5.0

      // store the matched points.
      matched_points = std::make_pair(filtered_match_points_a, filtered_match_points_b);
    }
//TODO(wheatman) mark to neighbors as bad
    bool MatchTilePairTask::error_check(float false_negative_rate) {
      std::vector<cv::Point2f> filtered_match_points_a = matched_points.first;
      std::vector<cv::Point2f> filtered_match_points_b = matched_points.second;

      std::vector<float> tmp_vector;
      tmp_vector.push_back(filtered_match_points_a.size()*1.0);

      cv::Point2f average = cv::Point2f(0.0,0.0);
      for (int i = 0; i < filtered_match_points_a.size(); i++) {
        average += filtered_match_points_a[i];
      }
      average /= 1.0*(filtered_match_points_a.size()+1);

      float var_x = 0.0;
      float var_y = 0.0;
      for (int i = 0; i < filtered_match_points_a.size(); i++) {
        float dx = filtered_match_points_a[i].x - average.x;
        float dy = filtered_match_points_a[i].y - average.y;
        var_x += dx*dx;
        var_y += dy*dy;
      }
      var_x /= filtered_match_points_a.size()+1;
      var_y /= filtered_match_points_a.size()+1;

      tmp_vector.push_back(var_x);
      tmp_vector.push_back(var_y);
      tmp_vector.push_back(a_tile->index);
      tmp_vector.push_back(b_tile->index);

      Tile tmp_a_tile = *a_tile;



      // put b at 0,0
      if (filtered_match_points_a.size() >= min_features_num) {
        double energy_before = 0.0;
        for (int _i = 0; _i < 5001; _i++) {
          double dx = 0.0;
          double dy = 0.0;
          energy_before = 0.0;
          for (int j = 0; j < filtered_match_points_a.size(); j++) {
            cv::Point2f p1 = cv::Point2f(1.0l*filtered_match_points_b[j].x,
                                         1.0l*filtered_match_points_b[j].y);
            cv::Point2f p2 = cv::Point2f(1.0l*filtered_match_points_a[j].x,
                                         1.0l*filtered_match_points_a[j].y);

            //cv::Point2f dp = b_tile->rigid_transform_d(filtered_match_points_b[j]) - tmp_a_tile.rigid_transform(filtered_match_points_a[j]);
            cv::Point2f dp = b_tile->rigid_transform(p1) - tmp_a_tile.rigid_transform(p2);
            dx += 2*dp.x * 1.0 / (filtered_match_points_a.size());
            dy += 2*dp.y * 1.0 / (filtered_match_points_a.size());
            energy_before += dp.x*dp.x + dp.y*dp.y;
          }
          if (_i != 5000) {
            tmp_a_tile.offset_x += 0.4*dx;
            tmp_a_tile.offset_y += 0.4*dy;
          }
        }

        // find angle.
       //double ANGLE_STEP = 1e-4;
       //double energy_after = energy_before;
       //double energy_after_best = energy_before;
       // for (int _i = 0; _i < 10000; _i++) {
       //   //double dx = 0.0;
       //   //double dy = 0.0;
       //   tmp_a_tile.angle += ANGLE_STEP;
       //   energy_after = 0.0;

       //   for (int j = 0; j < filtered_match_points_a.size(); j++) {
       //     cv::Point2f p1 = cv::Point2f(1.0l*filtered_match_points_b[j].x,
       //                                  1.0l*filtered_match_points_b[j].y);
       //     cv::Point2f p2 = cv::Point2f(1.0l*filtered_match_points_a[j].x,
       //                                  1.0l*filtered_match_points_a[j].y);

       //     cv::Point2f dp = b_tile->rigid_transform(p1) - tmp_a_tile.rigid_transform(p2);
       //     energy_after += dp.x*dp.x + dp.y*dp.y;
       //   }

       //   if (energy_after < energy_after_best) {
       //     energy_after_best = energy_after;
       //     continue;
       //   }

       //   tmp_a_tile.angle -= 2*ANGLE_STEP;
       //   energy_after = 0.0;
       //   for (int j = 0; j < filtered_match_points_a.size(); j++) {
       //     cv::Point2f p1 = cv::Point2f(1.0l*filtered_match_points_b[j].x,
       //                                  1.0l*filtered_match_points_b[j].y);
       //     cv::Point2f p2 = cv::Point2f(1.0l*filtered_match_points_a[j].x,
       //                                  1.0l*filtered_match_points_a[j].y);

       //     cv::Point2f dp = b_tile->rigid_transform(p1) - tmp_a_tile.rigid_transform(p2);
       //     energy_after += dp.x*dp.x + dp.y*dp.y;
       //  }
       //  if (energy_after < energy_after_best) {
       //     energy_after_best = energy_after;
       //     continue;
       //  }
       //  tmp_a_tile.angle += ANGLE_STEP;
       //  ANGLE_STEP = ANGLE_STEP/10.0;
       //  if (ANGLE_STEP < 1e-12) break;
       //}
       //double energy_after2 = 0.0;
       // for (int _i = 0; _i < 5001; _i++) {
       //   double dx = 0.0;
       //   double dy = 0.0;
       //   energy_after2 = 0.0;
       //   for (int j = 0; j < filtered_match_points_a.size(); j++) {
       //     cv::Point2f p1 = cv::Point2f(1.0l*filtered_match_points_b[j].x,
       //                                  1.0l*filtered_match_points_b[j].y);
       //     cv::Point2f p2 = cv::Point2f(1.0l*filtered_match_points_a[j].x,
       //                                  1.0l*filtered_match_points_a[j].y);

       //     //cv::Point2f dp = b_tile->rigid_transform_d(filtered_match_points_b[j]) - tmp_a_tile.rigid_transform(filtered_match_points_a[j]);
       //     cv::Point2f dp = b_tile->rigid_transform(p1) - tmp_a_tile.rigid_transform(p2);
       //     dx += 2*dp.x * 1.0 / (filtered_match_points_a.size());
       //     dy += 2*dp.y * 1.0 / (filtered_match_points_a.size());
       //     energy_after2 += dp.x*dp.x + dp.y*dp.y;
       //   }
       //   if (_i != 5000) {
       //     tmp_a_tile.offset_x += 0.4*dx;
       //     tmp_a_tile.offset_y += 0.4*dy;
       //   }
       // }

       // if (energy_after_best < energy_before) {
       //   printf("energy before %f energy after %f energy after 2 %f angle %.10e\n", energy_before, energy_after, energy_after2, tmp_a_tile.angle);
       // }

      }

        int overlap_x_start = a_tile->x_start > b_tile->x_start ?
                                  a_tile->x_start : b_tile->x_start;
        int overlap_x_finish = a_tile->x_finish < b_tile->x_finish ?
                                  a_tile->x_finish : b_tile->x_finish;
        int overlap_y_start = a_tile->y_start > b_tile->y_start ?
                                  a_tile->y_start : b_tile->y_start;
        int overlap_y_finish = a_tile->y_finish < b_tile->y_finish ?
                                  a_tile->y_finish : b_tile->y_finish;


      // compute the overlap region


      cv::Point2f overlap_midpoint = 0.5*cv::Point2f(overlap_x_start + overlap_x_finish,
                                                     overlap_y_start + overlap_y_finish);

      cv::Point2f overlap_point_a = overlap_midpoint - cv::Point2f(a_tile->x_start, a_tile->y_start);
      cv::Point2f overlap_point_b = overlap_midpoint - cv::Point2f(b_tile->x_start, b_tile->y_start);


      cv::Point2f a_point = cv::Point2f(tmp_a_tile.x_start+tmp_a_tile.offset_x,
                                          tmp_a_tile.y_start+tmp_a_tile.offset_y);
      cv::Point2f b_point = cv::Point2f(b_tile->x_start+b_tile->offset_x,
                                        b_tile->y_start+b_tile->offset_y);

      current_offset = cv::Point2f(tmp_a_tile.offset_x, tmp_a_tile.offset_y);//a_point - b_point;

      this->predicted_offset = current_offset;



      float val = 2.0;//tmp_a_tile.error_tile_pair(b_tile);
      //tmp_vector.push_back(val);
      this->feature_vector = tmp_a_tile.tile_pair_feature(b_tile);
      this->feature_vector.push_back(filtered_match_points_a.size());

      this->feature_vector.push_back(avg_response_a);
      this->feature_vector.push_back(avg_response_b);
      this->feature_vector.push_back(avg_size_a);
      this->feature_vector.push_back(avg_size_b);
      this->feature_vector.push_back(avg_octave_a);
      this->feature_vector.push_back(avg_octave_b);


      std::vector<float> tmp_feature_vector = a_tile->tile_pair_feature(b_tile);
      if (tmp_feature_vector.size() == 0) {
        printf("major error abort!\n");
        exit(0);
      }
      for (int i = 0; i < tmp_feature_vector.size(); i++) {
        this->feature_vector.push_back(tmp_feature_vector[i]);
      }

      if (this->feature_vector.size() == 0) {
        printf("feature vector size is 0\n");
        //exit(0);
        for (int i = 0; i < 10*10; i++) {
          this->feature_vector.push_back(rand()%256);
          this->feature_vector.push_back(rand()%256);
        }
      }
      this->feature_vector.push_back(this->successful_rod);
      if (false_negative_rate > 0) {
        //bool guess_ml = this->model->predict(a_tile->feature_vectors[b_tile]);
        //bool guess_ml = false;//this->model->predict(a_tile->feature_vectors[b_tile]);

        bool guess_ml;
        if (false_negative_rate <= 1.0) {
      //    printf("called predict\n");

      //for (int i = 0; i < feature_vector.size(); i++) {
      //  printf("%f\n", feature_vector[i]);
      //}
        if (FORCE_FAST_PASS_SUCCEED) {
          guess_ml = true;
        } else {
          guess_ml = this->model->predict(this->feature_vector);//this->model->predict(a_tile->feature_vectors[b_tile]);
        }

        //if (filtered_match_points_a.size() < min_features_num) {
        //  guess_ml = false;
        //}
        if (guess_ml == false) {
          printf("guess ml was false\n");
        }
        } else {
          guess_ml = true;
        }
        //guess_ml = true;
        //if (!second_pass) guess_ml = false;

        if (second_pass) guess_ml = true;
        //guess_ml = true;
        //guess_ml = true;
        //guess_ml = true;


        if (FORCE_FAST_PASS_SUCCEED) {
          guess_ml = true;
        }

        a_tile->ml_preds[b_tile] = guess_ml;


        // indicates training mode
        if (false_negative_rate > 100.0) {
          if (filtered_match_points_a.size() >= min_features_num) {
            return true;
          } else {
            return false;
          }
        }

        if (!(filtered_match_points_a.size() >= min_features_num)) {
          guess_ml = false;
        }

        //guess_ml = filtered_match_points_a.size() >= min_features_num;
        //if (false && /*guess_ml*/ val >= 0.75) {
        if ((guess_ml || false_negative_rate > 1.0) && filtered_match_points_a.size() >= min_features_num) {
          a_tile->ideal_offsets[b_tile->tile_id] = current_offset;
          a_tile->ideal_points[b_tile->tile_id] = std::make_pair(overlap_point_a, overlap_point_b);
          //a_tile->ideal_angles[b_tile->tile_id] = tmp_a_tile.angle;
          a_tile->neighbor_correlations[b_tile->tile_id] = val;
          a_tile->ml_preds[b_tile] = guess_ml;
          success = true;
          return true;
        } else {
          a_tile->ideal_offsets.erase(b_tile->tile_id);
          a_tile->ideal_points.erase(b_tile->tile_id);
          //a_tile->ideal_angles.erase(b_tile->tile_id);
          a_tile->ml_preds[b_tile] = guess_ml;
          success = false;
          //if (!second_pass) {
          //  return guess_ml;
          //}
          return false;
        }
      } else {
        return false;
      }
    }

    void MatchTilePairTask::commit() {
      std::vector<cv::Point2f> filtered_match_points_a = matched_points.first;
      std::vector<cv::Point2f> filtered_match_points_b = matched_points.second;
      if (success) {
        a_tile->insert_matches(b_tile, filtered_match_points_a, filtered_match_points_b);
      }
    }

    void MatchTilePairTask::get_parameter_options(std::vector<tfk::MRParams*>* vec) {
      for (float scale = .2; scale < 1.05; scale +=.1) {
        for (int num_features = 1; num_features <=4; num_features *=2) {
          for (int num_octaves = 5; num_octaves <= 15; num_octaves+=5) {
            //for (float contrast_threshold = .01; contrast_threshold <.03; contrast_threshold +=.05) {
              //for (float edge_threshold = 3; edge_threshold < 10; edge_threshold +=2) {
                //for (float sigma = 1.2; sigma < 2; sigma += .2) {
                  MRParams *new_param = new MRParams();
                  new_param->put_int_param("num_features", num_features);
                  new_param->put_int_param("num_octaves", num_octaves);
                  new_param->put_float_param("scale", scale);
                  //new_param.put_float_param("contrast_threshold", contrast_threshold);
                  //new_param.put_float_param("edge_threshold", edge_threshold);
                  //new_param.put_float_param("sigma", sigma);
                  vec->push_back(new_param);
                //}
              //}
            //}
          }
        }
      }
    }

    bool MatchTilePairTask::compare_results_and_update_model(MRTask * known_good, float accuracy) {
      MatchTilePairTask* other = dynamic_cast<MatchTilePairTask*>(known_good);
      double res = cv::norm(this->current_offset - other->current_offset);
      if (res < accuracy) {
        model->add_training_example(a_tile->feature_vectors[b_tile], 1, res);
        return true;
      }  else {
        model->add_training_example(a_tile->feature_vectors[b_tile], 0, res);
        return false;
      }
    }

}


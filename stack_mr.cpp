namespace tfk {

    MatchTilesTask::MatchTilesTask (Tile* tile, std::vector<Tile*> neighbors) {
      this->tile = tile;
      this->neighbors = neighbors;
      //this->param_adjustments.resize(7);
      //this->param_train_deltas.resize(7);
    }

    void MatchTilesTask::compute_tile_matches_pair(Tile* a_tile, Tile* b_tile,
      std::vector< cv::KeyPoint >& a_tile_keypoints, std::vector< cv::KeyPoint >& b_tile_keypoints,
      cv::Mat& a_tile_desc, cv::Mat& b_tile_desc,
      std::vector< cv::Point2f > &filtered_match_points_a,
      std::vector< cv::Point2f > &filtered_match_points_b, float ransac_thresh) {
    
      //std::vector<int> neighbors = get_all_close_tiles(a_tile->tile_id);
    
      if (a_tile_keypoints.size() < MIN_FEATURES_NUM) return;
      if (b_tile_keypoints.size() < MIN_FEATURES_NUM) return;
    
      // Filter the features, so that only features that are in the
      //   overlapping tile will be matches.
      std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;
    
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
    
      if (atile_kps_in_overlap.size() < MIN_FEATURES_NUM) return;
      if (btile_kps_in_overlap.size() < MIN_FEATURES_NUM) return;
    
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
    
        // Filter the matches with RANSAC
        std::vector<cv::Point2f> match_points_a, match_points_b;
        for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
          match_points_a.push_back(
              atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
          match_points_b.push_back(
              btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
        }
    
        if (matches.size() < MIN_FEATURES_NUM) {
          continue;
        }
    
        bool* mask = (bool*) calloc(match_points_a.size(), 1);
        double thresh = ransac_thresh;//5.0;
        tfk_simple_ransac(match_points_a, match_points_b, thresh, mask);
    
    
        filtered_match_points_a.clear();
        filtered_match_points_b.clear();
    
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
        if (num_matches_filtered >= MIN_FEATURES_NUM) {
          //a_tile->insert_matches(b_tile, filtered_match_points_a, filtered_match_points_b);
          break;
        } else {
          filtered_match_points_a.clear();
          filtered_match_points_b.clear();
        }
      }
    }


    void MatchTilesTask::set_random_train() {
       int index = rand()%param_train_deltas.size();
       for (int i = 0; i < param_train_deltas.size(); i++) {
         param_train_deltas[i] = 0;
       }
       int sign = rand()%2 ? -1 : 1;
       param_train_deltas[index] = sign;
    }

    //jvoid MatchTilesTask::update_result(float last_correct, float next_correct,
    //j    std::vector<int>& param_adjustments, std::vector<int>& param_train_deltas) {
    //j  if (next_correct > last_correct) {
    //j    for (int i = 0; i < param_train_deltas.size(); i++) {
    //j      param_adjustments[i] += param_train_deltas[i];
    //j      param_train_deltas[i] = 0;
    //j    }
    //j  }
    //j  printf("params:\n");
    //j  printf("scale_x %f\n", 0.25 + param_adjustments[0]*0.05);
    //j  printf("scale_y %f\n", 0.25 + param_adjustments[1]*0.05);
    //j  printf("num_features %f\n", 1.0 + param_adjustments[2]);
    //j  printf("num_octaves %f\n", 6.0 + param_adjustments[3]);
    //j  printf("contrast_thresh %f\n", 0.01 + param_adjustments[4]*0.001);
    //j  printf("edge_thresh %f\n", 20.0 + param_adjustments[5]);
    //j  printf("edge_thresh %f\n", 1.2 + param_adjustments[6]*0.05);
    //j}

    void MatchTilesTask::compute(float probability_correct, std::vector<int>& param_adjustments,
        std::vector<int>& param_train_deltas) {
      Tile* a_tile = tile;
      std::vector<cv::KeyPoint> a_tile_keypoints;
      cv::Mat a_tile_desc;
/*
scale_x 0.150000
scale_y -0.050000
num_features -1.000000
num_octaves 8.000000
contrast_thresh 0.011000
edge_thresh 17.000000
edge_thresh 1.000000
Result is 0.570000


*/
      tfk::params new_params;
      new_params.scale_x = 0.15;// + param_adjustments[0]*0.05 + param_train_deltas[0]*0.05;
      new_params.scale_y = 0.15;// + param_adjustments[1]*0.05 + param_train_deltas[1]*0.05;
      new_params.num_features = -1 + param_adjustments[2] + param_train_deltas[2];
      if (new_params.num_features < 0) new_params.num_features = 0;
      new_params.num_octaves = 8 + param_adjustments[3] + param_train_deltas[3];
      new_params.contrast_threshold = 0.011 + param_adjustments[4]*0.001 + param_train_deltas[4]*0.001;
      new_params.edge_threshold = 17 + param_adjustments[5] + param_train_deltas[5];
      new_params.sigma = 1.0 + param_adjustments[6]*0.05 + param_train_deltas[6]*0.05;

      a_tile->compute_sift_keypoints2d_params(new_params, a_tile_keypoints,
                                              a_tile_desc, a_tile);

      if (a_tile_keypoints.size() < MIN_FEATURES_NUM) return; // failure.

      //int neighbor_success_count = 0;
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
         
        std::vector<cv::KeyPoint> b_tile_keypoints;
        cv::Mat b_tile_desc;
        
        b_tile->compute_sift_keypoints2d_params(new_params, b_tile_keypoints,
                                                b_tile_desc, a_tile);
        if (b_tile_keypoints.size() < MIN_FEATURES_NUM) continue;
        
        std::vector< cv::Point2f > filtered_match_points_a(0);
        std::vector< cv::Point2f > filtered_match_points_b(0);

       
        this->compute_tile_matches_pair(a_tile, b_tile,
          a_tile_keypoints, b_tile_keypoints,
          a_tile_desc, b_tile_desc,
          filtered_match_points_a,
          filtered_match_points_b, 5.0);

        // store the matched points.
        neighbor_to_matched_points[b_tile] =
            std::make_pair(filtered_match_points_a, filtered_match_points_b);  
      }
    }

    bool MatchTilesTask::error_check(float false_negative_rate) {
      Tile* a_tile = tile;
      int neighbor_success_count = 0;
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        std::vector<cv::Point2f> filtered_match_points_a =
            neighbor_to_matched_points[b_tile].first;
        std::vector<cv::Point2f> filtered_match_points_b =
            neighbor_to_matched_points[b_tile].second;
        Tile tmp_a_tile = *a_tile;
        
        // put b at 0,0
        if (filtered_match_points_a.size() >= MIN_FEATURES_NUM) {
          for (int _i = 0; _i < 1000; _i++) {
            float dx = 0.0;
            float dy = 0.0;
            for (int j = 0; j < filtered_match_points_a.size(); j++) {
              cv::Point2f dp = b_tile->rigid_transform(filtered_match_points_b[j]) - tmp_a_tile.rigid_transform(filtered_match_points_a[j]);
              dx += 2*dp.x * 1.0 / (filtered_match_points_a.size());
              dy += 2*dp.y * 1.0 / (filtered_match_points_a.size());
            }
            tmp_a_tile.offset_x += 0.4*dx;
            tmp_a_tile.offset_y += 0.4*dy;
          }
        }
        float val = tmp_a_tile.error_tile_pair(b_tile);
        if (val >= 0.7) {
          neighbor_to_success[b_tile] = true;
          neighbor_success_count++;
        } else {
          neighbor_to_success[b_tile] = false;
        }
      }
      if (neighbor_success_count >= neighbors.size()*4.0/5.0) {
        return true;
      } else { 
        return false;
      }
    }

    void MatchTilesTask::commit() {
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        std::vector<cv::Point2f> filtered_match_points_a =
            neighbor_to_matched_points[b_tile].first;
        std::vector<cv::Point2f> filtered_match_points_b =
            neighbor_to_matched_points[b_tile].second;
        if (neighbor_to_success[b_tile]) {
          tile->insert_matches(b_tile, filtered_match_points_a, filtered_match_points_b);
        }
      }
    }

}


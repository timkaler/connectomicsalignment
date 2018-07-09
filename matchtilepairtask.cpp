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
      if (!train) {
        this->paramDB = a_tile->paramdbs[this->task_type_id];
        this->model = a_tile->ml_models[this->task_type_id];
      }
    }

    MatchTilePairTask::MatchTilePairTask (Tile* a_tile, Tile* b_tile) {
      this->a_tile = a_tile;
      this->b_tile = b_tile;
      this->task_type_id = MATCH_TILE_PAIR_TASK_ID;
      this->paramDB = a_tile->paramdbs[this->task_type_id];
      this->model = a_tile->ml_models[this->task_type_id];
    }

    //MatchTilePairTask::~MatchTilePairTask () {}

    void MatchTilePairTask::compute_tile_matches_pair(Tile* a_tile, Tile* b_tile,
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

      a_tile->keypoints_in_overlap[b_tile] = atile_kps_in_overlap.size();
      b_tile->keypoints_in_overlap[a_tile] = btile_kps_in_overlap.size();
    
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
        a_tile->matched_keypoints_in_overlap[b_tile] = matches.size();
        b_tile->matched_keypoints_in_overlap[a_tile] = matches.size();
    
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
        if (num_matches_filtered >= MIN_FEATURES_NUM && filtered_match_points_a.size() >= 0.1*matches.size()) {
          //a_tile->insert_matches(b_tile, filtered_match_points_a, filtered_match_points_b);
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

      this->mr_params = mr_params_local;

  params best_params;
  best_params.num_features = 1;
  best_params.num_octaves = 6;
  best_params.contrast_threshold = .015;//CONTRAST_THRESH;
  best_params.edge_threshold = 10;//EDGE_THRESH_2D;
  best_params.sigma = 1.6;//1.6;
  best_params.scale_x = 1.0;
  best_params.scale_y = 1.0;
  best_params.res = FULL;

      //tfk::params new_params;
      //new_params.scale_x = mr_params->get_float_param("scale");
      //new_params.scale_y = mr_params->get_float_param("scale");
      ////printf("scale %f\n", new_params.scale_x);
      ////printf("scale x %f scale y %f\n", new_params.scale_x, new_params.scale_y);
      //new_params.num_features = mr_params->get_int_param("num_features");
      //new_params.num_octaves = mr_params->get_int_param("num_octaves");
      //new_params.contrast_threshold = 0.015;//mr_params->get_float_param("contrast_threshold");
      //new_params.edge_threshold = 6;// mr_params->get_float_param("edge_threshold");
      //new_params.sigma = 1.6;//mr_params->get_float_param("sigma");


      


      //TODO(wheatman) doing extra work here, but makes it the same as the cached version
      if (dependencies.find(a_tile->tile_id) == dependencies.end()) {
        a_tile->compute_sift_keypoints2d_params(best_params, a_tile_keypoints,
                                                a_tile_desc, b_tile);
        //printf("computing A keypoints\n");
      } else {
        a_tile_desc = dependencies[a_tile->tile_id]->tile_desc;
        a_tile_keypoints = dependencies[a_tile->tile_id]->tile_keypoints;
      }

      if (a_tile_keypoints.size() < MIN_FEATURES_NUM) return; // failure.

      //int neighbor_success_count = 0;

      std::vector<cv::KeyPoint> b_tile_keypoints;
      cv::Mat b_tile_desc;

      if (dependencies.find(b_tile->tile_id) == dependencies.end()) {
        b_tile->compute_sift_keypoints2d_params(best_params, b_tile_keypoints,
                                                b_tile_desc, a_tile);
        //printf("computing B keypoints\n");
      } else {
        b_tile_desc = dependencies[b_tile->tile_id]->tile_desc;
        b_tile_keypoints = dependencies[b_tile->tile_id]->tile_keypoints;
      }
      if (b_tile_keypoints.size() < MIN_FEATURES_NUM) return;

      std::vector< cv::Point2f > filtered_match_points_a(0);
      std::vector< cv::Point2f > filtered_match_points_b(0);


      this->compute_tile_matches_pair(a_tile, b_tile,
        a_tile_keypoints, b_tile_keypoints,
        a_tile_desc, b_tile_desc,
        filtered_match_points_a,
        filtered_match_points_b, 5.0);

      // store the matched points.
      matched_points = std::make_pair(filtered_match_points_a, filtered_match_points_b);

    }
//TODO(wheatman) mark to neighbors as bad
    bool MatchTilePairTask::error_check(float false_negative_rate) {
      std::vector<cv::Point2f> filtered_match_points_a = matched_points.first;
      std::vector<cv::Point2f> filtered_match_points_b = matched_points.second;
      //a_tile->release_full_image();
      //b_tile->release_full_image();
      //a_tile->release_2d_keypoints();
      //b_tile->release_2d_keypoints();
    

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
      tmp_vector.push_back(1.0);

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
//<<<<<<< HEAD
//
//      float val = tmp_a_tile.error_tile_pair(b_tile);
//      //tmp_a_tile.get_feature_vector(b_tile, 3, 2).copyTo(a_tile->feature_vectors[b_tile]);
//      bool guess_ml = true;//this->model->predict(a_tile->feature_vectors[b_tile]);
//      a_tile->ml_preds[b_tile] = guess_ml;
//      if (/*guess_ml*/ val >= 0.75 && filtered_match_points_a.size() >= MIN_FEATURES_NUM) {
//        cv::Point2f a_point = cv::Point2f(tmp_a_tile.x_start+tmp_a_tile.offset_x,
//=======
      cv::Point2f a_point = cv::Point2f(tmp_a_tile.x_start+tmp_a_tile.offset_x,
//>>>>>>> 784ed66191eadda43e85f96b0c56c89bb3a7bcca
                                          tmp_a_tile.y_start+tmp_a_tile.offset_y);
      //TODO(wheatman) tell tim I have this done even if it failed
      cv::Point2f b_point = cv::Point2f(b_tile->x_start+b_tile->offset_x,
                                        b_tile->y_start+b_tile->offset_y);
      current_offset = a_point - b_point;

      this->predicted_offset = current_offset;
      //printf("predicted offset %f %f\n", current_offset.x, current_offset.y);

      //tmp_a_tile.get_feature_vector(b_tile, 5, 4).copyTo(a_tile->feature_vectors[b_tile]);
      // for fast path computation
      //this->feature_vector = a_tile->feature_vectors[b_tile];

      //printf("feature vector size: %d\n", feature_vector.size());

      float val = tmp_a_tile.error_tile_pair(b_tile);
      tmp_vector.push_back(val);
      this->feature_vector = tmp_vector;

       

      if (false_negative_rate > 0) {
        //bool guess_ml = this->model->predict(a_tile->feature_vectors[b_tile]);
        //bool guess_ml = false;//this->model->predict(a_tile->feature_vectors[b_tile]);
        bool guess_ml = this->model->predict(tmp_vector);//this->model->predict(a_tile->feature_vectors[b_tile]);

        a_tile->ml_preds[b_tile] = guess_ml;
        //if (false && /*guess_ml*/ val >= 0.75) {
        if (guess_ml) {
          a_tile->ideal_offsets[b_tile->tile_id] = current_offset;
          a_tile->neighbor_correlations[b_tile->tile_id] = val;
          success = true;
          return true;
        } else {
          success = false;
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


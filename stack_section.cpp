
tfk::Section::Section(int section_id) {
  this->section_id = section_id;
}




void tfk::Section::compute_tile_matches(int tile_id, Graph<vdata, edata>* graph) {

  std::vector<int> neighbors = get_all_close_tiles(tile_id);

  Tile* a_tile = this->tiles[tile_id];

  for (int i = 0; i < neighbors.size(); i++) {
    int atile_id = tile_id;
    int btile_id = neighbors[i];
    Tile* b_tile = this->tiles[btile_id];

    if (a_tile->p_kps->size() < MIN_FEATURES_NUM) continue;
    if (b_tile->p_kps->size() < MIN_FEATURES_NUM) continue;

    // Filter the features, so that only features that are in the
    //   overlapping tile will be matches.
    std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;
    atile_kps_in_overlap.reserve(a_tile->p_kps->size());
    btile_kps_in_overlap.reserve(b_tile->p_kps->size());
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
    } // End scoped block A

    if (atile_kps_in_overlap.size() < MIN_FEATURES_NUM) continue;
    if (btile_kps_in_overlap.size() < MIN_FEATURES_NUM) continue;

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
  }
}



void tfk::Section::compute_keypoints_and_matches() {
  // assume that section data doesn't exist.


  cilk_for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    tile->compute_sift_keypoints2d();
    tile->compute_sift_keypoints3d();
  }

  graph = new Graph<vdata, edata>();
  graph->resize(this->tiles.size());

  cilk_for (int i = 0; i < this->tiles.size(); i++) {
    this->compute_tile_matches(i, graph);
  }

    for (int i = 0; i < graph->num_vertices(); i++) {
      vdata* d = graph->getVertexData(i);
      //_tile_data tdata = p_sec_data->tiles[i];
      Tile* tile = this->tiles[i];
      d->vertex_id = i;
      d->mfov_id = tile->mfov_id;
      d->tile_index = tile->index;
      d->tile_id = i;
      d->start_x = tile->x_start;
      d->end_x = tile->x_finish;
      d->start_y = tile->y_start;
      d->end_y = tile->y_finish;
      d->offset_x = 0.0;
      d->offset_y = 0.0;
      d->iteration_count = 0;
      //d->last_radius_value = 9.0;
      d->z = /*p_align_data->base_section + */this->section_id;
      d->a00 = 1.0;
      d->a01 = 0.0;
      d->a10 = 0.0;
      d->a11 = 1.0;
      //d->neighbor_grad_x = 0.0;
      //d->neighbor_grad_y = 0.0;
      //d->converged = 0;
      d->original_center_point =
        cv::Point2f((tile->x_finish-tile->x_start)/2,
                    (tile->y_finish-tile->y_start)/2);
    }

    graph->section_id = this->section_id;

  // now compute keypoint matches
  //cilk_for (int i = 0; i < this->tiles.size(); i++) {
  //    compute_tile_matches_active_set(p_align_data, sec_id, active_set, graph);
  //}

}



std::vector<int> tfk::Section::get_all_close_tiles(int atile_id) {
  std::vector<int> neighbor_index_list(0);

  int indices_to_check_len = 0;

  Tile* a_tile = this->tiles[atile_id];
  for (int i = atile_id+1; i < this->tiles.size(); i++) {
    Tile* b_tile = this->tiles[i];
    if (a_tile->overlaps_with(b_tile)) {
      neighbor_index_list.push_back(i);
    }
  }

  return neighbor_index_list;
}

// Section from protobuf
tfk::Section::Section(SectionData& section_data) {
  //section_data_t *p_sec_data = &(p_tile_data->sec_data[i - p_tile_data->base_section]);

  this->section_id = section_data.section_id();
  this->n_tiles = 0;
  if (section_data.has_out_d1()) {
    this->out_d1 = section_data.out_d1();
  }
  if (section_data.has_out_d2()) {
    this->out_d2 = section_data.out_d2();
  }


  for (int j = 0; j < section_data.tiles_size(); j++) {
    TileData tile_data = section_data.tiles(j);

    Tile* tile = new Tile(tile_data);
    this->tiles.push_back(tile);
  }

}



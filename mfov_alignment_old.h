

void get_mfov_graph(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data,
                    std::map<std::pair<int, int>, int>& mfov_to_id,
                    Graph<vdata, edata>* mfov_graph) {

  //global_learning_rate = 0.05;
  //Graph<vdata, edata>* mfov_graph = new Graph<vdata, edata>();
  std::set<std::pair<int, int> > mfov_added_set; 
  std::vector<std::pair<int, int> > mfov_added_list;
  //std::map<std::pair<int, int>, int> mfov_to_id;
  mfov_to_id.clear();

  std::map<int, std::pair<double, double> > mfov_start_end;
  std::map<int, std::pair<int, int> > mfov_id_to_key;


  int count = 0;
  for (int v = 0; v < merged_graph->num_vertices(); v++) {
    int z = merged_graph->getVertexData(v)->z;
    int mfov_id = merged_graph->getVertexData(v)->mfov_id;
    std::pair<int, int> key = std::make_pair(z,mfov_id);
    if (mfov_added_set.find(key) == mfov_added_set.end()) {
      mfov_added_set.insert(key);
      mfov_added_list.push_back(key);
      mfov_to_id[key] = count++;
      mfov_id_to_key[mfov_to_id[key]] = key;
    }
  }



  printf("There are %d mfovs\n", mfov_added_list.size());

  mfov_graph->resize(mfov_added_list.size());

  for (int v = 0; v < merged_graph->num_vertices(); v++) {
    int z = merged_graph->getVertexData(v)->z;
    int mfov_id = merged_graph->getVertexData(v)->mfov_id;
    std::pair<int, int> key = std::make_pair(z,mfov_id);
    vdata* my_data = merged_graph->getVertexData(v);
    int my_id = mfov_to_id[key];
    for (int i = 0; i < merged_graph->edgeData[v].size(); i++) {
      edata edge = merged_graph->edgeData[v][i];
      vdata* neighbor_data = merged_graph->getVertexData(edge.neighbor_id);
      std::pair<int, int> neighbor_key =
          std::make_pair(neighbor_data->z, neighbor_data->mfov_id);
      int neighbor_id = mfov_to_id[neighbor_key];
      if (my_id < neighbor_id) {
        std::vector<cv::Point2f> my_points, neighbor_points;
        for (int j = 0; j < edge.v_points->size(); j++) {
          my_points.push_back(transform_point(my_data, (*edge.v_points)[j]));
          neighbor_points.push_back(transform_point(neighbor_data, (*edge.n_points)[j]));
        }
        mfov_graph->insert_matches(my_id, neighbor_id, my_points, neighbor_points, 1.0);
      }
    }
  }

    for (int i = 0; i < mfov_graph->num_vertices(); i++) {
      vdata* d = mfov_graph->getVertexData(i);
      //_tile_data tdata = p_sec_data->tiles[i];
      d->vertex_id = i;
      d->mfov_id = mfov_id_to_key[i].second;
      d->tile_index = i;
      d->tile_id = i;
      d->start_x = 0.0;
      d->end_x = 0.0;
      d->start_y = 0.0;
      d->end_y = 0.0;
      d->offset_x = 0.0;
      d->offset_y = 0.0;
      d->iteration_count = 0;
      d->last_radius_value = 9.0;
      d->z = /*p_align_data->base_section + */mfov_id_to_key[i].first;
      d->a00 = 1.0;
      d->a01 = 0.0;
      d->a10 = 0.0;
      d->a11 = 1.0;
      d->neighbor_grad_x = 0.0;
      d->neighbor_grad_y = 0.0;
      d->converged = 0;
    }

    //int ncolors = mfov_graph->compute_trivial_coloring();
    //Scheduler* scheduler;
    //engine<vdata, edata>* e;
    //scheduler =
    //    new Scheduler(mfov_graph->vertexColors, ncolors+1, mfov_graph->num_vertices());
    //scheduler->graph_void = (void*) mfov_graph;
    //e = new engine<vdata, edata>(mfov_graph, scheduler);

    //for (int i = 0; i < mfov_graph->num_vertices(); i++) {
    //  scheduler->add_task(i, updateVertex2DAlignMFOV);
    //}
    //printf("begin mfov run\n");
    //e->run();
    //printf("end mfov run\n");

    //std::map<std::pair<int, int>, std::pair<double, double> > offset_mfov_map;
    //for (int i = 0; i < mfov_graph->num_vertices(); i++) {
    //  vdata* vd = mfov_graph->getVertexData(i);
    //  std::pair<int, int> mfov_key = mfov_id_to_key[i];
    //  offset_mfov_map[mfov_key] = std::make_pair(vd->offset_x, vd->offset_y);
    //  printf("offset for mfov z=%d mfov=%d is %f, %f\n", vd->z, vd->mfov_id, vd->offset_x, vd->offset_y);
    //}
    //for (int v = 0; v < merged_graph->num_vertices(); v++) {
    //  int z = merged_graph->getVertexData(v)->z;
    //  int mfov_id = merged_graph->getVertexData(v)->mfov_id;
    //  std::pair<int, int> key = std::make_pair(z,mfov_id);
    //  std::pair<double, double> offset_pair = offset_mfov_map[key];
    //  //printf("offset for mfov z=%d mfov=%d is %f, %f\n", z, mfov_id, offset_pair.first, offset_pair.second);
    //  vdata* my_data = merged_graph->getVertexData(v);
    //  my_data->offset_x += offset_pair.first;
    //  my_data->offset_y += offset_pair.second;
    //}
}

void mfov_alignment_3d(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data) {

  global_learning_rate = 0.05;
  Graph<vdata, edata>* mfov_graph = new Graph<vdata, edata>();
  std::set<std::pair<int, int> > mfov_added_set; 
  std::vector<std::pair<int, int> > mfov_added_list;
  std::map<std::pair<int, int>, int> mfov_to_id;

  std::map<int, std::pair<double, double> > mfov_start_end;
  std::map<int, std::pair<int, int> > mfov_id_to_key;


  int count = 0;
  for (int v = 0; v < merged_graph->num_vertices(); v++) {
    int z = merged_graph->getVertexData(v)->z;
    int mfov_id = merged_graph->getVertexData(v)->mfov_id;
    std::pair<int, int> key = std::make_pair(z,mfov_id);
    if (mfov_added_set.find(key) == mfov_added_set.end()) {
      mfov_added_set.insert(key);
      mfov_added_list.push_back(key);
      mfov_to_id[key] = count++;
      mfov_id_to_key[mfov_to_id[key]] = key;
    }
  }



  printf("There are %d mfovs\n", mfov_added_list.size());

  mfov_graph->resize(mfov_added_list.size());

  for (int v = 0; v < merged_graph->num_vertices(); v++) {
    int z = merged_graph->getVertexData(v)->z;
    int mfov_id = merged_graph->getVertexData(v)->mfov_id;
    std::pair<int, int> key = std::make_pair(z,mfov_id);
    vdata* my_data = merged_graph->getVertexData(v);
    int my_id = mfov_to_id[key];
    for (int i = 0; i < merged_graph->edgeData[v].size(); i++) {
      edata edge = merged_graph->edgeData[v][i];
      vdata* neighbor_data = merged_graph->getVertexData(edge.neighbor_id);
      std::pair<int, int> neighbor_key =
          std::make_pair(neighbor_data->z, neighbor_data->mfov_id);
      int neighbor_id = mfov_to_id[neighbor_key];
      if (my_id < neighbor_id) {
        std::vector<cv::Point2f> my_points, neighbor_points;
        for (int j = 0; j < edge.v_points->size(); j++) {
          my_points.push_back(transform_point(my_data, (*edge.v_points)[j]));
          neighbor_points.push_back(transform_point(neighbor_data, (*edge.n_points)[j]));
        }
        mfov_graph->insert_matches(my_id, neighbor_id, my_points, neighbor_points, 1.0);
      }
    }
  }

    for (int i = 0; i < mfov_graph->num_vertices(); i++) {
      vdata* d = mfov_graph->getVertexData(i);
      //_tile_data tdata = p_sec_data->tiles[i];
      d->vertex_id = i;
      d->mfov_id = mfov_id_to_key[i].second;
      d->tile_index = i;
      d->tile_id = i;
      d->start_x = 0.0;
      d->end_x = 0.0;
      d->start_y = 0.0;
      d->end_y = 0.0;
      d->offset_x = 0.0;
      d->offset_y = 0.0;
      d->iteration_count = 0;
      d->last_radius_value = 9.0;
      d->z = /*p_align_data->base_section + */mfov_id_to_key[i].first;
      d->a00 = 1.0;
      d->a01 = 0.0;
      d->a10 = 0.0;
      d->a11 = 1.0;
      d->neighbor_grad_x = 0.0;
      d->neighbor_grad_y = 0.0;
      d->converged = 0;
    }

    int ncolors = mfov_graph->compute_trivial_coloring();
    Scheduler* scheduler;
    engine<vdata, edata>* e;
    scheduler =
        new Scheduler(mfov_graph->vertexColors, ncolors+1, mfov_graph->num_vertices());
    scheduler->graph_void = (void*) mfov_graph;
    e = new engine<vdata, edata>(mfov_graph, scheduler);

    for (int i = 0; i < mfov_graph->num_vertices(); i++) {
      scheduler->add_task(i, updateVertex2DAlignMFOV);
    }
    printf("begin mfov run\n");
    e->run();
    printf("end mfov run\n");

    std::map<std::pair<int, int>, std::pair<double, double> > offset_mfov_map;
    for (int i = 0; i < mfov_graph->num_vertices(); i++) {
      vdata* vd = mfov_graph->getVertexData(i);
      std::pair<int, int> mfov_key = mfov_id_to_key[i];
      offset_mfov_map[mfov_key] = std::make_pair(vd->offset_x, vd->offset_y);
      printf("offset for mfov z=%d mfov=%d is %f, %f\n", vd->z, vd->mfov_id, vd->offset_x, vd->offset_y);
    }
    for (int v = 0; v < merged_graph->num_vertices(); v++) {
      int z = merged_graph->getVertexData(v)->z;
      int mfov_id = merged_graph->getVertexData(v)->mfov_id;
      std::pair<int, int> key = std::make_pair(z,mfov_id);
      std::pair<double, double> offset_pair = offset_mfov_map[key];
      //printf("offset for mfov z=%d mfov=%d is %f, %f\n", z, mfov_id, offset_pair.first, offset_pair.second);
      vdata* my_data = merged_graph->getVertexData(v);
      my_data->offset_x += offset_pair.first;
      my_data->offset_y += offset_pair.second;
    }
}

void fine_alignment_3d_mfov(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data){

    printf("STARTING THE FINE ALIGNMENT IT SO FINE (I hope)");
    int vertex_id_offset = 0;

    for (int v = 0; v < merged_graph->num_vertices(); v++) {
      merged_graph->getVertexData(v)->offset_x += merged_graph->getVertexData(v)->start_x;
      merged_graph->getVertexData(v)->offset_y += merged_graph->getVertexData(v)->start_y;
      merged_graph->getVertexData(v)->start_x = 0.0;
      merged_graph->getVertexData(v)->start_y = 0.0;
      merged_graph->getVertexData(v)->end_x = 0.0;
      merged_graph->getVertexData(v)->end_y = 0.0;
    }


    double min_x = merged_graph->getVertexData(0)->start_x;
    double min_y = merged_graph->getVertexData(0)->start_x;
    double max_x = merged_graph->getVertexData(0)->start_y;
    double max_y = merged_graph->getVertexData(0)->start_y;
    for (int v = 0; v < merged_graph->num_vertices(); v++) {
      double vx = merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
      double vy = merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
      if (vx < min_x) min_x = vx;
      if (vx > max_x) max_x = vx;
      if (vy < min_y) min_y = vy;
      if (vy > max_y) max_y = vy;
    }
    std::map<std::pair<int, int>, int> mfov_to_id;
    Graph<vdata, edata>* mfov_graph = new Graph<vdata, edata>();
    get_mfov_graph(merged_graph, p_align_data,
                    mfov_to_id,
                    mfov_graph);

    for (int v = 0; v < merged_graph->num_vertices(); v++) {
      merged_graph->getVertexData(v)->center_point = transform_point(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
    }

    for (int iter = 0; iter < 1; iter++) {
    printf("Begin iter %d\n", iter);
    for (int section = 1; section < p_align_data->n_sections; section++) {

        int section_a = section;
        int section_b = section-1;

        // separate out the mfovs.
        std::set<int> mfov_set;
        std::vector<int> mfov_list;
        std::map<int, std::vector<int> > mfov_id_to_tiles;
        for (int v = 0; v < merged_graph->num_vertices(); v++) {
          vdata* vd = merged_graph->getVertexData(v);
          if (vd->z != section_a) continue;
          if (mfov_set.find(vd->mfov_id) == mfov_set.end()) {
            mfov_set.insert(vd->mfov_id);
            mfov_list.push_back(vd->mfov_id);
          }
          mfov_id_to_tiles[vd->mfov_id].push_back(v);
        }

        std::vector<int> tile_ids_b(0);
        for (int v = 0; v < merged_graph->num_vertices(); v++) {
          vdata* vd = merged_graph->getVertexData(v);
          if (vd->z != section_b) continue;
          tile_ids_b.push_back(v);
        }

        if (tile_ids_b.size() == 0) continue;

        for (int mfov_index = 0; mfov_index < mfov_list.size(); mfov_index++) {
          int mfov = mfov_list[mfov_index];
          std::vector<int> mfov_tiles = mfov_id_to_tiles[mfov];
          vdata* vd_mfov = merged_graph->getVertexData(mfov_tiles[0]);
          double mfov_start_x = vd_mfov->start_x + vd_mfov->offset_x;
          double mfov_start_y = vd_mfov->start_y + vd_mfov->offset_y;
          double mfov_end_x = mfov_start_x + vd_mfov->original_center_point.x*2;
          double mfov_end_y = mfov_start_y + vd_mfov->original_center_point.y*2;
          for (int i = 0; i < mfov_tiles.size(); i++) {
            vdata* vd_mfov = merged_graph->getVertexData(mfov_tiles[i]);
            mfov_start_x = std::min(mfov_start_x, vd_mfov->start_x + vd_mfov->offset_x);
            mfov_start_y = std::min(mfov_start_y, vd_mfov->start_y + vd_mfov->offset_y);
            mfov_end_x = std::max(mfov_end_x, mfov_start_x + vd_mfov->original_center_point.x*2);
            mfov_end_y = std::max(mfov_end_y, mfov_start_y + vd_mfov->original_center_point.y*2);
          }
        mfov_start_x -= 3000.0;
        mfov_start_y -= 3000.0;
        mfov_end_x += 3000.0;
        mfov_end_y += 3000.0;

        // Filter the matches with RANSAC
        int num_filtered = 0;
        std::vector< cv::Point2f > filtered_match_points_a(0);
        std::vector< cv::Point2f > filtered_match_points_b(0);
        std::vector<cv::Point2f> match_points_a, match_points_b;
        std::vector <cv::KeyPoint > atile_kps_in_overlap;
        std::vector <cv::Mat > atile_kps_desc_in_overlap_list;
        std::vector<int> atile_kps_tile_list;
        std::vector <cv::KeyPoint > btile_kps_in_overlap;
        std::vector <cv::Mat > btile_kps_desc_in_overlap_list;
        std::vector<int> btile_kps_tile_list;

        std::vector<double> atile_weights;
        std::vector<double> btile_weights;
        //std::set<int> tile_id_set;
        //tile_id_set.clear();

        // for everything in this mfov concat the matches into the a list.
        for (int i = 0; i < mfov_tiles.size(); i++) {
            int v = mfov_tiles[i];
            int curr_z = merged_graph->getVertexData(v)->z;
            _tile_data tdata_a = p_align_data->sec_data[curr_z].tiles[merged_graph->getVertexData(v)->tile_id];
            concat_two_tiles_all(merged_graph->getVertexData(v), &tdata_a, v, atile_kps_in_overlap, atile_kps_desc_in_overlap_list, atile_kps_tile_list);
        }

        std::vector<int> section_b_overlap_tiles;
        // for all the tiles in this mfov find the affine alignment with the mfov below it.
        for (int i = 0; i < tile_ids_b.size(); i++) {
          vdata* vd = merged_graph->getVertexData(tile_ids_b[i]);
          double start_x = vd->start_x+vd->offset_x;
          double start_y = vd->start_y+vd->offset_y;
          double end_x = start_x+vd->original_center_point.x*2;
          double end_y = start_y+vd->original_center_point.y*2;

          double overlap = (std::min(end_x, mfov_end_x) - std::max(start_x, mfov_start_x)) *
                           (std::min(end_y, mfov_end_y) - std::max(start_y, mfov_start_y));
          if (overlap > 2000.0*2000.0) {
            section_b_overlap_tiles.push_back(tile_ids_b[i]); 
            int v = tile_ids_b[i];
            int curr_z = merged_graph->getVertexData(v)->z;
            _tile_data tdata_b = p_align_data->sec_data[section_b].tiles[merged_graph->getVertexData(v)->tile_id];
            concat_two_tiles_all_filter(merged_graph->getVertexData(v),&tdata_b, v, btile_kps_in_overlap, btile_kps_desc_in_overlap_list, btile_kps_tile_list, mfov_start_x, mfov_start_y, mfov_end_x, mfov_end_y);
          }
        }
        if (section_b_overlap_tiles.size() == 0) {
          printf("An mfov had no overlap, skipping\n");
          continue;
        }

        printf("Total size of a tile kps is %d\n", atile_kps_in_overlap.size());
        printf("Total size of b tile kps is %d\n", btile_kps_in_overlap.size());
        //if (atile_kps_tile_list.size() == 0 || btile_kps_tile_list.size() == 0) continue;

        cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
        cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
        cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));
        bool done = false;

        std::vector< cv::DMatch > matches;
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       0.65);

        printf("Done with the matching. Num matches is %d\n", matches.size());
        if (matches.size() == 0) continue;

              for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
                int atile_id = atile_kps_tile_list[matches[tmpi].queryIdx];
                int btile_id = btile_kps_tile_list[matches[tmpi].trainIdx];

                match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
                match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
              }

                bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
                tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 20.0, mask);

                for (int c = 0; c < match_points_a.size(); c++) {
                  if (mask[c]) {
                    num_filtered++;
                    filtered_match_points_a.push_back(
                        match_points_a[c]);
                    filtered_match_points_b.push_back(
                        match_points_b[c]);
                  }
                }
                free(mask);

          printf("Second pass filter got %d matches\n", num_filtered);

          if (/*num_filtered < 0.05*filtered_match_points_a_pre.size() *//*matches.size()*/ /*||*/ num_filtered < 12) {
            //printf("Not enough matches %d for section %d with thresh %f\n", num_filtered, section_a, thresh_for_trial);
            printf("Not enough matches %d for section %d with thresh\n", num_filtered, section_a);
            //continue;
            // NOTE(TFK): Clear instead of just continuing so that we affine align within section.
            filtered_match_points_a.clear();
            filtered_match_points_b.clear();
          } else {
            //printf("Got enough matches %d for section %d with thresh %f\n", num_filtered, section_a, thresh_for_trial);
            printf("Got enough matches %d for section %d with thresh\n", num_filtered, section_a);
            //done = true;
          }


          std::pair<int, int> mfov_key = std::make_pair(section_a, mfov);
          int mfov_vid = mfov_to_id[mfov_key];
          vdata* mfov_vdata = mfov_graph->getVertexData(mfov_vid);
          for (int p = 0; p < mfov_graph->edgeData[mfov_vid].size(); p++) {
            std::vector<cv::Point2f>* v_points = mfov_graph->edgeData[mfov_vid][p].v_points;
            std::vector<cv::Point2f>* n_points = mfov_graph->edgeData[mfov_vid][p].n_points;
            int nid = mfov_graph->edgeData[mfov_vid][p].neighbor_id;
            for (int p2 = 0; p2 < v_points->size(); p2++) {
              filtered_match_points_a.push_back(transform_point(mfov_graph->getVertexData(mfov_vid), (*v_points)[p2]));
              filtered_match_points_b.push_back(transform_point(mfov_graph->getVertexData(nid), (*n_points)[p2]));
            }
          }

          cv::Mat warp_mat;
          cv::computeAffineTFK(filtered_match_points_a, filtered_match_points_b, warp_mat);

          std::cout << warp_mat << std::endl; 
          vdata tmp;
          tmp.a00 = warp_mat.at<double>(0, 0); 
          tmp.a01 = warp_mat.at<double>(0, 1);
          tmp.offset_x = warp_mat.at<double>(0, 2);
          tmp.a10 = warp_mat.at<double>(1, 0); 
          tmp.a11 = warp_mat.at<double>(1, 1); 
          tmp.offset_y = warp_mat.at<double>(1, 2);
          tmp.start_x = 0.0;
          tmp.start_y = 0.0;
          printf("Best values are %f %f %f %f %f %f\n", tmp.a00, tmp.a01, tmp.a10, tmp.a11, tmp.offset_x, tmp.offset_y);
          vdata best_vertex_data = tmp;

        for (int i = 0; i < mfov_tiles.size(); i++) {
            int v = mfov_tiles[i];
            updateAffineTransform(merged_graph->getVertexData(v), /*&best_vertex_data*/warp_mat);
        }
            updateAffineTransform(mfov_graph->getVertexData(mfov_vid), /*&best_vertex_data*/warp_mat);
      }
    }
  }
}



//void fine_alignment_3d_dampen(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data){
//    printf("STARTING THE FINE ALIGNMENT IT SO FINE (I hope)");
//    if (true) {
//    int vertex_id_offset = 0;
//
//    for (int v = 0; v < merged_graph->num_vertices(); v++) {
//      merged_graph->getVertexData(v)->offset_x += merged_graph->getVertexData(v)->start_x;
//      merged_graph->getVertexData(v)->offset_y += merged_graph->getVertexData(v)->start_y;
//      merged_graph->getVertexData(v)->start_x = 0.0;
//      merged_graph->getVertexData(v)->start_y = 0.0;
//      merged_graph->getVertexData(v)->end_x = 0.0;
//      merged_graph->getVertexData(v)->end_y = 0.0;
//    }
//
//
//    double min_x = merged_graph->getVertexData(0)->start_x;
//    double min_y = merged_graph->getVertexData(0)->start_x;
//    double max_x = merged_graph->getVertexData(0)->start_y;
//    double max_y = merged_graph->getVertexData(0)->start_y;
//    for (int v = 0; v < merged_graph->num_vertices(); v++) {
//      double vx = merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
//      double vy = merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
//      if (vx < min_x) min_x = vx;
//      if (vx > max_x) max_x = vx;
//      if (vy < min_y) min_y = vy;
//      if (vy > max_y) max_y = vy;
//    }
//
//    for (int v = 0; v < merged_graph->num_vertices(); v++) {
//      merged_graph->getVertexData(v)->center_point = transform_point(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
//    }
//
//
//
//
//      for (double box_iter_x = min_x; box_iter_x < max_x+29000; box_iter_x += 24000) {
//      for (double box_iter_y = min_y; box_iter_y < max_y+29000; box_iter_y += 24000) {
//    for (int section = 1; section < p_align_data->n_sections; section++) {
//        // Filter the matches with RANSAC
//        int section_a = section-1;
//        int section_b = section;
//                int num_filtered = 0;
//                std::vector< cv::Point2f > filtered_match_points_a(0);
//                std::vector< cv::Point2f > filtered_match_points_b(0);
//        std::vector<cv::Point2f> match_points_a, match_points_b;
//        double box_min_x = box_iter_x;//min_x + (max_x-min_x)*box_iter_x*1.0/20;
//        double box_max_x = box_iter_x+24000;//min_x + (max_x-min_x)*(box_iter_x+1)*1.0/20;
//        double box_min_y = box_iter_y;//min_y + (max_y-min_y)*box_iter_y*1.0/20;
//        double box_max_y = box_iter_y+24000;//min_y + (max_y-min_y)*(box_iter_y+1)*1.0/20;
//        std::vector <cv::KeyPoint > atile_kps_in_overlap;
//        std::vector <cv::Mat > atile_kps_desc_in_overlap_list;
//        std::vector<int> atile_kps_tile_list;
//        std::vector <cv::KeyPoint > btile_kps_in_overlap;
//        std::vector <cv::Mat > btile_kps_desc_in_overlap_list;
//        std::vector<int> btile_kps_tile_list;
//
//        std::vector<double> atile_weights;
//        std::vector<double> btile_weights;
//
//
//        std::set<int> tile_id_set;
//        tile_id_set.clear();
//        for (int v = 0; v < merged_graph->num_vertices(); v++) {
//          if (merged_graph->edgeData[v].size() == 0) continue;
//            cv::Point2f center = merged_graph->getVertexData(v)->center_point;//transform_point(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
//            double vx = 1.0*((double) center.x);//merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
//            double vy = 1.0*((double) center.y);//merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
//          //double vx = merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
//          //double vy = merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
//          //if (vx < box_min_x || vx > box_max_x || vy < box_min_y || vy > box_max_y) continue;
//          if (vx < box_min_x-6000.0 || vx > box_max_x+6000.0 || vy < box_min_y -6000.0 || vy > box_max_y+6000.0) continue;
//
//          if (merged_graph->getVertexData(v)->z == section_a/* || merged_graph->getVertexData(v)->z == section_a-1*/) {
//            tile_id_set.insert(v);
//            int curr_z = merged_graph->getVertexData(v)->z;
//            _tile_data tdata_a = p_align_data->sec_data[curr_z].tiles[merged_graph->getVertexData(v)->tile_id];
//            //printf("The center is at %f, %f\n", center.x, center.y);
//            //printf("Original center was at %f, %f\n", merged_graph->getVertexData(v)->original_center_point.x, merged_graph->getVertexData(v)->original_center_point.y);
//            //printf("Best values are %f %f %f %f %f %f\n", merged_graph->getVertexData(v)->a00, merged_graph->getVertexData(v)->a01, merged_graph->getVertexData(v)->a10, merged_graph->getVertexData(v)->a11, merged_graph->getVertexData(v)->offset_x, merged_graph->getVertexData(v)->offset_y);
//            concat_two_tiles_all_filter(merged_graph->getVertexData(v), &tdata_a, v, atile_kps_in_overlap, atile_kps_desc_in_overlap_list, atile_kps_tile_list, box_min_x, box_min_y, box_max_x, box_max_y);
//          } else if (merged_graph->getVertexData(v)->z == section_b) {
//            _tile_data tdata_b = p_align_data->sec_data[section_b].tiles[merged_graph->getVertexData(v)->tile_id];
//            concat_two_tiles_all_filter(merged_graph->getVertexData(v),&tdata_b, v, btile_kps_in_overlap, btile_kps_desc_in_overlap_list, btile_kps_tile_list, box_min_x, box_min_y, box_max_x, box_max_y);
//          }
//        }
//
//
//
//        printf("Total size of a tile kps is %d\n", atile_kps_in_overlap.size());
//        printf("Total size of b tile kps is %d\n", btile_kps_in_overlap.size());
//        if (atile_kps_tile_list.size() == 0 || btile_kps_tile_list.size() == 0) continue;
//
//
//
//        cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
//        cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
//        cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));
//        bool done = false;
//
//        //for (int trial_thresh = 0; trial_thresh < 4; trial_thresh++) {
//        //  if (done) break;
//        //for (int trial = 0; trial < 6; trial++) {
//        //  if (done) break;
//        //  float trial_rod = 0.55;
//        //  if (trial == 1) trial_rod = 0.6;
//        //  if (trial == 2) trial_rod = 0.7;
//        //  if (trial == 3) trial_rod = 0.8;
//        //  if (trial == 4) trial_rod = 0.92;
//        //  if (trial == 5) trial_rod = 0.96;
//
//        //  double thresh_for_trial = 4.0;
//        //  if (trial_thresh == 1) thresh_for_trial = 5.0;
//        //  if (trial_thresh == 2) thresh_for_trial = 6.0;
//        //  if (trial_thresh == 3) thresh_for_trial = 7.0;
//
//          std::vector< cv::DMatch > matches;
//          match_features(matches,
//                         atile_kps_desc_in_overlap,
//                         btile_kps_desc_in_overlap,
//                         0.65);
//          if (matches.size() == 0) continue;
//
//
//
//              printf("Done with the matching. Num matches is %d\n", matches.size());
//
//              p_align_data->sec_data[section_a].p_kps = new std::vector<cv::KeyPoint>();
//              p_align_data->sec_data[section_b].p_kps = new std::vector<cv::KeyPoint>();
//
//
//
//              for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
//                int atile_id = atile_kps_tile_list[matches[tmpi].queryIdx];
//                int btile_id = btile_kps_tile_list[matches[tmpi].trainIdx];
//
//                //int x_start_a = merged_graph->getVertexData(atile_id)->start_x + merged_graph->getVertexData(atile_id)->offset_x;
//                //int y_start_a = merged_graph->getVertexData(atile_id)->start_y + merged_graph->getVertexData(atile_id)->offset_y;
//
//                //int x_start_b = merged_graph->getVertexData(btile_id)->start_x + merged_graph->getVertexData(btile_id)->offset_x;
//                //int y_start_b = merged_graph->getVertexData(btile_id)->start_y + merged_graph->getVertexData(btile_id)->offset_y;
//
//                match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
//                match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
//
//                //p_align_data->sec_data[section_a].p_kps->push_back(atile_kps_in_overlap[matches[tmpi].queryIdx]);
//                //p_align_data->sec_data[section_b].p_kps->push_back(btile_kps_in_overlap[matches[tmpi].trainIdx]);
//
//                //match_points_a.push_back(
//                //    atile_kps_in_overlap[matches[tmpi].queryIdx].pt + cv::Point2f(x_start_a, y_start_a));
//                //match_points_b.push_back(
//                //    btile_kps_in_overlap[matches[tmpi].trainIdx].pt + cv::Point2f(x_start_b, y_start_b));
//              }
//
//             //   int left_box_count = 0; 
//             //   int right_box_count = 0; 
//             //   for (int v2 = 0; v2 < merged_graph->num_vertices(); v2++) {
//             //     cv::Point2f center = merged_graph->getVertexData(v2)->center_point;//transform_point(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
//             //     double vx = 1.0*((double) center.x);//merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
//             //     double vy = 1.0*((double) center.y);//merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
//             //     bool compute = false;
//             //     // left box.
//             //     if (vx <= box_max_x-24000.0 && vx >= box_min_x-24000.0 && vy >= box_min_y && vy <= box_max_y) {
//             //       left_box_count++;
//             //       compute = true;
//
//             //     } 
//             //     if (vy <= box_max_y-24000.0 && vy >= box_min_y-24000.0 && vx >= box_min_x && vx <= box_max_x) {
//             //       right_box_count++;
//             //       compute = true;
//             //     }
//             //     if (compute) {
//             //       for (int edge = 0; edge < merged_graph->edgeData[v2].size(); edge++) {
//             //         edata edge_data = merged_graph->edgeData[v2][edge];
//             //         if (tile_id_set.find(edge_data.neighbor_id) == tile_id_set.end()) {
//             //           continue;
//             //         }
//             //         for (int iter_n = 0; iter_n < edge_data.v_points->size(); iter_n++) {
//             //           match_points_a.push_back(transform_point(merged_graph->getVertexData(edge_data.neighbor_id), (*edge_data.n_points)[iter_n]));
//             //           match_points_b.push_back(transform_point(merged_graph->getVertexData(v2), (*edge_data.v_points)[iter_n]));
//             //         }
//             //       }
//             //     }
//             //   }
//             //   printf("I saw left matches %d and right matches %d\n", left_box_count, right_box_count);
//
//
//                bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
//                tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 50.0, mask);
//
//                for (int c = 0; c < match_points_a.size(); c++) {
//                  if (mask[c]) {
//                    num_filtered++;
//                    filtered_match_points_a.push_back(
//                        match_points_a[c]);
//                    filtered_match_points_b.push_back(
//                        match_points_b[c]);
//                  }
//                }
//                //int num_fake_matches = 10;
//
//                //
//
//                //if (num_fake_matches < match_points_a.size()/10) {
//                //  num_fake_matches = match_points_a.size()/10;
//                //}
//                //for (int fake = 0; fake < num_fake_matches; fake++) {
//                //  filtered_match_points_a.push_back(cv::Point2f(box_iter_x, box_iter_y + 24000.0/20));
//                //  filtered_match_points_a.push_back(cv::Point2f(box_iter_x+24000.0/20, box_iter_y));
//                //}
//
//                free(mask);
//
//              //}
//              //}
//
//              //bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
//              //std::pair<double,double> offset_pair;
//              //tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 1024.0, mask);
//              //std::vector< cv::Point2f > filtered_match_points_a_pre(0);
//              //std::vector< cv::Point2f > filtered_match_points_b_pre(0);
//
//          //int num_filtered = 0;
//          //for (int c = 0; c < match_points_a.size(); c++) {
//          //  if (mask[c]|| true) {
//          //    num_filtered++;
//          //    //int atile_id = atile_kps_tile_list[matches[c].queryIdx];
//          //    //int btile_id = btile_kps_tile_list[matches[c].trainIdx];
//          //    filtered_match_points_a_pre.push_back(
//          //        match_points_a[c]);
//          //    filtered_match_points_b_pre.push_back(
//          //        match_points_b[c]);
//          //  }
//          //}
//          //free(mask); 
//          //mask = (bool*)calloc(match_points_a.size()+1, 1);
//          //printf("First pass filter got %d matches\n", num_filtered);
//          //vdata best_vertex_data  = tfk_simple_ransac_strict_ret_affine(filtered_match_points_a_pre, filtered_match_points_b_pre, 10.0, mask);
//
//          //num_filtered = 0;
//          //for (int c = 0; c < filtered_match_points_a_pre.size(); c++) {
//          //  if (mask[c]) {
//          //    num_filtered++;
//          //    filtered_match_points_a.push_back(
//          //        filtered_match_points_a_pre[c]);
//          //    filtered_match_points_b.push_back(
//          //        filtered_match_points_b_pre[c]);
//          //  }
//          //}
//          printf("Second pass filter got %d matches\n", num_filtered);
//
//
//          if (/*num_filtered < 0.05*filtered_match_points_a_pre.size() *//*matches.size()*/ /*||*/ num_filtered < 12) {
//            //printf("Not enough matches %d for section %d with thresh %f\n", num_filtered, section_a, thresh_for_trial);
//            printf("Not enough matches %d for section %d with thresh\n", num_filtered, section_a);
//            continue;
//          } else {
//            //printf("Got enough matches %d for section %d with thresh %f\n", num_filtered, section_a, thresh_for_trial);
//            printf("Got enough matches %d for section %d with thresh\n", num_filtered, section_a);
//            //done = true;
//          }
//          //cv::Mat warp_mat = cv::estimateRigidTransform(filtered_match_points_a, filtered_match_points_b, false);//cv::findHomography(match_points_a, match_points_b);
//
//
//          cv::Mat warp_mat;
//          cv::computeAffineTFK(filtered_match_points_a, filtered_match_points_b, warp_mat);
////cv::    estimateRigidTransform(filtered_match_points_a, filtered_match_points_b, true);
//
//
//
//          std::cout << warp_mat << std::endl; 
//          vdata tmp;
//          tmp.a00 = warp_mat.at<double>(0, 0); 
//          tmp.a01 = warp_mat.at<double>(0, 1);
//          tmp.offset_x = warp_mat.at<double>(0, 2);
//          tmp.a10 = warp_mat.at<double>(1, 0); 
//          tmp.a11 = warp_mat.at<double>(1, 1); 
//          tmp.offset_y = warp_mat.at<double>(1, 2);
//          tmp.start_x = 0.0;
//          tmp.start_y = 0.0;
//          printf("Best values are %f %f %f %f %f %f\n", tmp.a00, tmp.a01, tmp.a10, tmp.a11, tmp.offset_x, tmp.offset_y);
//          vdata best_vertex_data = tmp;
//          //RETURNTFK
//
//          //int min_x = 0.0;
//          //int min_y = 0.0;
//          //bool first_check = true;
//          //for (int v = 0; v < merged_graph->num_vertices(); v++) {
//          //  double vx = merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
//          //  double vy = merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
//          //  //if (vx < box_min_x || vx > box_max_x || vy < box_min_y || vy > box_max_y) continue;
//          //  if (merged_graph->getVertexData(v)->z <= section_a) {
//          //    if (first_check) {
//          //      first_check = false;
//          //      min_x = merged_graph->getVertexData(v)->start_x;
//          //      min_y = merged_graph->getVertexData(v)->start_y;
//          //    } else {
//          //      double lx = merged_graph->getVertexData(v)->start_x;
//          //      double ly = merged_graph->getVertexData(v)->start_y;
//          //      if (lx < min_x) min_x = lx;
//          //      if (ly < min_y) min_y = ly;
//          //    }
//          //  }
//          //}
//
//          for (int v = 0; v < merged_graph->num_vertices(); v++) {
//            cv::Point2d center = merged_graph->getVertexData(v)->center_point;//transform_point_double(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
//            double vx = 1.0*((double) center.x);//merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
//            double vy = 1.0*((double) center.y);//merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
//            //if (vx < box_min_x-6000.0 || vx > box_max_x+6000.0 || vy < box_min_y -6000.0 || vy > box_max_y+6000.0) continue;
//            if (vx < box_min_x || vx > box_max_x || vy < box_min_y || vy > box_max_y) continue;
//            if (merged_graph->getVertexData(v)->z <= section_a || merged_graph->getVertexData(v)->z == section_a - 1 || merged_graph->getVertexData(v)->z == section_a-2) {
//
//              cv::Mat A(3, 3, cv::DataType<double>::type);
//              A.at<double>(0,0) = 1.0;
//              A.at<double>(0,1) = 0.0;
//              A.at<double>(0,2) = 0.0;
//              A.at<double>(1,0) = 0.0;
//              A.at<double>(1,1) = 1.0;
//              A.at<double>(1,2) = 0.0;
//              A.at<double>(2,0) = 0.0;
//              A.at<double>(2,1) = 0.0;
//              A.at<double>(2,2) = 1.0;
//              cv::Point2f box_midpoint = cv::Point2f(box_min_x + (box_max_x-box_min_x)/2, box_min_y + (box_max_y-box_min_y)/2);
//              double distance = std::sqrt((box_midpoint.x-vx)*(box_midpoint.x-vx) + (box_midpoint.y - vy)*(box_midpoint.y - vy));
//              double weight = 8000.0 / ((std::pow((8000.0/10),1.5) + 8000.0));//4000/(distance+4000);
//              cv::Mat B(3,3, cv::DataType<double>::type);
//              B.at<double>(0,0) = warp_mat.at<double>(0,0);
//              B.at<double>(0,1) = warp_mat.at<double>(0,1);
//              B.at<double>(0,2) = warp_mat.at<double>(0,2);
//              B.at<double>(1,0) = warp_mat.at<double>(1,0);
//              B.at<double>(1,1) = warp_mat.at<double>(1,1);
//              B.at<double>(1,2) = warp_mat.at<double>(1,2);
//              B.at<double>(2,0) = 0.0;
//              B.at<double>(2,1) = 0.0;
//              B.at<double>(2,2) = 1.0;
//              cv::Mat C = weight*B + (1.0-weight)*A;
//              updateAffineTransform(merged_graph->getVertexData(v), /*&best_vertex_data*/C);
//              continue;
//            }
//          }
//      }
//
//      }}
//      //}
//    //}
//
//    }
//}
//
//
//
//
//
//
//
//


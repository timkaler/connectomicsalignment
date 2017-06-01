
void fine_alignment_3d(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data){
    printf("STARTING THE FINE ALIGNMENT IT SO FINE (I hope)\n");

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

    for (int v = 0; v < merged_graph->num_vertices(); v++) {
      merged_graph->getVertexData(v)->center_point =
          transform_point(merged_graph->getVertexData(v),
                          merged_graph->getVertexData(v)->original_center_point);
      merged_graph->getVertexData(v)->boundary = false;
    }

    std::vector<cv::Mat> section_transforms(p_align_data->n_sections);
    cilk_for (int section = 1; section < p_align_data->n_sections; section++) {
      std::map<int, int> tile_id_to_match_count;
      std::vector< cv::Point2f > filtered_match_points_a(0);
      std::vector< cv::Point2f > filtered_match_points_b(0);
      int section_a = section-1;
      int section_b = section;

      for (double box_iter_x = min_x; box_iter_x < max_x+29000; box_iter_x += 24000) {
      for (double box_iter_y = min_y; box_iter_y < max_y+29000; box_iter_y += 24000) {
        // Filter the matches with RANSAC
        int num_filtered = 0;
        std::vector<cv::Point2f> match_points_a, match_points_b;
        double box_min_x = box_iter_x;
        double box_max_x = box_iter_x+24000;
        double box_min_y = box_iter_y;
        double box_max_y = box_iter_y+24000;

        std::set<int> mfov_ids_a;
        std::set<int> mfov_ids_b;

        for (int v = 0; v < merged_graph->num_vertices(); v++) {
          cv::Point2f center = merged_graph->getVertexData(v)->center_point;
          double vx = 1.0*((double) center.x);
          double vy = 1.0*((double) center.y);
          if (vx < box_min_x-6000.0 || vx > box_max_x+6000.0 || vy < box_min_y -6000.0 || vy > box_max_y+6000.0) continue;
          if (merged_graph->getVertexData(v)->z == section_a) {

            if (vx < box_min_x+6000 || vx > box_max_x-6000.0 || vy < box_min_y +6000.0 || vy > box_max_y-6000.0) {
             merged_graph->getVertexData(v)->boundary = true;
           }
            mfov_ids_a.insert(merged_graph->getVertexData(v)->mfov_id);
          } else if (merged_graph->getVertexData(v)->z == section_b) {
            mfov_ids_b.insert(merged_graph->getVertexData(v)->mfov_id);
          }
        }

        std::vector <cv::KeyPoint > atile_kps_in_overlap;
        std::vector <cv::Mat > atile_kps_desc_in_overlap_list;
        std::vector<int> atile_kps_tile_list;
        std::vector <cv::KeyPoint > btile_kps_in_overlap;
        std::vector <cv::Mat > btile_kps_desc_in_overlap_list;
        std::vector<int> btile_kps_tile_list;

        std::vector<double> atile_weights;
        std::vector<double> btile_weights;


        std::set<int> tile_id_set;
        tile_id_set.clear();
        for (int v = 0; v < merged_graph->num_vertices(); v++) {
          if (merged_graph->edgeData[v].size() == 0) continue;
          cv::Point2f center = merged_graph->getVertexData(v)->center_point;
          double vx = 1.0*((double) center.x);
          double vy = 1.0*((double) center.y);
          if (vx < box_min_x - 6000.0 || vx > box_max_x + 6000.0 ||
              vy < box_min_y - 6000.0 || vy > box_max_y + 6000.0) continue;

          if (merged_graph->getVertexData(v)->z == section_a) {
            tile_id_set.insert(v);
            int curr_z = merged_graph->getVertexData(v)->z;
            _tile_data tdata_a = p_align_data->sec_data[curr_z].tiles[merged_graph->getVertexData(v)->tile_id];
            concat_two_tiles_all_filter(merged_graph->getVertexData(v), &tdata_a, v, atile_kps_in_overlap, atile_kps_desc_in_overlap_list, atile_kps_tile_list, box_min_x, box_min_y, box_max_x, box_max_y);
          } else if (merged_graph->getVertexData(v)->z == section_b) {
            _tile_data tdata_b = p_align_data->sec_data[section_b].tiles[merged_graph->getVertexData(v)->tile_id];
            concat_two_tiles_all_filter(merged_graph->getVertexData(v),&tdata_b, v, btile_kps_in_overlap, btile_kps_desc_in_overlap_list, btile_kps_tile_list, box_min_x, box_min_y, box_max_x, box_max_y);
          }
        }
        //printf("Total size of a tile kps is %lu\n", atile_kps_in_overlap.size());
        //printf("Total size of b tile kps is %lu\n", btile_kps_in_overlap.size());
        if (atile_kps_tile_list.size() == 0 || btile_kps_tile_list.size() == 0) continue;

        cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
        cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
        cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));

        std::vector< cv::DMatch > matches;
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       0.65);
        if (matches.size() == 0) continue;
        //printf("Done with the matching. Num matches is %lu\n", matches.size());

        for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
          match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
          match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
        }

        bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
        tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 25.0, mask);
        for (int c = 0; c < match_points_a.size(); c++) {
          if (mask[c]) {
            num_filtered++;
          }
        }
        //printf("Second pass filter got %d matches\n", num_filtered);

        if (num_filtered < 12) {
          //printf("Not enough matches %d for section %d with thresh\n", num_filtered, section_a);
          free(mask);
          continue;
        } else {
            //printf("Got enough matches %d for section %d with thresh\n", num_filtered, section_a);
        }

        for (int c = 0; c < match_points_a.size(); c++) {
          if (mask[c]) {
            int atile_id = atile_kps_tile_list[matches[c].queryIdx];
            tile_id_to_match_count[atile_id] += 1;
            filtered_match_points_a.push_back(
                match_points_a[c]);
            filtered_match_points_b.push_back(
                match_points_b[c]);
          }
        }
        free(mask);
      }
    } // END BOX DOUBLE LOOP

    cv::computeAffineTFK(filtered_match_points_a, filtered_match_points_b, section_transforms[section]);
  }

  // Apply the section transformations that we computed in parallel.
  for (int section = 1; section < p_align_data->n_sections; section++) {
    int section_a = section-1;
    std::set<int> sections_done;
    sections_done.clear();
    for (int v = 0; v < merged_graph->num_vertices(); v++) {
      if (merged_graph->getVertexData(v)->z <= section_a) {
        updateAffineTransform(merged_graph->getVertexData(v), section_transforms[section]);
        if (sections_done.find(merged_graph->getVertexData(v)->z) == sections_done.end()) {
          sections_done.insert(merged_graph->getVertexData(v)->z);
          updateAffineSectionTransform(merged_graph->getVertexData(v), section_transforms[section]);
        }
        continue;
      }
    }
  }
}


void fine_alignment_3d_2(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data, double ransac_thresh, std::vector<tfkMatch>& mesh_matches){
    printf("STARTING THE FINE ALIGNMENT IT SO FINE (I hope)\n");

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

    for (int v = 0; v < merged_graph->num_vertices(); v++) {
      merged_graph->getVertexData(v)->center_point = transform_point(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
      merged_graph->getVertexData(v)->boundary = false;
    }

    std::vector<std::vector<tfkMatch> > section_mesh_matches(p_align_data->n_sections);

    cilk_for (int section = 1; section < p_align_data->n_sections; section++) {
      std::vector< cv::Point2f > filtered_match_points_a(0);
      std::vector< cv::Point2f > filtered_match_points_b(0);
      int section_a = section-1;
      int section_b = section;

      for (double box_iter_x = min_x; box_iter_x < max_x+15000; box_iter_x += 24000) {
      for (double box_iter_y = min_y; box_iter_y < max_y+15000; box_iter_y += 24000) {
        // Filter the matches with RANSAC
        int num_filtered = 0;
        std::vector<cv::Point2f> match_points_a, match_points_b;
        double box_min_x = box_iter_x;
        double box_max_x = box_iter_x+24000;
        double box_min_y = box_iter_y;
        double box_max_y = box_iter_y+24000;

        std::set<int> mfov_ids_a;
        std::set<int> mfov_ids_b;
        for (int v = 0; v < merged_graph->num_vertices(); v++) {
          cv::Point2f center = merged_graph->getVertexData(v)->center_point;
          double vx = 1.0*((double) center.x);
          double vy = 1.0*((double) center.y);
          if (vx < box_min_x-6000.0 || vx > box_max_x+6000.0 || vy < box_min_y -6000.0 || vy > box_max_y+6000.0) continue;
          if (merged_graph->getVertexData(v)->z == section_a) {

            if (vx < box_min_x+6000 || vx > box_max_x-6000.0 || vy < box_min_y +6000.0 || vy > box_max_y-6000.0) {
             merged_graph->getVertexData(v)->boundary = true;
           }

            mfov_ids_a.insert(merged_graph->getVertexData(v)->mfov_id);
          } else if (merged_graph->getVertexData(v)->z == section_b) {
            mfov_ids_b.insert(merged_graph->getVertexData(v)->mfov_id);
          }
        }

        std::vector <cv::KeyPoint > atile_kps_in_overlap;
        std::vector <cv::Mat > atile_kps_desc_in_overlap_list;
        std::vector<int> atile_kps_tile_list;
        std::vector <cv::KeyPoint > btile_kps_in_overlap;
        std::vector <cv::Mat > btile_kps_desc_in_overlap_list;
        std::vector<int> btile_kps_tile_list;

        std::vector<double> atile_weights;
        std::vector<double> btile_weights;


        std::set<int> tile_id_set;
        tile_id_set.clear();
        for (int v = 0; v < merged_graph->num_vertices(); v++) {
          if (merged_graph->edgeData[v].size() == 0) continue;
            cv::Point2f center = merged_graph->getVertexData(v)->center_point;
            double vx = 1.0*((double) center.x);
            double vy = 1.0*((double) center.y);
          if (vx < box_min_x-6000.0 || vx > box_max_x+6000.0 || vy < box_min_y -6000.0 || vy > box_max_y+6000.0) continue;

          if (merged_graph->getVertexData(v)->z == section_a) {
            tile_id_set.insert(v);
            int curr_z = merged_graph->getVertexData(v)->z;
            _tile_data tdata_a = p_align_data->sec_data[curr_z].tiles[merged_graph->getVertexData(v)->tile_id];
            concat_two_tiles_all_filter(merged_graph->getVertexData(v), &tdata_a, v,
                                        atile_kps_in_overlap, atile_kps_desc_in_overlap_list,
                                        atile_kps_tile_list, box_min_x, box_min_y, box_max_x,
                                        box_max_y);
          } else if (merged_graph->getVertexData(v)->z == section_b) {
            _tile_data tdata_b = p_align_data->sec_data[section_b].tiles[merged_graph->getVertexData(v)->tile_id];
            concat_two_tiles_all_filter(merged_graph->getVertexData(v),&tdata_b, v,
                                        btile_kps_in_overlap, btile_kps_desc_in_overlap_list,
                                        btile_kps_tile_list, box_min_x, box_min_y, box_max_x,
                                        box_max_y);
          }
        }



        //printf("Total size of a tile kps is %lu\n", atile_kps_in_overlap.size());
        //printf("Total size of b tile kps is %lu\n", btile_kps_in_overlap.size());
        if (atile_kps_tile_list.size() == 0 || btile_kps_tile_list.size() == 0) continue;

        cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
        cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
        cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));

        std::vector< cv::DMatch > matches;
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       0.92);
        if (matches.size() == 0) continue;



        //printf("Done with the matching. Num matches is %lu\n", matches.size());


        for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
          match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
          match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
        }

        bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
        tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, ransac_thresh, mask);
        for (int c = 0; c < match_points_a.size(); c++) {
          if (mask[c]) {
            num_filtered++;
          }
        }
        //printf("Second pass filter got %d matches\n", num_filtered);

        if (num_filtered < 120) {
          //printf("Not enough matches %d for section %d with thresh\n", num_filtered, section_a);
          free(mask);
          continue;
        } else {
          //printf("Got enough matches %d for section %d with thresh\n", num_filtered, section_a);
        }

        for (int c = 0; c < match_points_a.size(); c++) {
          if (mask[c]) {
            filtered_match_points_a.push_back(
                match_points_a[c]);
            filtered_match_points_b.push_back(
                match_points_b[c]);
          }
        }
        free(mask);
      }
    }

    graph_section_data* section_data_a;
    graph_section_data* section_data_b;

      for (int v = 0; v < merged_graph->num_vertices(); v++) {
        if (merged_graph->getVertexData(v)->z == section_a) {
          section_data_a = merged_graph->getVertexData(v)->section_data;
          break;
        }
      }
      for (int v = 0; v < merged_graph->num_vertices(); v++) {
        if (merged_graph->getVertexData(v)->z == section_b) {
          section_data_b = merged_graph->getVertexData(v)->section_data;
          break;
        }
      }


      for (int m = 0; m < filtered_match_points_a.size(); m++) {
        cv::Point2f my_pt = filtered_match_points_a[m];
        cv::Point2f n_pt = filtered_match_points_b[m];

        tfkMatch match;
        // find the triangle...
        std::vector<tfkTriangle>* triangles = section_data_a->triangles;
        std::vector<cv::Point2f>* mesh = section_data_a->mesh;

        std::vector<tfkTriangle>* n_triangles = section_data_b->triangles;
        std::vector<cv::Point2f>* n_mesh = section_data_b->mesh;


        int my_triangle_index = -1;
        int n_triangle_index = -1;
        for (int s = 0; s < triangles->size(); s++) {
          float u,v,w;
          cv::Point2f pt1 = (*mesh)[(*triangles)[s].index1];
          cv::Point2f pt2 = (*mesh)[(*triangles)[s].index2];
          cv::Point2f pt3 = (*mesh)[(*triangles)[s].index3];
          Barycentric(my_pt, pt1,pt2,pt3,u,v,w);
          if (u <= 0 || v <= 0 || w <= 0) continue;
          my_triangle_index = s;
          match.my_tri = (*triangles)[my_triangle_index];
          match.my_barys[0] = (double)1.0*u;
          match.my_barys[1] = (double)1.0*v;
          match.my_barys[2] = (double)1.0*w;
          break;
        }

        for (int s = 0; s < n_triangles->size(); s++) {
          float u,v,w;
          cv::Point2f pt1 = (*n_mesh)[(*n_triangles)[s].index1];
          cv::Point2f pt2 = (*n_mesh)[(*n_triangles)[s].index2];
          cv::Point2f pt3 = (*n_mesh)[(*n_triangles)[s].index3];
          Barycentric(n_pt, pt1,pt2,pt3,u,v,w);
          if (u <= 0 || v <= 0 || w <= 0) continue;
          n_triangle_index = s; 
          match.n_tri = (*n_triangles)[n_triangle_index];
          match.n_barys[0] = (double)1.0*u;
          match.n_barys[1] = (double)1.0*v;
          match.n_barys[2] = (double)1.0*w;
          break;
        }
        if (my_triangle_index == -1 || n_triangle_index == -1) continue;
        match.my_section_data = *section_data_a; 
        match.n_section_data = *section_data_b;
        section_mesh_matches[section].push_back(match); 
      }
    }

    for (int section = 1; section < p_align_data->n_sections; section++) {
      for (int m = 0; m < section_mesh_matches[section].size(); m++) {
        mesh_matches.push_back(section_mesh_matches[section][m]);
      }
    }
}


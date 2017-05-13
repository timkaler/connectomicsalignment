
void fine_alignment_3d(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data){
    printf("STARTING THE FINE ALIGNMENT IT SO FINE (I hope)");
    if (true) {
    //int vertex_id_offset = 0;

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




      for (double box_iter_x = min_x; box_iter_x < max_x+29000; box_iter_x += 24000) {
      for (double box_iter_y = min_y; box_iter_y < max_y+29000; box_iter_y += 24000) {

    for (int section = 1; section < p_align_data->n_sections; section++) {
        // Filter the matches with RANSAC
        int section_a = section-1;
        int section_b = section;
                int num_filtered = 0;
                std::vector< cv::Point2f > filtered_match_points_a(0);
                std::vector< cv::Point2f > filtered_match_points_b(0);
        std::vector<cv::Point2f> match_points_a, match_points_b;
        double box_min_x = box_iter_x;//min_x + (max_x-min_x)*box_iter_x*1.0/20;
        double box_max_x = box_iter_x+24000;//min_x + (max_x-min_x)*(box_iter_x+1)*1.0/20;
        double box_min_y = box_iter_y;//min_y + (max_y-min_y)*box_iter_y*1.0/20;
        double box_max_y = box_iter_y+24000;//min_y + (max_y-min_y)*(box_iter_y+1)*1.0/20;

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
            cv::Point2f center = merged_graph->getVertexData(v)->center_point;//transform_point(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
            double vx = 1.0*((double) center.x);//merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
            double vy = 1.0*((double) center.y);//merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
          //double vx = merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
          //double vy = merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
          //if (vx < box_min_x || vx > box_max_x || vy < box_min_y || vy > box_max_y) continue;
          if (vx < box_min_x-6000.0 || vx > box_max_x+6000.0 || vy < box_min_y -6000.0 || vy > box_max_y+6000.0) continue;
          //if (merged_graph->getVertexData(v)->z == section_a &&
          //    mfov_ids_a.find(merged_graph->getVertexData(v)->mfov_id) == mfov_ids_a.end()) continue; 
          //if (merged_graph->getVertexData(v)->z == section_b &&
          //    mfov_ids_b.find(merged_graph->getVertexData(v)->mfov_id) == mfov_ids_b.end()) continue; 

          if (merged_graph->getVertexData(v)->z == section_a /*|| merged_graph->getVertexData(v)->z == section_a-1*/) {
            tile_id_set.insert(v);
            int curr_z = merged_graph->getVertexData(v)->z;
            _tile_data tdata_a = p_align_data->sec_data[curr_z].tiles[merged_graph->getVertexData(v)->tile_id];
            //printf("The center is at %f, %f\n", center.x, center.y);
            //printf("Original center was at %f, %f\n", merged_graph->getVertexData(v)->original_center_point.x, merged_graph->getVertexData(v)->original_center_point.y);
            //printf("Best values are %f %f %f %f %f %f\n", merged_graph->getVertexData(v)->a00, merged_graph->getVertexData(v)->a01, merged_graph->getVertexData(v)->a10, merged_graph->getVertexData(v)->a11, merged_graph->getVertexData(v)->offset_x, merged_graph->getVertexData(v)->offset_y);
            concat_two_tiles_all_filter(merged_graph->getVertexData(v), &tdata_a, v, atile_kps_in_overlap, atile_kps_desc_in_overlap_list, atile_kps_tile_list, box_min_x, box_min_y, box_max_x, box_max_y);
          } else if (merged_graph->getVertexData(v)->z == section_b) {
            _tile_data tdata_b = p_align_data->sec_data[section_b].tiles[merged_graph->getVertexData(v)->tile_id];
            concat_two_tiles_all_filter(merged_graph->getVertexData(v),&tdata_b, v, btile_kps_in_overlap, btile_kps_desc_in_overlap_list, btile_kps_tile_list, box_min_x, box_min_y, box_max_x, box_max_y);
          }
        }



        printf("Total size of a tile kps is %lu\n", atile_kps_in_overlap.size());
        printf("Total size of b tile kps is %lu\n", btile_kps_in_overlap.size());
        if (atile_kps_tile_list.size() == 0 || btile_kps_tile_list.size() == 0) continue;



        cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
        cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
        cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));
        //bool done = false;

        //for (int trial_thresh = 0; trial_thresh < 4; trial_thresh++) {
        //  if (done) break;
        //for (int trial = 0; trial < 6; trial++) {
        //  if (done) break;
        //  float trial_rod = 0.55;
        //  if (trial == 1) trial_rod = 0.6;
        //  if (trial == 2) trial_rod = 0.7;
        //  if (trial == 3) trial_rod = 0.8;
        //  if (trial == 4) trial_rod = 0.92;
        //  if (trial == 5) trial_rod = 0.96;

        //  double thresh_for_trial = 4.0;
        //  if (trial_thresh == 1) thresh_for_trial = 5.0;
        //  if (trial_thresh == 2) thresh_for_trial = 6.0;
        //  if (trial_thresh == 3) thresh_for_trial = 7.0;

          std::vector< cv::DMatch > matches;
          match_features(matches,
                         atile_kps_desc_in_overlap,
                         btile_kps_desc_in_overlap,
                         0.65);
          if (matches.size() == 0) continue;



              printf("Done with the matching. Num matches is %lu\n", matches.size());

              p_align_data->sec_data[section_a].p_kps = new std::vector<cv::KeyPoint>();
              p_align_data->sec_data[section_b].p_kps = new std::vector<cv::KeyPoint>();



              for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
                //int atile_id = atile_kps_tile_list[matches[tmpi].queryIdx];
                //int btile_id = btile_kps_tile_list[matches[tmpi].trainIdx];

                //int x_start_a = merged_graph->getVertexData(atile_id)->start_x + merged_graph->getVertexData(atile_id)->offset_x;
                //int y_start_a = merged_graph->getVertexData(atile_id)->start_y + merged_graph->getVertexData(atile_id)->offset_y;

                //int x_start_b = merged_graph->getVertexData(btile_id)->start_x + merged_graph->getVertexData(btile_id)->offset_x;
                //int y_start_b = merged_graph->getVertexData(btile_id)->start_y + merged_graph->getVertexData(btile_id)->offset_y;

                match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
                match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);

                //p_align_data->sec_data[section_a].p_kps->push_back(atile_kps_in_overlap[matches[tmpi].queryIdx]);
                //p_align_data->sec_data[section_b].p_kps->push_back(btile_kps_in_overlap[matches[tmpi].trainIdx]);

                //match_points_a.push_back(
                //    atile_kps_in_overlap[matches[tmpi].queryIdx].pt + cv::Point2f(x_start_a, y_start_a));
                //match_points_b.push_back(
                //    btile_kps_in_overlap[matches[tmpi].trainIdx].pt + cv::Point2f(x_start_b, y_start_b));
              }

             //   int left_box_count = 0; 
             //   int right_box_count = 0; 
             //   for (int v2 = 0; v2 < merged_graph->num_vertices(); v2++) {
             //     cv::Point2f center = merged_graph->getVertexData(v2)->center_point;//transform_point(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
             //     double vx = 1.0*((double) center.x);//merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
             //     double vy = 1.0*((double) center.y);//merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
             //     bool compute = false;
             //     // left box.
             //     if (vx <= box_max_x-24000.0 && vx >= box_min_x-24000.0 && vy >= box_min_y && vy <= box_max_y) {
             //       left_box_count++;
             //       compute = true;

             //     } 
             //     if (vy <= box_max_y-24000.0 && vy >= box_min_y-24000.0 && vx >= box_min_x && vx <= box_max_x) {
             //       right_box_count++;
             //       compute = true;
             //     }
             //     if (compute) {
             //       for (int edge = 0; edge < merged_graph->edgeData[v2].size(); edge++) {
             //         edata edge_data = merged_graph->edgeData[v2][edge];
             //         if (tile_id_set.find(edge_data.neighbor_id) == tile_id_set.end()) {
             //           continue;
             //         }
             //         for (int iter_n = 0; iter_n < edge_data.v_points->size(); iter_n++) {
             //           match_points_a.push_back(transform_point(merged_graph->getVertexData(edge_data.neighbor_id), (*edge_data.n_points)[iter_n]));
             //           match_points_b.push_back(transform_point(merged_graph->getVertexData(v2), (*edge_data.v_points)[iter_n]));
             //         }
             //       }
             //     }
             //   }
             //   printf("I saw left matches %d and right matches %d\n", left_box_count, right_box_count);


                bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
                tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 25.0, mask);

                for (int c = 0; c < match_points_a.size(); c++) {
                  if (mask[c]) {
                    num_filtered++;
                    filtered_match_points_a.push_back(
                        match_points_a[c]);
                    filtered_match_points_b.push_back(
                        match_points_b[c]);
                  }
                }
                //int num_fake_matches = 10;

                //

                //if (num_fake_matches < match_points_a.size()/10) {
                //  num_fake_matches = match_points_a.size()/10;
                //}
                //for (int fake = 0; fake < num_fake_matches; fake++) {
                //  filtered_match_points_a.push_back(cv::Point2f(box_iter_x, box_iter_y + 24000.0/20));
                //  filtered_match_points_a.push_back(cv::Point2f(box_iter_x+24000.0/20, box_iter_y));
                //}

                free(mask);

              //}
              //}

              //bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
              //std::pair<double,double> offset_pair;
              //tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 1024.0, mask);
              //std::vector< cv::Point2f > filtered_match_points_a_pre(0);
              //std::vector< cv::Point2f > filtered_match_points_b_pre(0);

          //int num_filtered = 0;
          //for (int c = 0; c < match_points_a.size(); c++) {
          //  if (mask[c]|| true) {
          //    num_filtered++;
          //    //int atile_id = atile_kps_tile_list[matches[c].queryIdx];
          //    //int btile_id = btile_kps_tile_list[matches[c].trainIdx];
          //    filtered_match_points_a_pre.push_back(
          //        match_points_a[c]);
          //    filtered_match_points_b_pre.push_back(
          //        match_points_b[c]);
          //  }
          //}
          //free(mask); 
          //mask = (bool*)calloc(match_points_a.size()+1, 1);
          //printf("First pass filter got %d matches\n", num_filtered);
          //vdata best_vertex_data  = tfk_simple_ransac_strict_ret_affine(filtered_match_points_a_pre, filtered_match_points_b_pre, 10.0, mask);

          //num_filtered = 0;
          //for (int c = 0; c < filtered_match_points_a_pre.size(); c++) {
          //  if (mask[c]) {
          //    num_filtered++;
          //    filtered_match_points_a.push_back(
          //        filtered_match_points_a_pre[c]);
          //    filtered_match_points_b.push_back(
          //        filtered_match_points_b_pre[c]);
          //  }
          //}
          printf("Second pass filter got %d matches\n", num_filtered);


          if (/*num_filtered < 0.05*filtered_match_points_a_pre.size() *//*matches.size()*/ /*||*/ num_filtered < 12) {
            //printf("Not enough matches %d for section %d with thresh %f\n", num_filtered, section_a, thresh_for_trial);
            printf("Not enough matches %d for section %d with thresh\n", num_filtered, section_a);
            continue;
          } else {
            //printf("Got enough matches %d for section %d with thresh %f\n", num_filtered, section_a, thresh_for_trial);
            printf("Got enough matches %d for section %d with thresh\n", num_filtered, section_a);
            //done = true;
          }
          //cv::Mat warp_mat = cv::estimateRigidTransform(filtered_match_points_a, filtered_match_points_b, false);//cv::findHomography(match_points_a, match_points_b);


          cv::Mat warp_mat;
          cv::computeAffineTFK(filtered_match_points_a, filtered_match_points_b, warp_mat);
//cv::    estimateRigidTransform(filtered_match_points_a, filtered_match_points_b, true);



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
          //RETURNTFK

          //int min_x = 0.0;
          //int min_y = 0.0;
          //bool first_check = true;
          //for (int v = 0; v < merged_graph->num_vertices(); v++) {
          //  double vx = merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
          //  double vy = merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
          //  //if (vx < box_min_x || vx > box_max_x || vy < box_min_y || vy > box_max_y) continue;
          //  if (merged_graph->getVertexData(v)->z <= section_a) {
          //    if (first_check) {
          //      first_check = false;
          //      min_x = merged_graph->getVertexData(v)->start_x;
          //      min_y = merged_graph->getVertexData(v)->start_y;
          //    } else {
          //      double lx = merged_graph->getVertexData(v)->start_x;
          //      double ly = merged_graph->getVertexData(v)->start_y;
          //      if (lx < min_x) min_x = lx;
          //      if (ly < min_y) min_y = ly;
          //    }
          //  }
          //}

        std::map<int,std::set<int> > mfov_ids_general;
        for (int v = 0; v < merged_graph->num_vertices(); v++) {
          cv::Point2f center = merged_graph->getVertexData(v)->center_point;
          double vx = 1.0*((double) center.x);
          double vy = 1.0*((double) center.y);
          if (vx < box_min_x || vx > box_max_x || vy < box_min_y || vy > box_max_y) continue;
          int mfov_id = merged_graph->getVertexData(v)->mfov_id;
          int z = merged_graph->getVertexData(v)->z;
          mfov_ids_general[z].insert(mfov_id);
        }

          for (int v = 0; v < merged_graph->num_vertices(); v++) {
            cv::Point2d center = merged_graph->getVertexData(v)->center_point;//transform_point_double(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
            double vx = 1.0*((double) center.x);//merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
            double vy = 1.0*((double) center.y);//merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
            //if (vx < box_min_x-6000.0 || vx > box_max_x+6000.0 || vy < box_min_y -6000.0 || vy > box_max_y+6000.0) continue;
            if (vx < box_min_x || vx > box_max_x || vy < box_min_y || vy > box_max_y) continue;
            
          //int mfov_id = merged_graph->getVertexData(v)->mfov_id;
          //int z = merged_graph->getVertexData(v)->z;
          //if (mfov_ids_general[z].find(mfov_id) == mfov_ids_general[z].end()) continue;

            //if (merged_graph->getVertexData(v)->z <= section_a || merged_graph->getVertexData(v)->z == section_a - 1 || merged_graph->getVertexData(v)->z == section_a-2) {
            if (merged_graph->getVertexData(v)->z <= section_a/*merged_graph->getVertexData(v)->z <= section_a || merged_graph->getVertexData(v)->z == section_a - 1 || merged_graph->getVertexData(v)->z == section_a-2*/) {

              //cv::Mat A(3, 3, cv::DataType<double>::type);
              //A.at<double>(0,0) = 1.0;
              //A.at<double>(0,1) = 0.0;
              //A.at<double>(0,2) = 0.0;
              //A.at<double>(1,0) = 0.0;
              //A.at<double>(1,1) = 1.0;
              //A.at<double>(1,2) = 0.0;
              //A.at<double>(2,0) = 0.0;
              //A.at<double>(2,1) = 0.0;
              //A.at<double>(2,2) = 1.0;
              //cv::Point2f box_midpoint = cv::Point2f(box_min_x + (box_max_x-box_min_x)/2, box_min_y + (box_max_y-box_min_y)/2);
              //double distance = std::sqrt((box_midpoint.x-vx)*(box_midpoint.x-vx) + (box_midpoint.y - vy)*(box_midpoint.y - vy));
              //double weight = 4000/(distance+4000);
              //cv::Mat B(3,3, cv::DataType<double>::type);
              //B.at<double>(0,0) = warp_mat.at<double>(0,0);
              //B.at<double>(0,1) = warp_mat.at<double>(0,1);
              //B.at<double>(0,2) = warp_mat.at<double>(0,2);
              //B.at<double>(1,0) = warp_mat.at<double>(1,0);
              //B.at<double>(1,1) = warp_mat.at<double>(1,1);
              //B.at<double>(1,2) = warp_mat.at<double>(1,2);
              //B.at<double>(2,0) = 0.0;
              //B.at<double>(2,1) = 0.0;
              //B.at<double>(2,2) = 1.0;
              //cv::Mat C = weight*B + (1.0-weight)*A;
              updateAffineTransform(merged_graph->getVertexData(v), /*&best_vertex_data*/warp_mat);
              continue;
            }
          }
      }

      }}
      //}
    //}

    }
}



void fine_alignment_3d_dampen(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data){
    printf("STARTING THE FINE ALIGNMENT IT SO FINE (I hope)");
    if (true) {
    //int vertex_id_offset = 0;

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
    }




      for (double box_iter_x = min_x; box_iter_x < max_x+29000; box_iter_x += 24000) {
      for (double box_iter_y = min_y; box_iter_y < max_y+29000; box_iter_y += 24000) {
    for (int section = 1; section < p_align_data->n_sections; section++) {
        // Filter the matches with RANSAC
        int section_a = section-1;
        int section_b = section;
                int num_filtered = 0;
                std::vector< cv::Point2f > filtered_match_points_a(0);
                std::vector< cv::Point2f > filtered_match_points_b(0);
        std::vector<cv::Point2f> match_points_a, match_points_b;
        double box_min_x = box_iter_x;//min_x + (max_x-min_x)*box_iter_x*1.0/20;
        double box_max_x = box_iter_x+24000;//min_x + (max_x-min_x)*(box_iter_x+1)*1.0/20;
        double box_min_y = box_iter_y;//min_y + (max_y-min_y)*box_iter_y*1.0/20;
        double box_max_y = box_iter_y+24000;//min_y + (max_y-min_y)*(box_iter_y+1)*1.0/20;
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
            cv::Point2f center = merged_graph->getVertexData(v)->center_point;//transform_point(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
            double vx = 1.0*((double) center.x);//merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
            double vy = 1.0*((double) center.y);//merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
          //double vx = merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
          //double vy = merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
          //if (vx < box_min_x || vx > box_max_x || vy < box_min_y || vy > box_max_y) continue;
          if (vx < box_min_x-6000.0 || vx > box_max_x+6000.0 || vy < box_min_y -6000.0 || vy > box_max_y+6000.0) continue;

          if (merged_graph->getVertexData(v)->z == section_a/* || merged_graph->getVertexData(v)->z == section_a-1*/) {
            tile_id_set.insert(v);
            int curr_z = merged_graph->getVertexData(v)->z;
            _tile_data tdata_a = p_align_data->sec_data[curr_z].tiles[merged_graph->getVertexData(v)->tile_id];
            //printf("The center is at %f, %f\n", center.x, center.y);
            //printf("Original center was at %f, %f\n", merged_graph->getVertexData(v)->original_center_point.x, merged_graph->getVertexData(v)->original_center_point.y);
            //printf("Best values are %f %f %f %f %f %f\n", merged_graph->getVertexData(v)->a00, merged_graph->getVertexData(v)->a01, merged_graph->getVertexData(v)->a10, merged_graph->getVertexData(v)->a11, merged_graph->getVertexData(v)->offset_x, merged_graph->getVertexData(v)->offset_y);
            concat_two_tiles_all_filter(merged_graph->getVertexData(v), &tdata_a, v, atile_kps_in_overlap, atile_kps_desc_in_overlap_list, atile_kps_tile_list, box_min_x, box_min_y, box_max_x, box_max_y);
          } else if (merged_graph->getVertexData(v)->z == section_b) {
            _tile_data tdata_b = p_align_data->sec_data[section_b].tiles[merged_graph->getVertexData(v)->tile_id];
            concat_two_tiles_all_filter(merged_graph->getVertexData(v),&tdata_b, v, btile_kps_in_overlap, btile_kps_desc_in_overlap_list, btile_kps_tile_list, box_min_x, box_min_y, box_max_x, box_max_y);
          }
        }



        printf("Total size of a tile kps is %lu\n", atile_kps_in_overlap.size());
        printf("Total size of b tile kps is %lu\n", btile_kps_in_overlap.size());
        if (atile_kps_tile_list.size() == 0 || btile_kps_tile_list.size() == 0) continue;



        cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
        cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
        cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));
        //bool done = false;

        //for (int trial_thresh = 0; trial_thresh < 4; trial_thresh++) {
        //  if (done) break;
        //for (int trial = 0; trial < 6; trial++) {
        //  if (done) break;
        //  float trial_rod = 0.55;
        //  if (trial == 1) trial_rod = 0.6;
        //  if (trial == 2) trial_rod = 0.7;
        //  if (trial == 3) trial_rod = 0.8;
        //  if (trial == 4) trial_rod = 0.92;
        //  if (trial == 5) trial_rod = 0.96;

        //  double thresh_for_trial = 4.0;
        //  if (trial_thresh == 1) thresh_for_trial = 5.0;
        //  if (trial_thresh == 2) thresh_for_trial = 6.0;
        //  if (trial_thresh == 3) thresh_for_trial = 7.0;

          std::vector< cv::DMatch > matches;
          match_features(matches,
                         atile_kps_desc_in_overlap,
                         btile_kps_desc_in_overlap,
                         0.92);
          if (matches.size() == 0) continue;



              printf("Done with the matching. Num matches is %lu\n", matches.size());

              p_align_data->sec_data[section_a].p_kps = new std::vector<cv::KeyPoint>();
              p_align_data->sec_data[section_b].p_kps = new std::vector<cv::KeyPoint>();



              for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
                //int atile_id = atile_kps_tile_list[matches[tmpi].queryIdx];
                //int btile_id = btile_kps_tile_list[matches[tmpi].trainIdx];

                //int x_start_a = merged_graph->getVertexData(atile_id)->start_x + merged_graph->getVertexData(atile_id)->offset_x;
                //int y_start_a = merged_graph->getVertexData(atile_id)->start_y + merged_graph->getVertexData(atile_id)->offset_y;

                //int x_start_b = merged_graph->getVertexData(btile_id)->start_x + merged_graph->getVertexData(btile_id)->offset_x;
                //int y_start_b = merged_graph->getVertexData(btile_id)->start_y + merged_graph->getVertexData(btile_id)->offset_y;

                match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
                match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);

                //p_align_data->sec_data[section_a].p_kps->push_back(atile_kps_in_overlap[matches[tmpi].queryIdx]);
                //p_align_data->sec_data[section_b].p_kps->push_back(btile_kps_in_overlap[matches[tmpi].trainIdx]);

                //match_points_a.push_back(
                //    atile_kps_in_overlap[matches[tmpi].queryIdx].pt + cv::Point2f(x_start_a, y_start_a));
                //match_points_b.push_back(
                //    btile_kps_in_overlap[matches[tmpi].trainIdx].pt + cv::Point2f(x_start_b, y_start_b));
              }

             //   int left_box_count = 0; 
             //   int right_box_count = 0; 
             //   for (int v2 = 0; v2 < merged_graph->num_vertices(); v2++) {
             //     cv::Point2f center = merged_graph->getVertexData(v2)->center_point;//transform_point(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
             //     double vx = 1.0*((double) center.x);//merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
             //     double vy = 1.0*((double) center.y);//merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
             //     bool compute = false;
             //     // left box.
             //     if (vx <= box_max_x-24000.0 && vx >= box_min_x-24000.0 && vy >= box_min_y && vy <= box_max_y) {
             //       left_box_count++;
             //       compute = true;

             //     } 
             //     if (vy <= box_max_y-24000.0 && vy >= box_min_y-24000.0 && vx >= box_min_x && vx <= box_max_x) {
             //       right_box_count++;
             //       compute = true;
             //     }
             //     if (compute) {
             //       for (int edge = 0; edge < merged_graph->edgeData[v2].size(); edge++) {
             //         edata edge_data = merged_graph->edgeData[v2][edge];
             //         if (tile_id_set.find(edge_data.neighbor_id) == tile_id_set.end()) {
             //           continue;
             //         }
             //         for (int iter_n = 0; iter_n < edge_data.v_points->size(); iter_n++) {
             //           match_points_a.push_back(transform_point(merged_graph->getVertexData(edge_data.neighbor_id), (*edge_data.n_points)[iter_n]));
             //           match_points_b.push_back(transform_point(merged_graph->getVertexData(v2), (*edge_data.v_points)[iter_n]));
             //         }
             //       }
             //     }
             //   }
             //   printf("I saw left matches %d and right matches %d\n", left_box_count, right_box_count);


                bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
                tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 25.0, mask);

                for (int c = 0; c < match_points_a.size(); c++) {
                  if (mask[c]) {
                    num_filtered++;
                    filtered_match_points_a.push_back(
                        match_points_a[c]);
                    filtered_match_points_b.push_back(
                        match_points_b[c]);
                  }
                }
                //int num_fake_matches = 10;

                //

                //if (num_fake_matches < match_points_a.size()/10) {
                //  num_fake_matches = match_points_a.size()/10;
                //}
                //for (int fake = 0; fake < num_fake_matches; fake++) {
                //  filtered_match_points_a.push_back(cv::Point2f(box_iter_x, box_iter_y + 24000.0/20));
                //  filtered_match_points_a.push_back(cv::Point2f(box_iter_x+24000.0/20, box_iter_y));
                //}

                free(mask);

              //}
              //}

              //bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
              //std::pair<double,double> offset_pair;
              //tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 1024.0, mask);
              //std::vector< cv::Point2f > filtered_match_points_a_pre(0);
              //std::vector< cv::Point2f > filtered_match_points_b_pre(0);

          //int num_filtered = 0;
          //for (int c = 0; c < match_points_a.size(); c++) {
          //  if (mask[c]|| true) {
          //    num_filtered++;
          //    //int atile_id = atile_kps_tile_list[matches[c].queryIdx];
          //    //int btile_id = btile_kps_tile_list[matches[c].trainIdx];
          //    filtered_match_points_a_pre.push_back(
          //        match_points_a[c]);
          //    filtered_match_points_b_pre.push_back(
          //        match_points_b[c]);
          //  }
          //}
          //free(mask); 
          //mask = (bool*)calloc(match_points_a.size()+1, 1);
          //printf("First pass filter got %d matches\n", num_filtered);
          //vdata best_vertex_data  = tfk_simple_ransac_strict_ret_affine(filtered_match_points_a_pre, filtered_match_points_b_pre, 10.0, mask);

          //num_filtered = 0;
          //for (int c = 0; c < filtered_match_points_a_pre.size(); c++) {
          //  if (mask[c]) {
          //    num_filtered++;
          //    filtered_match_points_a.push_back(
          //        filtered_match_points_a_pre[c]);
          //    filtered_match_points_b.push_back(
          //        filtered_match_points_b_pre[c]);
          //  }
          //}
          printf("Second pass filter got %d matches\n", num_filtered);


          if (/*num_filtered < 0.05*filtered_match_points_a_pre.size() *//*matches.size()*/ /*||*/ num_filtered < 12) {
            //printf("Not enough matches %d for section %d with thresh %f\n", num_filtered, section_a, thresh_for_trial);
            printf("Not enough matches %d for section %d with thresh\n", num_filtered, section_a);
            continue;
          } else {
            //printf("Got enough matches %d for section %d with thresh %f\n", num_filtered, section_a, thresh_for_trial);
            printf("Got enough matches %d for section %d with thresh\n", num_filtered, section_a);
            //done = true;
          }
          //cv::Mat warp_mat = cv::estimateRigidTransform(filtered_match_points_a, filtered_match_points_b, false);//cv::findHomography(match_points_a, match_points_b);


          cv::Mat warp_mat;
          cv::computeAffineTFK(filtered_match_points_a, filtered_match_points_b, warp_mat);
//cv::    estimateRigidTransform(filtered_match_points_a, filtered_match_points_b, true);



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
          //RETURNTFK

          //int min_x = 0.0;
          //int min_y = 0.0;
          //bool first_check = true;
          //for (int v = 0; v < merged_graph->num_vertices(); v++) {
          //  double vx = merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
          //  double vy = merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
          //  //if (vx < box_min_x || vx > box_max_x || vy < box_min_y || vy > box_max_y) continue;
          //  if (merged_graph->getVertexData(v)->z <= section_a) {
          //    if (first_check) {
          //      first_check = false;
          //      min_x = merged_graph->getVertexData(v)->start_x;
          //      min_y = merged_graph->getVertexData(v)->start_y;
          //    } else {
          //      double lx = merged_graph->getVertexData(v)->start_x;
          //      double ly = merged_graph->getVertexData(v)->start_y;
          //      if (lx < min_x) min_x = lx;
          //      if (ly < min_y) min_y = ly;
          //    }
          //  }
          //}

          for (int v = 0; v < merged_graph->num_vertices(); v++) {
            cv::Point2d center = merged_graph->getVertexData(v)->center_point;//transform_point_double(merged_graph->getVertexData(v), merged_graph->getVertexData(v)->original_center_point);
            double vx = 1.0*((double) center.x);//merged_graph->getVertexData(v)->start_x + merged_graph->getVertexData(v)->offset_x;
            double vy = 1.0*((double) center.y);//merged_graph->getVertexData(v)->start_y + merged_graph->getVertexData(v)->offset_y;
            //if (vx < box_min_x-6000.0 || vx > box_max_x+6000.0 || vy < box_min_y -6000.0 || vy > box_max_y+6000.0) continue;
            if (vx < box_min_x || vx > box_max_x || vy < box_min_y || vy > box_max_y) continue;
//            if (merged_graph->getVertexData(v)->z <= section_a || merged_graph->getVertexData(v)->z == section_a - 1 || merged_graph->getVertexData(v)->z == section_a-2) {
            if (merged_graph->getVertexData(v)->z == section_a ){

              //cv::Mat A(3, 3, cv::DataType<double>::type);
              //A.at<double>(0,0) = 1.0;
              //A.at<double>(0,1) = 0.0;
              //A.at<double>(0,2) = 0.0;
              //A.at<double>(1,0) = 0.0;
              //A.at<double>(1,1) = 1.0;
              //A.at<double>(1,2) = 0.0;
              //A.at<double>(2,0) = 0.0;
              //A.at<double>(2,1) = 0.0;
              //A.at<double>(2,2) = 1.0;
              //cv::Point2f box_midpoint = cv::Point2f(box_min_x + (box_max_x-box_min_x)/2, box_min_y + (box_max_y-box_min_y)/2);
              //double distance = std::sqrt((box_midpoint.x-vx)*(box_midpoint.x-vx) + (box_midpoint.y - vy)*(box_midpoint.y - vy));
              //double weight = 8000.0 / ((std::pow((8000.0/10),1.5) + 8000.0));//4000/(distance+4000);
              //cv::Mat B(3,3, cv::DataType<double>::type);
              //B.at<double>(0,0) = warp_mat.at<double>(0,0);
              //B.at<double>(0,1) = warp_mat.at<double>(0,1);
              //B.at<double>(0,2) = warp_mat.at<double>(0,2);
              //B.at<double>(1,0) = warp_mat.at<double>(1,0);
              //B.at<double>(1,1) = warp_mat.at<double>(1,1);
              //B.at<double>(1,2) = warp_mat.at<double>(1,2);
              //B.at<double>(2,0) = 0.0;
              //B.at<double>(2,1) = 0.0;
              //B.at<double>(2,2) = 1.0;
              //cv::Mat C = weight*B + (1.0-weight)*A;
              updateAffineTransform(merged_graph->getVertexData(v), /*&best_vertex_data*/warp_mat);
              continue;
            }
          }
      }

      }}
      //}
    //}

    }
}


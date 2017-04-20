namespace cv {

  static bool computeAffineTFK(const std::vector<cv::Point2f> &srcPoints, const std::vector<cv::Point2f> &dstPoints, Mat &transf)
  {
      // sanity check
      if ((srcPoints.size() < 3) || (srcPoints.size() != dstPoints.size()))
          return false;
  
      // container for output
      transf.create(2, 3, CV_64F);
  
      // fill the matrices
      const int n = (int)srcPoints.size(), m = 3;
      Mat A(n,m,CV_64F), xc(n,1,CV_64F), yc(n,1,CV_64F);
      for(int i=0; i<n; i++)
      {
          double x = srcPoints[i].x, y = srcPoints[i].y;
          double rowI[m] = {x, y, 1};
          Mat(1,m,CV_64F,rowI).copyTo(A.row(i));
          xc.at<double>(i,0) = dstPoints[i].x;
          yc.at<double>(i,0) = dstPoints[i].y;
      }
  
      // solve linear equations (for x and for y)
      Mat aTa, resX, resY;
      mulTransposed(A, aTa, true);
      solve(aTa, A.t()*xc, resX, DECOMP_CHOLESKY);
      solve(aTa, A.t()*yc, resY, DECOMP_CHOLESKY);
  
      // store result
      memcpy(transf.ptr<double>(0), resX.data, m*sizeof(double));
      memcpy(transf.ptr<double>(1), resY.data, m*sizeof(double));
  
      return true;
  }

}

void updateVertex2DAlign(int vid, void* scheduler_void) {
  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
  Graph<vdata, edata>* graph = reinterpret_cast<Graph<vdata,edata>*>(scheduler->graph_void);

  //printf("starting vertex %d\n", vid);
  vdata* vertex_data = graph->getVertexData(vid);
  std::vector<edata> edges = graph->edgeData[vid];

  double original_offset_x = vertex_data->offset_x;
  double original_offset_y = vertex_data->offset_y;

  std::vector<cv::Point2f> source_points(0), dest_points(0);

  if (edges.size() == 0) return;

  std::vector<cv::Point2f> original_points;
  std::vector<cv::Point2f> original_points2;
  std::vector<float> weights;

  for (int i = 0; i < edges.size(); i++) {
    std::vector<cv::Point2f>* v_points = edges[i].v_points;
    std::vector<cv::Point2f>* n_points = edges[i].n_points;
    vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
    double curr_weight = 1.0;

    if (graph->getVertexData(edges[i].neighbor_id)->z != graph->getVertexData(vid)->z) {
      printf("Danger mesh optimize code has edges across sections!\n");
    }

    for (int j = 0; j < v_points->size(); j++) {
      cv::Point2f ptx1 = transform_point(vertex_data, (*v_points)[j]);
      cv::Point2f ptx2 = transform_point(neighbor_vertex, (*n_points)[j]);
      
      source_points.push_back(ptx1);
      dest_points.push_back(ptx2);
      original_points.push_back((*v_points)[j]);
      original_points2.push_back((*n_points)[j]);
      weights.push_back(curr_weight);
    }
  }
  
  std::vector<cv::Point2f>& filtered_match_points_a = source_points;
  std::vector<cv::Point2f>& filtered_match_points_b = dest_points;

  std::vector<cv::Point2d> match_points_a_fixed(0);
  if (filtered_match_points_a.size() > 0) {
    double grad_error_x = 0.0;
    double grad_error_y = 0.0;
    double weight_sum = 1.0;
    for (int iter = 0; iter < filtered_match_points_a.size(); iter++) {
       double delta_x = filtered_match_points_b[iter].x - filtered_match_points_a[iter].x;
       double delta_y = filtered_match_points_b[iter].y - filtered_match_points_a[iter].y;
       grad_error_x += 2*(delta_x);//*weights[iter];
       grad_error_y += 2*(delta_y);//*weights[iter];
       weight_sum += 1.0;
    }
    vertex_data->offset_x += /*(1.0-2*4.0/30.0)*/grad_error_x*0.49/(weight_sum);
    vertex_data->offset_y += /*(1.0-2*4.0/30.0)*/grad_error_y*0.49/(weight_sum);
    //vertex_data->offset_x += (2*4.0/30.0)*zgrad_error_x*0.49/(zweight_sum+weight_sum);
    //vertex_data->offset_y += (2*4.0/30.0)*zgrad_error_y*0.49/(zweight_sum+weight_sum);

    if ( vertex_data->iteration_count < 20000000 && (
      std::abs(vertex_data->offset_x - original_offset_x) +
      std::abs(vertex_data->offset_y - original_offset_y) > 1e-2)) {
      scheduler->add_task(vid, updateVertex2DAlign);
      for (int i = 0; i < edges.size(); i++) {
        scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
      }
    }
  }
  vertex_data->iteration_count++;
}


void coarse_alignment_3d(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data){
    if (true) {
    int vertex_id_offset = 0;

    for (int v = 0; v < merged_graph->num_vertices(); v++) {
      merged_graph->getVertexData(v)->start_x += merged_graph->getVertexData(v)->offset_x;
      merged_graph->getVertexData(v)->end_x += merged_graph->getVertexData(v)->offset_x;
      merged_graph->getVertexData(v)->start_y += merged_graph->getVertexData(v)->offset_y;
      merged_graph->getVertexData(v)->end_y += merged_graph->getVertexData(v)->offset_y;
      merged_graph->getVertexData(v)->offset_x = 0.0;
      merged_graph->getVertexData(v)->offset_y = 0.0;
    }


    for (int section = 1; section < p_align_data->n_sections; section++) {
      int section_a = section-1;
      int section_b = section;
      std::vector <cv::KeyPoint > atile_kps_in_overlap;
      std::vector <cv::Mat > atile_kps_desc_in_overlap_list;
      std::vector<int> atile_kps_tile_list; 
      std::vector <cv::KeyPoint > btile_kps_in_overlap;
      std::vector <cv::Mat > btile_kps_desc_in_overlap_list;
      std::vector<int> btile_kps_tile_list;

      for (int v = 1; v < merged_graph->num_vertices(); v++) {
        if (merged_graph->getVertexData(v)->z == section_a || merged_graph->getVertexData(v)->z == section_a-1) {
          _tile_data tdata_a = p_align_data->sec_data[section_a].tiles[merged_graph->getVertexData(v)->tile_id];
          concat_two_tiles_all(&tdata_a, v, atile_kps_in_overlap, atile_kps_desc_in_overlap_list, atile_kps_tile_list);
        } else if (merged_graph->getVertexData(v)->z == section_b) {
          _tile_data tdata_b = p_align_data->sec_data[section_b].tiles[merged_graph->getVertexData(v)->tile_id];
          concat_two_tiles_all(&tdata_b, v, btile_kps_in_overlap, btile_kps_desc_in_overlap_list, btile_kps_tile_list);
        }
      }

      printf("Total size of a tile kps is %d\n", atile_kps_in_overlap.size());
      printf("Total size of b tile kps is %d\n", btile_kps_in_overlap.size());
 
      cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
      cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
      cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));

      std::vector< cv::DMatch > matches;
      match_features(matches,
                     atile_kps_desc_in_overlap,
                     btile_kps_desc_in_overlap,
                     0.65);
          printf("Done with the matching. Num matches is %d\n", matches.size());
          // Filter the matches with RANSAC
          std::vector<cv::Point2f> match_points_a, match_points_b;
          for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
            int atile_id = atile_kps_tile_list[matches[tmpi].queryIdx];
            int btile_id = btile_kps_tile_list[matches[tmpi].trainIdx];

            int x_start_a = merged_graph->getVertexData(atile_id)->start_x + merged_graph->getVertexData(atile_id)->offset_x;
            int y_start_a = merged_graph->getVertexData(atile_id)->start_y + merged_graph->getVertexData(atile_id)->offset_y;

            int x_start_b = merged_graph->getVertexData(btile_id)->start_x + merged_graph->getVertexData(btile_id)->offset_x;
            int y_start_b = merged_graph->getVertexData(btile_id)->start_y + merged_graph->getVertexData(btile_id)->offset_y;

            match_points_a.push_back(
                transform_point(merged_graph->getVertexData(atile_id),atile_kps_in_overlap[matches[tmpi].queryIdx].pt));
            match_points_b.push_back(
                transform_point(merged_graph->getVertexData(btile_id),btile_kps_in_overlap[matches[tmpi].trainIdx].pt));
            //match_points_a.push_back(
            //    atile_kps_in_overlap[matches[tmpi].queryIdx].pt + cv::Point2f(x_start_a, y_start_a));
            //match_points_b.push_back(
            //    btile_kps_in_overlap[matches[tmpi].trainIdx].pt + cv::Point2f(x_start_b, y_start_b));
          }
          bool* mask = (bool*)calloc(matches.size()+1, 1);
          std::pair<double,double> offset_pair;
          vdata best_vertex_data  = tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 25.0, mask); 
          std::vector< cv::Point2f > filtered_match_points_a(0);
          std::vector< cv::Point2f > filtered_match_points_b(0);
      for (int c = 0; c < matches.size(); c++) {
        if (mask[c]) {
          int atile_id = atile_kps_tile_list[matches[c].queryIdx];
          int btile_id = btile_kps_tile_list[matches[c].trainIdx];
          filtered_match_points_a.push_back(
              match_points_a[c]);
          filtered_match_points_b.push_back(
              match_points_b[c]);
        }
      }
      //cv::Mat warp_mat = cv::estimateRigidTransform(filtered_match_points_a, filtered_match_points_b, false);//cv::findHomography(match_points_a, match_points_b);


      cv::Mat warp_mat;
      cv::computeAffineTFK(filtered_match_points_a, filtered_match_points_b, warp_mat);
//cv::estimateRigidTransform(filtered_match_points_a, filtered_match_points_b, true);



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
      best_vertex_data = tmp;
      //RETURNTFK

      int min_x = 0.0;
      int min_y = 0.0;
      bool first_check = true;
      for (int v = 0; v < merged_graph->num_vertices(); v++) {
        if (merged_graph->getVertexData(v)->z <= section_a) {
          if (first_check) {
            first_check = false;
            min_x = merged_graph->getVertexData(v)->start_x;
            min_y = merged_graph->getVertexData(v)->start_y;
          } else {
            double lx = merged_graph->getVertexData(v)->start_x;
            double ly = merged_graph->getVertexData(v)->start_y;
            if (lx < min_x) min_x = lx;
            if (ly < min_y) min_y = ly;
          }
        }
      }

      for (int v = 0; v < merged_graph->num_vertices(); v++) {
        if (merged_graph->getVertexData(v)->z <= section_a) {
          updateAffineTransform(merged_graph->getVertexData(v), /*&best_vertex_data*/warp_mat);
          continue;
        }
      }

    }

    }


}





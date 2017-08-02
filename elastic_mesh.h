void elastic_mesh_optimize(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data) {


  TFK_TIMER_VAR(elastic_timer);

  TFK_START_TIMER(&elastic_timer);
  std::set<int> sections_done;
  sections_done.clear();
  // now transform the mesh points using section transforms.
  for (int v = 0; v < merged_graph->num_vertices(); v++) {
    vdata* vd = merged_graph->getVertexData(v);
    if (sections_done.find(vd->z) == sections_done.end()) {
      printf("Doing an update for section %d\n", vd->z);
      sections_done.insert(vd->z);
      graph_section_data* section_data = vd->section_data;
      std::vector<cv::Point2f>* mesh = section_data->mesh;
      std::vector<cv::Point2f>* mesh_orig = section_data->mesh_orig;
      cv::Mat warp_mat = *(section_data->transform);
      vdata tmp;
      tmp.a00 = warp_mat.at<double>(0, 0); 
      tmp.a01 = warp_mat.at<double>(0, 1);
      tmp.offset_x = warp_mat.at<double>(0, 2);
      tmp.a10 = warp_mat.at<double>(1, 0); 
      tmp.a11 = warp_mat.at<double>(1, 1); 
      tmp.offset_y = warp_mat.at<double>(1, 2);
      tmp.start_x = 0.0;
      tmp.start_y = 0.0;

      for (int mesh_index = 0; mesh_index < mesh->size(); mesh_index++) {
        (*mesh)[mesh_index] = transform_point(&tmp, (*mesh)[mesh_index]);
        (*mesh_orig)[mesh_index] = transform_point(&tmp, (*mesh_orig)[mesh_index]);
      }
    }
  }
//  return;
  // this is going to give us the matches.
  std::vector<tfkMatch> mesh_matches;
  fine_alignment_3d_2(merged_graph, p_align_data,64.0, mesh_matches);


  TFK_STOP_TIMER(&elastic_timer, "After fine_alignment_3d_2");

  printf("The number of mesh matches is %lu\n", mesh_matches.size());
  {
    double cross_slice_weight = 1.0;
    double cross_slice_winsor = 20.0;
    double intra_slice_weight = 0.9;
    double intra_slice_winsor = 200.0;
    int max_iterations = 5000; //ORIGINALL 5000
    //double min_stepsize = 1e-20;
    double stepsize = 0.0001;
    double momentum = 0.5;
    std::map<int, double> gradient_momentum;
    std::map<int, graph_section_data> section_data_map;
    section_data_map.clear();
    for (int i = 0; i < mesh_matches.size(); i++) {
      int az = mesh_matches[i].my_section_data.z;
      int bz = mesh_matches[i].n_section_data.z;
      if (section_data_map.find(az) == section_data_map.end()) {
        section_data_map[az] = mesh_matches[i].my_section_data;
      }
      if (section_data_map.find(bz) == section_data_map.end()) {
        section_data_map[bz] = mesh_matches[i].n_section_data;
      }
    }

      for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
           it != section_data_map.end(); ++it) {
        it->second.gradients = new cv::Point2f[it->second.mesh->size()];
        it->second.gradients_with_momentum = new cv::Point2f[it->second.mesh->size()];
        it->second.rest_lengths = new double[it->second.triangle_edges->size()];
        it->second.rest_areas = new double[it->second.triangles->size()];
        it->second.mesh_old = new std::vector<cv::Point2f>();

        for (int j = 0; j < it->second.mesh->size(); j++) {
          it->second.mesh_old->push_back((*(it->second.mesh))[j]);
        }

        // init
        for (int j = 0; j < it->second.mesh->size(); j++) {
          it->second.gradients[j] = cv::Point2f(0.0,0.0);  
          it->second.gradients_with_momentum[j] = cv::Point2f(0.0,0.0);  
        }

        for (int j = 0; j < it->second.triangle_edges->size(); j++) {
          cv::Point2f p1 = (*(it->second.mesh))[(*(it->second.triangle_edges))[j].first];
          cv::Point2f p2 = (*(it->second.mesh))[(*(it->second.triangle_edges))[j].second];
          double dx = p1.x-p2.x;
          double dy = p1.y-p2.y;
          double len = std::sqrt(dx*dx+dy*dy);
          it->second.rest_lengths[j] = len;
          //printf("Rest length is %f\n", len);
        }

        // now triangle areas.
        for (int j = 0; j < it->second.triangles->size(); j++) {
          tfkTriangle tri = (*(it->second.triangles))[j];
          cv::Point2f p1 = (*(it->second.mesh))[tri.index1];
          cv::Point2f p2 = (*(it->second.mesh))[tri.index2];
          cv::Point2f p3 = (*(it->second.mesh))[tri.index3];
          it->second.rest_areas[j] = computeTriangleArea(p1,p2,p3);
          //printf("Rest area is %f\n", it->second.rest_areas[j]);
        }
      }
    double prev_cost = 0.0; 
    for (int iter = 0; iter < max_iterations; iter++) {
      double cost = 0.0;

      for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
           it != section_data_map.end(); ++it) {
        // clear old gradients.
        for (int j = 0; j < it->second.mesh->size(); j++) {
          it->second.gradients[j] = cv::Point2f(0.0,0.0);  
        }
      }

      // compute internal gradients.
      for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
           it != section_data_map.end(); ++it) {
        // internal_mesh_derivs
        double all_weight = intra_slice_weight; 
        double sigma = intra_slice_winsor; 
        std::vector<cv::Point2f>* mesh = it->second.mesh;
        cv::Point2f* gradients = it->second.gradients;

        std::vector<std::pair<int, int> >* triangle_edges = it->second.triangle_edges;
        std::vector<tfkTriangle >* triangles = it->second.triangles;
        double* rest_lengths = it->second.rest_lengths;
        double* rest_areas = it->second.rest_areas;

        //// update all edges
        for (int j = 0; j < triangle_edges->size(); j++) {
          cost += internal_mesh_derivs(mesh, gradients, (*triangle_edges)[j], rest_lengths[j],
                                       all_weight, sigma);
        }

        //// update all triangles
        for (int j = 0; j < triangles->size(); j++) {
          int triangle_indices[3] = {(*triangles)[j].index1,
                                     (*triangles)[j].index2,
                                     (*triangles)[j].index3};
          cost += area_mesh_derivs(mesh, gradients, triangle_indices, rest_areas[j], all_weight);
        }
      }

      for (int j = 0; j < mesh_matches.size(); j++) {
        std::vector<cv::Point2f>* mesh1 = mesh_matches[j].my_section_data.mesh;
        std::vector<cv::Point2f>* mesh2 = mesh_matches[j].n_section_data.mesh;

        int myz = mesh_matches[j].my_section_data.z;
        int nz = mesh_matches[j].n_section_data.z;

        cv::Point2f* gradients1 = section_data_map[myz].gradients;
        cv::Point2f* gradients2 = section_data_map[nz].gradients;
        double* barys1 = mesh_matches[j].my_barys;
        double* barys2 = mesh_matches[j].n_barys;
        double all_weight = cross_slice_weight;
        double sigma = cross_slice_winsor;

        int indices1[3] = {mesh_matches[j].my_tri.index1,
                           mesh_matches[j].my_tri.index2,
                           mesh_matches[j].my_tri.index3};
        int indices2[3] = {mesh_matches[j].n_tri.index1,
                           mesh_matches[j].n_tri.index2,
                           mesh_matches[j].n_tri.index3};

        cost += crosslink_mesh_derivs(mesh1, mesh2,
                                      gradients1, gradients2,
                                      indices1, indices2,
                                      barys1, barys2,
                                      all_weight, sigma);

        cost += crosslink_mesh_derivs(mesh2, mesh1,
                                      gradients2, gradients1,
                                      indices2, indices1,
                                      barys2, barys1,
                                      all_weight, sigma);
      }

      if (iter == 0) prev_cost = cost+10.0;

      if (cost <= prev_cost) {
        stepsize *= 1.1;
        if (stepsize > 1.0) {
          stepsize = 1.0;
        }
        // TODO(TFK): momentum.

        for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
             it != section_data_map.end(); ++it) {
          std::vector<cv::Point2f>* mesh = it->second.mesh;
          std::vector<cv::Point2f>* mesh_old = it->second.mesh_old;
          cv::Point2f* gradients = it->second.gradients;
          cv::Point2f* gradients_with_momentum = it->second.gradients_with_momentum;
          for (int j = 0; j < mesh->size(); j++) {
            gradients_with_momentum[j] = gradients[j] + momentum*gradients_with_momentum[j];
          }

          for (int j = 0; j < mesh->size(); j++) {
            (*mesh_old)[j] = ((*mesh)[j]);
          }
          for (int j = 0; j < mesh->size(); j++) {
            (*mesh)[j].x -= (float)(stepsize * (gradients_with_momentum)[j].x);
            (*mesh)[j].y -= (float)(stepsize * (gradients_with_momentum)[j].y);
          }
        }
        if (iter%100 == 0) {
          printf("Good step old cost %f, new cost %f, iteration %d\n", prev_cost, cost, iter);
        }
        prev_cost = cost;
      } else {
        stepsize *= 0.5;
        // bad step undo.
        for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
             it != section_data_map.end(); ++it) {
          std::vector<cv::Point2f>* mesh = it->second.mesh;
          std::vector<cv::Point2f>* mesh_old = it->second.mesh_old;
          cv::Point2f* gradients_with_momentum = it->second.gradients_with_momentum;
          for (int j = 0; j < mesh->size(); j++) {
            gradients_with_momentum[j] = cv::Point2f(0.0,0.0);
          }

          //if (mesh_old->size() != mesh->size()) continue;
          for (int j = 0; j < mesh->size(); j++) {
            (*mesh)[j] = (*mesh_old)[j];
          }
        }
        if (iter%1000 == 0) {
          printf("Bad step old cost %f, new cost %f, iteration %d\n", prev_cost, cost, iter);
        }
      }
    }
  } // END BLOCK.
  TFK_STOP_TIMER(&elastic_timer, "After elastic");
}


void filter_overlap_points_3d(Graph<vdata, edata>* graph, align_data_t* p_align_data) {
  for (int v = 0; v < graph->num_vertices(); v++) {
    int curr_z = graph->getVertexData(v)->z;
    int tile_id = graph->getVertexData(v)->tile_id;
    _tile_data& tdata = p_align_data->sec_data[curr_z].tiles[tile_id];
    printf("curr_z is %d, tile_id is %d\n", curr_z, tile_id);
    tdata.ignore = (bool*) calloc(tdata.p_kps_3d->size()+1, sizeof(bool));


    cv::Point2f offset_pt =
        cv::Point2f(graph->getVertexData(v)->start_x + graph->getVertexData(v)->offset_x,
                    graph->getVertexData(v)->start_y + graph->getVertexData(v)->offset_y);
    if (tdata.p_kps_3d->size() == 0) continue;
    for (int i = 0; i < tdata.p_kps_3d->size(); i++) {
      cv::Point2f pt = (*(tdata.p_kps_3d))[i].pt;
      pt.x += offset_pt.x;
      pt.y += offset_pt.y;
      for (int j = 0; j < graph->num_vertices(); j++) {
        if (j == v) continue;
        if (j > v) continue;
        double x_start = graph->getVertexData(j)->start_x + graph->getVertexData(j)->offset_x;
        double x_end = graph->getVertexData(j)->end_x + graph->getVertexData(j)->offset_x;
        double y_start = graph->getVertexData(j)->start_y + graph->getVertexData(j)->offset_y;
        double y_end = graph->getVertexData(j)->end_y + graph->getVertexData(j)->offset_y;
        if (pt.x > x_start && pt.x < x_end && pt.y > y_start && pt.y < y_end) {
          tdata.ignore[i] = true;
        }
      }
    }
    int filter_count = 0;
    for (int i = 0; i < tdata.p_kps_3d->size(); i++) {
      if (tdata.ignore[i]) filter_count++; 
    }
    printf("Filtered %d out of %d keypoints\n", filter_count, (int)tdata.p_kps_3d->size());
  }
}


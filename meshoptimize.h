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
    vertex_data->offset_x += (1.0-2*4.0/30.0)*grad_error_x*0.49/(weight_sum);
    vertex_data->offset_y += (1.0-2*4.0/30.0)*grad_error_y*0.49/(weight_sum);
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


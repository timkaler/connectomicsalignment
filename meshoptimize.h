static int global_iteration_count = 0;
static double global_error_sq = 0;
static double global_learning_rate = 0.1;
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
      //solve(aTa, A.t()*xc, resX, DECOMP_CHOLESKY);
      //solve(aTa, A.t()*yc, resY, DECOMP_CHOLESKY);
      solve(aTa, A.t()*xc, resX, DECOMP_SVD);
      solve(aTa, A.t()*yc, resY, DECOMP_SVD);
  
      // store result
      memcpy(transf.ptr<double>(0), resX.data, m*sizeof(double));
      memcpy(transf.ptr<double>(1), resY.data, m*sizeof(double));
  
      return true;
  }

}

void computeError2DAlign(int vid, void* scheduler_void) {
  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
  Graph<vdata, edata>* graph = reinterpret_cast<Graph<vdata,edata>*>(scheduler->graph_void);

  //printf("starting vertex %d\n", vid);
  vdata* vertex_data = graph->getVertexData(vid);
  std::vector<edata> edges = graph->edgeData[vid];

  //double original_offset_x = vertex_data->offset_x;
  //double original_offset_y = vertex_data->offset_y;

  std::vector<cv::Point2d> source_points(0), dest_points(0);

  if (edges.size() == 0) {
     //printf("Vertex %d has no edges\n", vid);
     return;
  }
  //if (!vertex_data->converged) printf("vertex %d didn't converge\n", vid);

  std::vector<cv::Point2d> original_points;
  std::vector<cv::Point2d> original_points2;
  std::vector<float> weights;
  bool empty_converged = true;
  for (int i = 0; i < edges.size(); i++) {
    std::vector<cv::Point2f>* v_points = edges[i].v_points;
    std::vector<cv::Point2f>* n_points = edges[i].n_points;
    vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
    //if (!neighbor_vertex->converged) continue;
    empty_converged = false;
    double curr_weight = 1.0/v_points->size();

    if (graph->getVertexData(edges[i].neighbor_id)->z != graph->getVertexData(vid)->z) {
      //printf("Danger mesh optimize code has edges across sections!\n");
    }

    for (int j = 0; j < v_points->size(); j++) {
      cv::Point2d ptx1 = transform_point_double(vertex_data, (*v_points)[j]);
      cv::Point2d ptx2 = transform_point_double(neighbor_vertex, (*n_points)[j]);

      source_points.push_back(ptx1);
      dest_points.push_back(ptx2);
      original_points.push_back((*v_points)[j]);
      original_points2.push_back((*n_points)[j]);
      weights.push_back(curr_weight);
    }
  }
  if (empty_converged) return; 
  std::vector<cv::Point2d>& filtered_match_points_a = source_points;
  std::vector<cv::Point2d>& filtered_match_points_b = dest_points;

  std::vector<cv::Point2d> match_points_a_fixed(0);
  if (filtered_match_points_a.size() > 0) {
    double grad_error_x = 0.0;
    double grad_error_y = 0.0;
    double weight_sum = 1.0;
    double error_sq = 0.0;
    for (int iter = 0; iter < filtered_match_points_a.size(); iter++) {
       double delta_x = filtered_match_points_b[iter].x - filtered_match_points_a[iter].x;
       double delta_y = filtered_match_points_b[iter].y - filtered_match_points_a[iter].y;
       error_sq += (delta_x*delta_x + delta_y*delta_y)*weights[iter];
       grad_error_x += (delta_x)*weights[iter];
       grad_error_y += (delta_y)*weights[iter];
       weight_sum += weights[iter];
    }
    //vertex_data->offset_x += /*(1.0-2*4.0/30.0)*/grad_error_x*0.49/(weight_sum);
    //vertex_data->offset_y += /*(1.0-2*4.0/30.0)*/grad_error_y*0.49/(weight_sum);
    //vertex_data->offset_x += (2*4.0/30.0)*zgrad_error_x*0.49/(zweight_sum+weight_sum);
    //vertex_data->offset_y += (2*4.0/30.0)*zgrad_error_y*0.49/(zweight_sum+weight_sum);
    //double error_sq2 = grad_error_x*grad_error_x + grad_error_y*grad_error_y;
  
    double error_sq2 = std::sqrt(grad_error_x*grad_error_x + grad_error_y*grad_error_y);
    //int error_sq2_val = lround(error_sq2);
    global_error_sq += error_sq2;
    //__sync_fetch_and_add(&global_error_sq, error_sq2_val); 
 
    if (error_sq2 > 1e-2) {
       //printf("The error of vertex %d is %f vs %f edges size %d num matches %d\n", vid, error_sq2, error_sq, edges.size(), filtered_match_points_a.size());
    }

    //if ( vertex_data->iteration_count < 5000000 && /*(
    //  std::abs(vertex_data->offset_x - original_offset_x) +
    //  std::abs(vertex_data->offset_y - original_offset_y) > 1e-3)*/ error_sq > 2*100.0) {
    //  int c = __sync_fetch_and_add(&global_iteration_count, 1);
    //  if (c%100000 == 0) {
    //    printf("The error is %f\n", error_sq);
    //  } 
    //  scheduler->add_task(vid, updateVertex2DAlign);
    //  for (int i = 0; i < edges.size(); i++) {
    //    scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
    //  }
    //}
  }
  vertex_data->iteration_count++;
}

//void serialUpdateValues(int vid, void* scheduler_void,
//                               std::priority_queue<std::pair<double, int> >* queue) {
//  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
//  Graph<vdata, edata>* graph = reinterpret_cast<Graph<vdata,edata>*>(scheduler->graph_void);
//  //printf("starting vertex %d\n", vid);
//  vdata* vertex_data = graph->getVertexData(vid);
//  std::vector<edata> edges = graph->edgeData[vid];
//
//  //double original_offset_x = vertex_data->offset_x;
//  //double original_offset_y = vertex_data->offset_y;
//
//  std::vector<cv::Point2d> source_points(0), dest_points(0);
//
//  if (edges.size() == 0) return;
//
//  std::vector<cv::Point2d> original_points;
//  std::vector<cv::Point2d> original_points2;
//  std::vector<double> weights;
//
//  std::vector<vdata*> neighbor_pointers;
//
//  //bool trigger_change = false;
//
//  for (int i = 0; i < edges.size(); i++) {
//    //if (i != vertex_data->iteration_count%edges.size()) continue;
//    std::vector<cv::Point2f>* v_points = edges[i].v_points;
//    std::vector<cv::Point2f>* n_points = edges[i].n_points;
//    vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
//    //int n_conv = neighbor_vertex->converged;
//    //if (n_conv == 0) continue;
//    //if (n_conv > vertex_data->converged) continue;
//
//    //if (!neighbor_vertex->converged) continue;
//    double curr_weight = 1.0/v_points->size();
//
//    if (graph->getVertexData(edges[i].neighbor_id)->z != graph->getVertexData(vid)->z) {
//      //printf("Danger mesh optimize code has edges across sections!\n");
//    }
//
//    for (int j = 0; j < v_points->size(); j++) {
//      cv::Point2d ptx1 = transform_point_double(vertex_data, (*v_points)[j]);
//      cv::Point2d ptx2 = transform_point_double(neighbor_vertex, (*n_points)[j]);
//      neighbor_pointers.push_back(neighbor_vertex); 
//      source_points.push_back(ptx1);
//      dest_points.push_back(ptx2);
//      original_points.push_back((*v_points)[j]);
//      original_points2.push_back((*n_points)[j]);
//      weights.push_back(curr_weight);
//    }
//  }
//  
//  std::vector<cv::Point2d>& filtered_match_points_a = source_points;
//  std::vector<cv::Point2d>& filtered_match_points_b = dest_points;
//
//  std::vector<cv::Point2d> match_points_a_fixed(0);
//  if (filtered_match_points_a.size() > 0) {
//    //double learning_rate = global_learning_rate;//0.4 + 0.6*((rand()%100)*1.0/100);
//    while (true) {
//    double grad_error_x = 0.0;
//    double grad_error_y = 0.0;
//    double weight_sum = 1.0;
//    double error_sq = 0.0;
//    std::map<int, double> neighbor_errors_x;
//    std::map<int, double> neighbor_errors_y;
//
//    for (int iter = 0; iter < filtered_match_points_a.size(); iter++) {
//       double delta_x = filtered_match_points_b[iter].x - filtered_match_points_a[iter].x;
//       double delta_y = filtered_match_points_b[iter].y - filtered_match_points_a[iter].y;
//       error_sq += (delta_x*delta_x + delta_y*delta_y)*weights[iter];
//       grad_error_x += 2*(delta_x)*weights[iter];
//       grad_error_y += 2*(delta_y)*weights[iter];
//       //neighbor_pointers[iter]->neighbor_grad_x -= 2*delta_x*weights[iter];
//       //neighbor_pointers[iter]->neighbor_grad_y -= 2*delta_y*weights[iter];
//       //neighbor_errors_x[neighbor_pointers[iter]->vertex_id] += delta_x*weights[iter];
//       //neighbor_errors_y[neighbor_pointers[iter]->vertex_id] += delta_y*weights[iter];
//       weight_sum += weights[iter];
//    }
//
//    grad_error_x /= weight_sum;
//    grad_error_y /= weight_sum;
//    double total_error = std::sqrt(grad_error_x*grad_error_x + grad_error_y*grad_error_y);
//    //vertex_data->last_error_value = total_error;
//    queue->push(std::make_pair(total_error, vid));
//    break;
//  }
//  vertex_data->iteration_count++;
//  }
//}
//
//void serialUpdateVertex2DAlign(int vid, double check_value, void* scheduler_void,
//                               std::priority_queue<std::pair<double, int> >* queue) {
//  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
//  Graph<vdata, edata>* graph = reinterpret_cast<Graph<vdata,edata>*>(scheduler->graph_void);
//  //printf("starting vertex %d\n", vid);
//  vdata* vertex_data = graph->getVertexData(vid);
//  std::vector<edata> edges = graph->edgeData[vid];
//
//  if (check_value > 0.0 && vertex_data->last_error_value != check_value) return; // stale.
//
//  //double original_offset_x = vertex_data->offset_x;
//  //double original_offset_y = vertex_data->offset_y;
//
//  std::vector<cv::Point2d> source_points(0), dest_points(0);
//
//  if (edges.size() == 0) return;
//
//  std::vector<cv::Point2d> original_points;
//  std::vector<cv::Point2d> original_points2;
//  std::vector<double> weights;
//
//  std::vector<vdata*> neighbor_pointers;
//
//  //bool trigger_change = false;
//
//  for (int i = 0; i < edges.size(); i++) {
//    //if (i != vertex_data->iteration_count%edges.size()) continue;
//    std::vector<cv::Point2f>* v_points = edges[i].v_points;
//    std::vector<cv::Point2f>* n_points = edges[i].n_points;
//    vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
//    //int n_conv = neighbor_vertex->converged;
//    //if (n_conv == 0) continue;
//    //if (n_conv > vertex_data->converged) continue;
//
//    //if (!neighbor_vertex->converged) continue;
//    double curr_weight = 1.0/v_points->size();
//
//    if (graph->getVertexData(edges[i].neighbor_id)->z != graph->getVertexData(vid)->z) {
//      //printf("Danger mesh optimize code has edges across sections!\n");
//    }
//
//    for (int j = 0; j < v_points->size(); j++) {
//      cv::Point2f ptx1 = transform_point_double(vertex_data, (*v_points)[j]);
//      cv::Point2f ptx2 = transform_point_double(neighbor_vertex, (*n_points)[j]);
//      neighbor_pointers.push_back(neighbor_vertex); 
//      source_points.push_back(ptx1);
//      dest_points.push_back(ptx2);
//      original_points.push_back((*v_points)[j]);
//      original_points2.push_back((*n_points)[j]);
//      weights.push_back(curr_weight);
//    }
//  }
//  
//  std::vector<cv::Point2d>& filtered_match_points_a = source_points;
//  std::vector<cv::Point2d>& filtered_match_points_b = dest_points;
//
//  std::vector<cv::Point2d> match_points_a_fixed(0);
//  if (filtered_match_points_a.size() > 0) {
//    double learning_rate = global_learning_rate;//0.4 + 0.6*((rand()%100)*1.0/100);
//    while (true) {
//    double grad_error_x = 0.0;
//    double grad_error_y = 0.0;
//    double weight_sum = 1.0;
//    double error_sq = 0.0;
//    std::map<int, double> neighbor_errors_x;
//    std::map<int, double> neighbor_errors_y;
//
//    for (int iter = 0; iter < filtered_match_points_a.size(); iter++) {
//       double delta_x = filtered_match_points_b[iter].x - filtered_match_points_a[iter].x;
//       double delta_y = filtered_match_points_b[iter].y - filtered_match_points_a[iter].y;
//       error_sq += (delta_x*delta_x + delta_y*delta_y)*weights[iter];
//       grad_error_x += 2*(delta_x)*weights[iter];
//       grad_error_y += 2*(delta_y)*weights[iter];
//       //neighbor_pointers[iter]->neighbor_grad_x -= 2*delta_x*weights[iter];
//       //neighbor_pointers[iter]->neighbor_grad_y -= 2*delta_y*weights[iter];
//       //neighbor_errors_x[neighbor_pointers[iter]->vertex_id] += delta_x*weights[iter];
//       //neighbor_errors_y[neighbor_pointers[iter]->vertex_id] += delta_y*weights[iter];
//       weight_sum += weights[iter];
//    }
//
//    vertex_data->offset_x += /*(1.0-2*4.0/30.0)*/grad_error_x*learning_rate/(weight_sum);
//    vertex_data->offset_y += /*(1.0-2*4.0/30.0)*/grad_error_y*learning_rate/(weight_sum);
//
//    serialUpdateValues(vid, scheduler_void, queue);
//    for (int i = 0; i < edges.size(); i++) {
//      serialUpdateValues(edges[i].neighbor_id, scheduler_void, queue);
//    }
//    break;
//  }
//  vertex_data->iteration_count++;
//  }
//}

void updateVertex2DAlignMFOV(int vid, void* scheduler_void) {
  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
  Graph<vdata, edata>* graph = reinterpret_cast<Graph<vdata,edata>*>(scheduler->graph_void);
  //printf("starting vertex %d\n", vid);
  vdata* vertex_data = graph->getVertexData(vid);
  std::vector<edata> edges = graph->edgeData[vid];

  //double original_offset_x = vertex_data->offset_x;
  //double original_offset_y = vertex_data->offset_y;

  std::vector<cv::Point2f> source_points(0), dest_points(0);

  if (edges.size() == 0) return;

  std::vector<cv::Point2f> original_points;
  std::vector<cv::Point2f> original_points2;
  std::vector<float> weights;

  std::vector<vdata*> neighbor_pointers;

  //bool trigger_change = false;

  //for (int i = 0; i < edges.size(); i++) {
  //  //if (i != vertex_data->iteration_count%edges.size()) continue;
  //  vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
  //  int n_conv = neighbor_vertex->converged;
  //  if (n_conv == 0) continue;
  //  if (vertex_data->converged == 0) {
  //    vertex_data->converged = n_conv;
  //    trigger_change = true;
  //  }

  //  if (n_conv > vertex_data->converged) continue;
  //  if (n_conv + 1< vertex_data->converged) {
  //    vertex_data->converged = n_conv+1;
  //    trigger_change = true;
  //  }
  //}


  for (int i = 0; i < edges.size(); i++) {
    //if (i != vertex_data->iteration_count%edges.size()) continue;
    std::vector<cv::Point2f>* v_points = edges[i].v_points;
    std::vector<cv::Point2f>* n_points = edges[i].n_points;
    vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
    //int n_conv = neighbor_vertex->converged;
    //if (n_conv == 0) continue;
    //if (n_conv > vertex_data->converged) continue;

    //if (!neighbor_vertex->converged) continue;
    double curr_weight = 1.0/v_points->size();

    if (graph->getVertexData(edges[i].neighbor_id)->z != graph->getVertexData(vid)->z) {
      //printf("Danger mesh optimize code has edges across sections!\n");
    }

    for (int j = 0; j < v_points->size(); j++) {
      cv::Point2f ptx1 = transform_point(vertex_data, (*v_points)[j]);
      cv::Point2f ptx2 = transform_point(neighbor_vertex, (*n_points)[j]);
      neighbor_pointers.push_back(neighbor_vertex); 
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
    double learning_rate = global_learning_rate;//0.4 + 0.6*((rand()%100)*1.0/100);
    while (true) {
    double grad_error_x = 0.0;
    double grad_error_y = 0.0;
    double weight_sum = 1.0;
    double error_sq = 0.0;
    std::map<int, double> neighbor_errors_x;
    std::map<int, double> neighbor_errors_y;

    for (int iter = 0; iter < filtered_match_points_a.size(); iter++) {
       double delta_x = filtered_match_points_b[iter].x - filtered_match_points_a[iter].x;
       double delta_y = filtered_match_points_b[iter].y - filtered_match_points_a[iter].y;
       error_sq += (delta_x*delta_x + delta_y*delta_y)*weights[iter];
       grad_error_x += 2*(delta_x)*weights[iter];
       grad_error_y += 2*(delta_y)*weights[iter];
       //neighbor_pointers[iter]->neighbor_grad_x -= 2*delta_x*weights[iter];
       //neighbor_pointers[iter]->neighbor_grad_y -= 2*delta_y*weights[iter];
       //neighbor_errors_x[neighbor_pointers[iter]->vertex_id] += delta_x*weights[iter];
       //neighbor_errors_y[neighbor_pointers[iter]->vertex_id] += delta_y*weights[iter];
       weight_sum += weights[iter];
    }

    //vertex_data->neighbor_grad_x = 0.0;
    //vertex_data->neighbor_grad_y = 0.0;

    vertex_data->offset_x += /*(1.0-2*4.0/30.0)*/grad_error_x*learning_rate/(weight_sum);
    vertex_data->offset_y += /*(1.0-2*4.0/30.0)*/grad_error_y*learning_rate/(weight_sum);



    //double total_error = std::sqrt(grad_error_x*grad_error_x + grad_error_y*grad_error_y);
    if (/*vertex_data->iteration_count < 40000*/ (vertex_data->iteration_count < 1000 /* || error_sq > 25.0*edges.size()*/ /*|| trigger_change*/)/*(
      std::abs(vertex_data->offset_x - original_offset_x) +
      std::abs(vertex_data->offset_y - original_offset_y) > 1.0)*/ /*error_sq > 50.0*/) {
      //int c = __sync_fetch_and_add(&global_iteration_count, 1);
      //if (vertex_data->iteration_count > 10000*0.9 && grad_error_x*grad_error_x + grad_error_y*grad_error_y > 10.0) vertex_data->iteration_count -= 2;
      //if (c%100000 == 0) {
      //  printf("The error is %f\n", total_error);
      //}
      scheduler->add_task(vid, updateVertex2DAlignMFOV);
      for (int i = 0; i < edges.size(); i++) {
        //int n_conv = edges[i].neighbor_id;
        //if (n_conv == 0 || n_conv <= vertex_data->converged) {
          scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlignMFOV);
        //}
      }
    } else {
      //for (int i = 0; i < edges.size(); i++) {
      //  int n_conv = edges[i].neighbor_id;
      //  if (n_conv == 0 || n_conv < vertex_data->converged) {
      //    scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
      //  }
      //}
    } /*else {
      vertex_data->converged = true; 
    }*/
 /*else {
      for (int i = 0; i < edges.size(); i++) {
        double ngrad_x = graph->getVertexData(edges[i].neighbor_id)->neighbor_grad_x;
        double ngrad_y = graph->getVertexData(edges[i].neighbor_id)->neighbor_grad_y;
        if (ngrad_x*ngrad_x+ngrad_y*ngrad_y > 20.0) {
          scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
        }
      }
    }*/
  break;
  }
  vertex_data->iteration_count++;
  }
}

void updateVertex2DAlignFULLFast(int vid, void* scheduler_void) {
  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
  Graph<vdata, edata>* graph = reinterpret_cast<Graph<vdata,edata>*>(scheduler->graph_void);

  vdata* vertex_data = graph->getVertexData(vid);
  std::vector<edata>& edges = graph->edgeData[vid];

  if (edges.size() == 0) return;

  double learning_rate = global_learning_rate;//0.4 + 0.6*((rand()%100)*1.0/100);
  double grad_error_x = 0.0;
  double grad_error_y = 0.0;
  double weight_sum = 1.0;
  //int added_points = 0;
  for (int i = 0; i < edges.size(); i++) {
    std::vector<cv::Point2f>* v_points = edges[i].v_points;
    std::vector<cv::Point2f>* n_points = edges[i].n_points;
    vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);

    double curr_weight = 1.0/v_points->size();

    for (int j = 0; j < v_points->size(); j++) {
      cv::Point2f ptx1 = transform_point(vertex_data, (*v_points)[j]);
      cv::Point2f ptx2 = transform_point(neighbor_vertex, (*n_points)[j]);

      double delta_x = ptx2.x - ptx1.x;
      double delta_y = ptx2.y - ptx1.y;
      grad_error_x += 2 * delta_x * curr_weight;
      grad_error_y += 2 * delta_y * curr_weight;
      weight_sum += curr_weight;
    }
  }

  // update the gradients.
  vertex_data->offset_x += grad_error_x*learning_rate/(weight_sum);
  vertex_data->offset_y += grad_error_y*learning_rate/(weight_sum);

  if (vertex_data->iteration_count < 2500) {
    scheduler->add_task(vid, updateVertex2DAlignFULLFast);
    //for (int i = 0; i < edges.size(); i++) {
    //  //int n_conv = edges[i].neighbor_id;
    //  //if (n_conv == 0 || n_conv <= vertex_data->converged) {
    //    scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlignFULL);
    //  //}
    //}
  }
  vertex_data->iteration_count++;
}


void updateVertex2DAlignFULL(int vid, void* scheduler_void) {
  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
  Graph<vdata, edata>* graph = reinterpret_cast<Graph<vdata,edata>*>(scheduler->graph_void);
  //printf("starting vertex %d\n", vid);
  vdata* vertex_data = graph->getVertexData(vid);
  std::vector<edata> edges = graph->edgeData[vid];

  ///double original_offset_x = vertex_data->offset_x;
  ///double original_offset_y = vertex_data->offset_y;

  std::vector<cv::Point2f> source_points(0), dest_points(0);

  if (edges.size() == 0) return;

  std::vector<cv::Point2f> original_points;
  std::vector<cv::Point2f> original_points2;
  std::vector<float> weights;

  std::vector<vdata*> neighbor_pointers;


  for (int i = 0; i < edges.size(); i++) {
    //if (i != vertex_data->iteration_count%edges.size()) continue;
    std::vector<cv::Point2f>* v_points = edges[i].v_points;
    std::vector<cv::Point2f>* n_points = edges[i].n_points;
    vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
    //int n_conv = neighbor_vertex->converged;
    //if (n_conv == 0) continue;
    //if (n_conv > vertex_data->converged) continue;

    //if (!neighbor_vertex->converged) continue;
    double curr_weight = 1.0/v_points->size();

    if (graph->getVertexData(edges[i].neighbor_id)->z != graph->getVertexData(vid)->z) {
      //printf("Danger mesh optimize code has edges across sections!\n");
    }

    for (int j = 0; j < v_points->size(); j++) {
      cv::Point2f ptx1 = transform_point(vertex_data, (*v_points)[j]);
      cv::Point2f ptx2 = transform_point(neighbor_vertex, (*n_points)[j]);
      neighbor_pointers.push_back(neighbor_vertex); 
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
    double learning_rate = global_learning_rate;//0.4 + 0.6*((rand()%100)*1.0/100);
    if (true) {
    double grad_error_x = 0.0;
    double grad_error_y = 0.0;
    double weight_sum = 1.0;
    double error_sq = 0.0;
    std::map<int, double> neighbor_errors_x;
    std::map<int, double> neighbor_errors_y;

    for (int iter = 0; iter < filtered_match_points_a.size(); iter++) {
       double delta_x = filtered_match_points_b[iter].x - filtered_match_points_a[iter].x;
       double delta_y = filtered_match_points_b[iter].y - filtered_match_points_a[iter].y;
       error_sq += (delta_x*delta_x + delta_y*delta_y)*weights[iter];
       grad_error_x += 2*(delta_x)*weights[iter];
       grad_error_y += 2*(delta_y)*weights[iter];
       //neighbor_pointers[iter]->neighbor_grad_x -= 2*delta_x*weights[iter];
       //neighbor_pointers[iter]->neighbor_grad_y -= 2*delta_y*weights[iter];
       //neighbor_errors_x[neighbor_pointers[iter]->vertex_id] += delta_x*weights[iter];
       //neighbor_errors_y[neighbor_pointers[iter]->vertex_id] += delta_y*weights[iter];
       weight_sum += weights[iter];
    }
    //grad_error_x += vertex_data->neighbor_grad_x;
    //grad_error_y += vertex_data->neighbor_grad_y;
    //for (std::map<int, double>::iterator iter = neighbor_errors_x.begin(); iter != neighbor_errors_x.end(); ++iter) {
    //  graph->getVertexData(iter->first)->neighbor_grad_x += vertex_data->neighbor_grad_x*0.4;
    //  graph->getVertexData(iter->first)->neighbor_grad_y += vertex_data->neighbor_grad_y*0.4;
    //}

    //vertex_data->neighbor_grad_x = 0.0;
    //vertex_data->neighbor_grad_y = 0.0;
    //if (!vertex_data->converged) {
      vertex_data->offset_x += /*(1.0-2*4.0/30.0)*/grad_error_x*learning_rate/(weight_sum);
      vertex_data->offset_y += /*(1.0-2*4.0/30.0)*/grad_error_y*learning_rate/(weight_sum);
    //}

    //for (int iter = 0; iter < filtered_match_points_a.size(); iter++) {
    //   double delta_x = filtered_match_points_b[iter].x - filtered_match_points_a[iter].x;
    //   double delta_y = filtered_match_points_b[iter].y - filtered_match_points_a[iter].y;
    //   //error_sq += (delta_x*delta_x + delta_y*delta_y)*weights[iter];
    //   //grad_error_x += 2*(delta_x)*weights[iter];
    //   //grad_error_y += 2*(delta_y)*weights[iter];

    //   //neighbor_pointers[iter]->neighbor_grad_x -= 0.4*2*delta_x*weights[iter];
    //   //neighbor_pointers[iter]->neighbor_grad_y -= 0.4*2*delta_y*weights[iter];
    //   //neighbor_errors_x[neighbor_pointers[iter]->vertex_id] += delta_x*weights[iter];
    //   //neighbor_errors_y[neighbor_pointers[iter]->vertex_id] += delta_y*weights[iter];
    //   weight_sum += weights[iter];
    //}


    //error_sq = 0.0;
    //for (std::map<int, double>::iterator iter = neighbor_errors_x.begin(); iter != neighbor_errors_x.end(); ++iter) {
    //  error_sq += (iter->second)*(iter->second);
    //}
    //for (std::map<int, double>::iterator iter = neighbor_errors_y.begin(); iter != neighbor_errors_y.end(); ++iter) {
    //  error_sq += (iter->second)*(iter->second);
    //}


    //vertex_data->offset_x += (2*4.0/30.0)*zgrad_error_x*0.49/(zweight_sum+weight_sum);
    //vertex_data->offset_y += (2*4.0/30.0)*zgrad_error_y*0.49/(zweight_sum+weight_sum);
    //double total_error = std::sqrt(grad_error_x*grad_error_x + grad_error_y*grad_error_y);
    //if (/*vertex_data->iteration_count < 40000*/ (total_error > 1e-4*edges.size() /* || error_sq > 25.0*edges.size()*/ || trigger_change)/*(
    if (/*vertex_data->iteration_count < 40000*/ (vertex_data->iteration_count < 5000 /* || error_sq > 25.0*edges.size()*/)/*(
      std::abs(vertex_data->offset_x - original_offset_x) +
      std::abs(vertex_data->offset_y - original_offset_y) > 1.0)*/ /*error_sq > 50.0*/) {
      //int c = __sync_fetch_and_add(&global_iteration_count, 1);
      //if (vertex_data->iteration_count > 10000*0.9 && grad_error_x*grad_error_x + grad_error_y*grad_error_y > 10.0) vertex_data->iteration_count -= 2;
      //if (c%100000 == 0) {
      //  printf("The error is %f\n", total_error);
      //}
      scheduler->add_task(vid, updateVertex2DAlignFULL);
      for (int i = 0; i < edges.size(); i++) {
        //int n_conv = edges[i].neighbor_id;
        //if (n_conv == 0 || n_conv <= vertex_data->converged) {
          scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlignFULL);
        //}
      }
    } else {
      //for (int i = 0; i < edges.size(); i++) {
      //  int n_conv = edges[i].neighbor_id;
      //  if (n_conv == 0 || n_conv < vertex_data->converged) {
      //    scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
      //  }
      //}
    } /*else {
      vertex_data->converged = true; 
    }*/
 /*else {
      for (int i = 0; i < edges.size(); i++) {
        double ngrad_x = graph->getVertexData(edges[i].neighbor_id)->neighbor_grad_x;
        double ngrad_y = graph->getVertexData(edges[i].neighbor_id)->neighbor_grad_y;
        if (ngrad_x*ngrad_x+ngrad_y*ngrad_y > 20.0) {
          scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
        }
      }
    }*/
  //break;
  }
  vertex_data->iteration_count++;
  }
}


void updateVertex2DAlign(int vid, void* scheduler_void) {
  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
  Graph<vdata, edata>* graph = reinterpret_cast<Graph<vdata,edata>*>(scheduler->graph_void);
  //printf("starting vertex %d\n", vid);
  vdata* vertex_data = graph->getVertexData(vid);
  std::vector<edata> edges = graph->edgeData[vid];

  //double original_offset_x = vertex_data->offset_x;
  //double original_offset_y = vertex_data->offset_y;

  std::vector<cv::Point2f> source_points(0), dest_points(0);

  if (edges.size() == 0) return;

  std::vector<cv::Point2f> original_points;
  std::vector<cv::Point2f> original_points2;
  std::vector<float> weights;

  std::vector<vdata*> neighbor_pointers;

  //bool trigger_change = false;

  //for (int i = 0; i < edges.size(); i++) {
  //  //if (i != vertex_data->iteration_count%edges.size()) continue;
  //  vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
  //  if (neighbor_vertex->mfov_id != vertex_data->mfov_id) continue;
  //  int n_conv = neighbor_vertex->converged;
  //  if (n_conv == 0) continue;
  //  if (vertex_data->converged == 0) {
  //    vertex_data->converged = n_conv;
  //    trigger_change = true;
  //  }

  //  if (n_conv > vertex_data->converged) continue;
  //  if (n_conv + 1< vertex_data->converged) {
  //    vertex_data->converged = n_conv+1;
  //    trigger_change = true;
  //  }
  //}


  for (int i = 0; i < edges.size(); i++) {
    //if (i != vertex_data->iteration_count%edges.size()) continue;
    std::vector<cv::Point2f>* v_points = edges[i].v_points;
    std::vector<cv::Point2f>* n_points = edges[i].n_points;
    vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
    if (neighbor_vertex->mfov_id != vertex_data->mfov_id) continue;
    //int n_conv = neighbor_vertex->converged;
    //if (n_conv == 0) continue;
    //if (n_conv > vertex_data->converged) continue;

    //if (!neighbor_vertex->converged) continue;
    double curr_weight = 1.0/v_points->size();

    if (graph->getVertexData(edges[i].neighbor_id)->z != graph->getVertexData(vid)->z) {
      //printf("Danger mesh optimize code has edges across sections!\n");
    }

    for (int j = 0; j < v_points->size(); j++) {
      cv::Point2f ptx1 = transform_point(vertex_data, (*v_points)[j]);
      cv::Point2f ptx2 = transform_point(neighbor_vertex, (*n_points)[j]);
      neighbor_pointers.push_back(neighbor_vertex); 
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
    double learning_rate = global_learning_rate;//0.4 + 0.6*((rand()%100)*1.0/100);
    while (true) {
    double grad_error_x = 0.0;
    double grad_error_y = 0.0;
    double weight_sum = 1.0;
    double error_sq = 0.0;
    std::map<int, double> neighbor_errors_x;
    std::map<int, double> neighbor_errors_y;

    for (int iter = 0; iter < filtered_match_points_a.size(); iter++) {
       double delta_x = filtered_match_points_b[iter].x - filtered_match_points_a[iter].x;
       double delta_y = filtered_match_points_b[iter].y - filtered_match_points_a[iter].y;
       error_sq += (delta_x*delta_x + delta_y*delta_y)*weights[iter];
       grad_error_x += 2*(delta_x)*weights[iter];
       grad_error_y += 2*(delta_y)*weights[iter];
       //neighbor_pointers[iter]->neighbor_grad_x -= 2*delta_x*weights[iter];
       //neighbor_pointers[iter]->neighbor_grad_y -= 2*delta_y*weights[iter];
       //neighbor_errors_x[neighbor_pointers[iter]->vertex_id] += delta_x*weights[iter];
       //neighbor_errors_y[neighbor_pointers[iter]->vertex_id] += delta_y*weights[iter];
       weight_sum += weights[iter];
    }
    //grad_error_x += vertex_data->neighbor_grad_x;
    //grad_error_y += vertex_data->neighbor_grad_y;
    //for (std::map<int, double>::iterator iter = neighbor_errors_x.begin(); iter != neighbor_errors_x.end(); ++iter) {
    //  graph->getVertexData(iter->first)->neighbor_grad_x += vertex_data->neighbor_grad_x*0.4;
    //  graph->getVertexData(iter->first)->neighbor_grad_y += vertex_data->neighbor_grad_y*0.4;
    //}

    //vertex_data->neighbor_grad_x = 0.0;
    //vertex_data->neighbor_grad_y = 0.0;
    if (isnan(grad_error_x*learning_rate/(weight_sum))) return; 
    if (isnan(grad_error_y*learning_rate/(weight_sum))) return; 
    //if (!vertex_data->converged) {
      vertex_data->offset_x += /*(1.0-2*4.0/30.0)*/grad_error_x*learning_rate/(weight_sum);
      vertex_data->offset_y += /*(1.0-2*4.0/30.0)*/grad_error_y*learning_rate/(weight_sum);
    //}

    //for (int iter = 0; iter < filtered_match_points_a.size(); iter++) {
    //   double delta_x = filtered_match_points_b[iter].x - filtered_match_points_a[iter].x;
    //   double delta_y = filtered_match_points_b[iter].y - filtered_match_points_a[iter].y;
    //   //error_sq += (delta_x*delta_x + delta_y*delta_y)*weights[iter];
    //   //grad_error_x += 2*(delta_x)*weights[iter];
    //   //grad_error_y += 2*(delta_y)*weights[iter];

    //   //neighbor_pointers[iter]->neighbor_grad_x -= 0.4*2*delta_x*weights[iter];
    //   //neighbor_pointers[iter]->neighbor_grad_y -= 0.4*2*delta_y*weights[iter];
    //   //neighbor_errors_x[neighbor_pointers[iter]->vertex_id] += delta_x*weights[iter];
    //   //neighbor_errors_y[neighbor_pointers[iter]->vertex_id] += delta_y*weights[iter];
    //   weight_sum += weights[iter];
    //}


    //error_sq = 0.0;
    //for (std::map<int, double>::iterator iter = neighbor_errors_x.begin(); iter != neighbor_errors_x.end(); ++iter) {
    //  error_sq += (iter->second)*(iter->second);
    //}
    //for (std::map<int, double>::iterator iter = neighbor_errors_y.begin(); iter != neighbor_errors_y.end(); ++iter) {
    //  error_sq += (iter->second)*(iter->second);
    //}


    //vertex_data->offset_x += (2*4.0/30.0)*zgrad_error_x*0.49/(zweight_sum+weight_sum);
    //vertex_data->offset_y += (2*4.0/30.0)*zgrad_error_y*0.49/(zweight_sum+weight_sum);
    //double total_error = std::sqrt(grad_error_x*grad_error_x + grad_error_y*grad_error_y);
    //if (/*vertex_data->iteration_count < 40000*/ (total_error > 1e-4*edges.size() /* || error_sq > 25.0*edges.size()*/ || trigger_change)/*(
    if (/*vertex_data->iteration_count < 40000*/ (vertex_data->iteration_count < 5000)/*(
      std::abs(vertex_data->offset_x - original_offset_x) +
      std::abs(vertex_data->offset_y - original_offset_y) > 1.0)*/ /*error_sq > 50.0*/) {
      //int c = __sync_fetch_and_add(&global_iteration_count, 1);
      //if (vertex_data->iteration_count > 10000*0.9 && grad_error_x*grad_error_x + grad_error_y*grad_error_y > 10.0) vertex_data->iteration_count -= 2;
      //if (c%100000 == 0) {
      //  printf("The error is %f\n", total_error);
      //}
      scheduler->add_task(vid, updateVertex2DAlign);
      for (int i = 0; i < edges.size(); i++) {
        //int n_conv = edges[i].neighbor_id;
        //if (n_conv == 0 || n_conv <= vertex_data->converged) {
          scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
        //}
      }
    } else {
      //for (int i = 0; i < edges.size(); i++) {
      //  int n_conv = edges[i].neighbor_id;
      //  if (n_conv == 0 || n_conv < vertex_data->converged) {
      //    scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
      //  }
      //}
    } /*else {
      vertex_data->converged = true; 
    }*/
 /*else {
      for (int i = 0; i < edges.size(); i++) {
        double ngrad_x = graph->getVertexData(edges[i].neighbor_id)->neighbor_grad_x;
        double ngrad_y = graph->getVertexData(edges[i].neighbor_id)->neighbor_grad_y;
        if (ngrad_x*ngrad_x+ngrad_y*ngrad_y > 20.0) {
          scheduler->add_task(edges[i].neighbor_id, updateVertex2DAlign);
        }
      }
    }*/
  break;
  }
  vertex_data->iteration_count++;
  }
}

#include "fine_alignment.h"
//#include "mfov_alignment.h"

void coarse_alignment_3d(Graph<vdata, edata>* merged_graph, align_data_t* p_align_data, double distance_thresh){

  for (int v = 0; v < merged_graph->num_vertices(); v++) {
    merged_graph->getVertexData(v)->offset_x += merged_graph->getVertexData(v)->start_x;
    merged_graph->getVertexData(v)->offset_y += merged_graph->getVertexData(v)->start_y;
    merged_graph->getVertexData(v)->start_x = 0.0;
    merged_graph->getVertexData(v)->start_y = 0.0;
    merged_graph->getVertexData(v)->end_x = 0.0;
    merged_graph->getVertexData(v)->end_y = 0.0;
  }


  // let's store the transforms in a vector. //NOTE(TFK): Entry 0 will be junk for coarse_align_3d.
  std::vector<cv::Mat> section_transforms(p_align_data->n_sections);

  cilk_for (int section = 1; section < p_align_data->n_sections; section++) {
    int section_a = section-1;
    int section_b = section;
    std::vector <cv::KeyPoint > atile_kps_in_overlap;
    std::vector <cv::Mat > atile_kps_desc_in_overlap_list;
    std::vector<int> atile_kps_tile_list;
    std::vector <cv::KeyPoint > btile_kps_in_overlap;
    std::vector <cv::Mat > btile_kps_desc_in_overlap_list;
    std::vector<int> btile_kps_tile_list;

    std::vector<double> atile_weights;
    std::vector<double> btile_weights;

    for (int v = 0; v < merged_graph->num_vertices(); v++) {
      if (merged_graph->edgeData[v].size() == 0 && false) continue;
      //printf("Vertex is %d\n", v);
      //printf("tile_id is is %d\n", merged_graph->getVertexData(v)->vertex_id);
      if (merged_graph->getVertexData(v)->z == section_a /*|| merged_graph->getVertexData(v)->z == section_a-1*/) {
        int curr_z = merged_graph->getVertexData(v)->z;
        //printf("curr_z is %d\n", curr_z);
        _tile_data tdata_a = p_align_data->sec_data[curr_z].tiles[merged_graph->getVertexData(v)->tile_id];
        concat_two_tiles_all(merged_graph->getVertexData(v), &tdata_a, v, atile_kps_in_overlap, atile_kps_desc_in_overlap_list, atile_kps_tile_list);
        
      } else if (merged_graph->getVertexData(v)->z == section_b) {
        _tile_data tdata_b = p_align_data->sec_data[section_b].tiles[merged_graph->getVertexData(v)->tile_id];
        concat_two_tiles_all(merged_graph->getVertexData(v), &tdata_b, v, btile_kps_in_overlap, btile_kps_desc_in_overlap_list, btile_kps_tile_list);
      }
    }

    printf("Total size of a tile kps is %lu\n", atile_kps_in_overlap.size());
    printf("Total size of b tile kps is %lu\n", btile_kps_in_overlap.size());

    cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
    cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
    cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));

    std::vector< cv::DMatch > matches;
    match_features(matches,
                   atile_kps_desc_in_overlap,
                   btile_kps_desc_in_overlap,
                   0.92);

   printf("Done with the matching. Num matches is %lu\n", matches.size());
   // Filter the matches with RANSAC
   std::vector<cv::Point2f> match_points_a, match_points_b;

   p_align_data->sec_data[section_a].p_kps = new std::vector<cv::KeyPoint>();
   p_align_data->sec_data[section_b].p_kps = new std::vector<cv::KeyPoint>();

   for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
     match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
     match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
     p_align_data->sec_data[section_a].p_kps->push_back(atile_kps_in_overlap[matches[tmpi].queryIdx]);
     p_align_data->sec_data[section_b].p_kps->push_back(btile_kps_in_overlap[matches[tmpi].trainIdx]);
   }

   bool* mask = (bool*)calloc(matches.size()+1, 1);
   std::pair<double,double> offset_pair;

   // pre-filter matches with very forgiving ransac threshold.
   tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 1024, mask);
   std::vector< cv::Point2f > filtered_match_points_a_pre(0);
   std::vector< cv::Point2f > filtered_match_points_b_pre(0);

   int num_filtered = 0;
   for (int c = 0; c < matches.size(); c++) {
      if (mask[c]) {
        num_filtered++;
        filtered_match_points_a_pre.push_back(
            match_points_a[c]);
        filtered_match_points_b_pre.push_back(
            match_points_b[c]);
      }
    }
    free(mask);

    mask = (bool*)calloc(matches.size()+1, 1);
    printf("First pass filter got %d matches\n", num_filtered);
    if (num_filtered < 32) {
      printf("Not enough matches, skipping section\n");
      continue;
    }
    //tfk_simple_ransac(filtered_match_points_a_pre, filtered_match_points_b_pre, distance_thresh, mask);
    tfk_simple_ransac_strict_ret_affine(filtered_match_points_a_pre, filtered_match_points_b_pre, distance_thresh, mask);

    std::vector< cv::Point2f > filtered_match_points_a(0);
    std::vector< cv::Point2f > filtered_match_points_b(0);

    num_filtered = 0;
    for (int c = 0; c < filtered_match_points_a_pre.size(); c++) {
      if (mask[c]) {
        num_filtered++;
        filtered_match_points_a.push_back(
            filtered_match_points_a_pre[c]);
        filtered_match_points_b.push_back(
            filtered_match_points_b_pre[c]);
      }
    }
    printf("Second pass filter got %d matches\n", num_filtered);


    if (num_filtered < 12) {
      printf("Not enough matches %d for section %d with thresh\n", num_filtered, section_a);
      continue;
    } else {
      printf("Got enough matches %d for section %d with thresh\n", num_filtered, section_a);
    }

    //cv::Mat warp_mat;
    cv::computeAffineTFK(filtered_match_points_a, filtered_match_points_b, section_transforms[section]/*warp_mat*/);

    std::cout << section_transforms[section] << std::endl;

    //vdata tmp;
    //tmp.a00 = warp_mat.at<double>(0, 0); 
    //tmp.a01 = warp_mat.at<double>(0, 1);
    //tmp.offset_x = warp_mat.at<double>(0, 2);
    //tmp.a10 = warp_mat.at<double>(1, 0); 
    //tmp.a11 = warp_mat.at<double>(1, 1); 
    //tmp.offset_y = warp_mat.at<double>(1, 2);
    //tmp.start_x = 0.0;
    //tmp.start_y = 0.0;
    //printf("Best values are %f %f %f %f %f %f\n", tmp.a00, tmp.a01, tmp.a10, tmp.a11, tmp.offset_x, tmp.offset_y);
    //best_vertex_data = tmp;
    //float error_sqrt = 0.0;
    //for (int t = 0; t < filtered_match_points_a.size(); t++) {
    //  cv::Point2f t_point = transform_point(&best_vertex_data,filtered_match_points_a[t]);
    //  cv::Point2f o_point = filtered_match_points_b[t];
    //  float error_sq = (t_point.x-o_point.x)*(t_point.x-o_point.x) +
    //                   (t_point.y-o_point.y)*(t_point.y-o_point.y);
    //  error_sqrt += std::sqrt(error_sq);
    //}

    //printf("Total error sqrt is total:%f avg:%f\n", error_sqrt, error_sqrt / filtered_match_points_a.size());
    //std::set<int> sections_done;
    //sections_done.clear();
    //for (int v = 0; v < merged_graph->num_vertices(); v++) {
    //  if (merged_graph->getVertexData(v)->z <= section_a) {
    //    updateAffineTransform(merged_graph->getVertexData(v), warp_mat);
    //    if (sections_done.find(merged_graph->getVertexData(v)->z) == sections_done.end()) {
    //      sections_done.insert(merged_graph->getVertexData(v)->z);
    //      updateAffineSectionTransform(merged_graph->getVertexData(v), warp_mat);
    //    }
    //    continue;
    //  }
    //}
  }

  // now apply the transformations.
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





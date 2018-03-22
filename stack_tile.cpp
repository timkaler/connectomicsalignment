

void updateTile2DAlign(int vid, void* scheduler_void) {
  //double global_learning_rate = 0.49;
   
  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
  Graph* graph = reinterpret_cast<Graph*>(scheduler->graph_void);

  vdata* vertex_data = graph->getVertexData(vid);
  tfk::Tile* tile = (tfk::Tile*) vertex_data->tile;

  if (vid != tile->tile_id) printf("Failure!\n");

  tile->local2DAlignUpdate();

  if (vertex_data->iteration_count < 5000) {
    scheduler->add_task(vid, updateTile2DAlign);
  }
  vertex_data->iteration_count++;
}


float tfk::Tile::compute_deviation(Tile* b_tile) {
      cv::Point2f a_point = cv::Point2f(this->x_start + this->offset_x,
                                        this->y_start + this->offset_y);
      cv::Point2f b_point = cv::Point2f(b_tile->x_start+b_tile->offset_x,
                                        b_tile->y_start+b_tile->offset_y);
      cv::Point2f delta = a_point-b_point;
      cv::Point2f idelta = this->ideal_offsets[b_tile->tile_id];
      float dx = delta.x-idelta.x; 
      float dy = delta.y-idelta.y; 
      return std::sqrt(dx*dx+dy*dy);
}

float tfk::Tile::error_tile_pair(Tile *other) {
  if (!(this->overlaps_with(other))) {
    return -2;
  }


  cv::Mat tile_p_image_1;
  cv::Mat tile_p_image_2;
  tile_p_image_1 = this->get_tile_data(Resolution::FULL); //cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
  tile_p_image_2 = other->get_tile_data(Resolution::FULL); //cv::imread(other->filepath, CV_LOAD_IMAGE_UNCHANGED);


  std::pair<cv::Point2f, cv::Point2f> tile_1_bounds = this->get_bbox(); 
  std::pair<cv::Point2f, cv::Point2f> tile_2_bounds = other->get_bbox(); 

  int nrows = std::min(tile_1_bounds.second.y, tile_2_bounds.second.y) - std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);
  int ncols = std::min(tile_1_bounds.second.x, tile_2_bounds.second.x) - std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
  //printf("rows = %d, cols = %d\n", nrows, ncols);
  if ((nrows <= 0) || (ncols <= 0) ) {
    return -2;
  }
  int offset_x = std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
  int offset_y = std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);
  cv::Mat transform_1 = cv::Mat::zeros(nrows, ncols, CV_8UC1);
  cv::Mat transform_2 = cv::Mat::zeros(nrows, ncols, CV_8UC1);

  // make the transformed images in the same size with the same cells in the same locations
  for (int _y = 0; _y < tile_p_image_1.rows; _y++) {
    for (int _x = 0; _x < tile_p_image_1.cols; _x++) {
      cv::Point2f p = cv::Point2f(_x, _y);
      cv::Point2f transformed_p = this->rigid_transform(p);

      int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
      int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
      if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
        transform_1.at<unsigned char>(y_c, x_c) +=
           tile_p_image_1.at<unsigned char>(_y, _x);
      }
    }
  }

  for (int _y = 0; _y < tile_p_image_2.rows; _y++) {
    for (int _x = 0; _x < tile_p_image_2.cols; _x++) {
      cv::Point2f p = cv::Point2f(_x, _y);
      cv::Point2f transformed_p = other->rigid_transform(p);

      int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
      int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
      if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
        transform_2.at<unsigned char>(y_c, x_c) +=
           tile_p_image_2.at<unsigned char>(_y, _x);
      }
    }
  }

  // clear any location which only has a value for one of them
  // note that the transforms are the same size
  for (int _y = 0; _y < transform_1.rows; _y++) {
    for (int _x = 0; _x < transform_1.cols; _x++) {
      if (transform_2.at<unsigned char>(_y, _x) == 0) {
       transform_1.at<unsigned char>(_y, _x) = 0;
      }
      else if (transform_1.at<unsigned char>(_y, _x) == 0) {
       transform_2.at<unsigned char>(_y, _x) = 0;
      }
    }
  }
  cv::Mat result_CCOEFF_NORMED;
  cv::matchTemplate(transform_1, transform_2, result_CCOEFF_NORMED, CV_TM_CCOEFF_NORMED);
  return result_CCOEFF_NORMED.at<float>(0,0);
}

/*
gets the feature vector for a pair of tiles
type == 1 gives the irst element of the size of the overlapping region followed by the mean and stddev for each tile for each of the subsections
type == 2 gives the size of the subsection followed by the pixel corralation or each subsection
*/
cv::Mat tfk::Tile::get_feature_vector(Tile *other, int boxes, int type) {
  if (type == 1) {
    cv::Mat vector = cv::Mat::zeros(1, boxes*boxes*4+1, CV_32F);
    if (!(this->overlaps_with(other))) {
      return vector;
    }


    cv::Mat tile_p_image_1;
    cv::Mat tile_p_image_2;
    tile_p_image_1 = this->get_tile_data(Resolution::FULL); //cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
    tile_p_image_2 = other->get_tile_data(Resolution::FULL); //cv::imread(other->filepath, CV_LOAD_IMAGE_UNCHANGED);


    std::pair<cv::Point2f, cv::Point2f> tile_1_bounds = this->get_bbox();
    std::pair<cv::Point2f, cv::Point2f> tile_2_bounds = other->get_bbox();

    int nrows = std::min(tile_1_bounds.second.y, tile_2_bounds.second.y) - std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);
    int ncols = std::min(tile_1_bounds.second.x, tile_2_bounds.second.x) - std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
    //printf("rows = %d, cols = %d\n", nrows, ncols);
    if ((nrows <= 0) || (ncols <= 0) ) {
      return vector;
    }
    int offset_x = std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
    int offset_y = std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);
    cv::Mat transform_1 = cv::Mat::zeros(nrows, ncols, CV_8UC1);
    cv::Mat transform_2 = cv::Mat::zeros(nrows, ncols, CV_8UC1);

    // make the transformed images in the same size with the same cells in the same locations
    for (int _y = 0; _y < tile_p_image_1.rows; _y++) {
      for (int _x = 0; _x < tile_p_image_1.cols; _x++) {
        cv::Point2f p = cv::Point2f(_x, _y);
        cv::Point2f transformed_p = this->rigid_transform(p);

        int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
        int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
        if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
          transform_1.at<unsigned char>(y_c, x_c) +=
             tile_p_image_1.at<unsigned char>(_y, _x);
        }
      }
    }

    for (int _y = 0; _y < tile_p_image_2.rows; _y++) {
      for (int _x = 0; _x < tile_p_image_2.cols; _x++) {
        cv::Point2f p = cv::Point2f(_x, _y);
        cv::Point2f transformed_p = other->rigid_transform(p);

        int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
        int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
        if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
          transform_2.at<unsigned char>(y_c, x_c) +=
             tile_p_image_2.at<unsigned char>(_y, _x);
        }
      }
    }

    // clear any location which only has a value for one of them
    // note that the transforms are the same size
    for (int _y = 0; _y < transform_1.rows; _y++) {
      for (int _x = 0; _x < transform_1.cols; _x++) {
        if (transform_2.at<unsigned char>(_y, _x) == 0) {
         transform_1.at<unsigned char>(_y, _x) = 0;
        }
        else if (transform_1.at<unsigned char>(_y, _x) == 0) {
         transform_2.at<unsigned char>(_y, _x) = 0;
        }
      }
    }

    vector.at<int>(0) = nrows*ncols;

    for (int i = 0; i < boxes; i++) {
      for (int j = 0; j < boxes; j++) {
        cv::Scalar m1;
        cv::Scalar m2;
        cv::Scalar stdv1;
        cv::Scalar stdv2;
        int start_x = i*ncols/boxes;
        int start_y = j*nrows/boxes;
        int end_x = std::min((i+1)*ncols/boxes, ncols-1);
        int end_y = std::min((j+1)*nrows/boxes, nrows-1);
        cv::Mat tmp1 = transform_1(cv::Rect(start_x,start_y,end_x-start_x,end_y-start_y));
        cv::Mat tmp2 = transform_2(cv::Rect(start_x,start_y,end_x-start_x,end_y-start_y));
        cv::meanStdDev(tmp1, m1, stdv1);
        cv::meanStdDev(tmp2, m2, stdv2);
        //std::cout << "M = "<< std::endl << " "  << tmp1 << std::endl << std::endl;
        //printf("%f, %f\n", m1[0],  stdv1[0]);
        vector.at<float>(1+4*(i*boxes+j)) = (float)m1[0];
        vector.at<float>(2+4*(i*boxes+j)) = (float)m2[0];
        vector.at<float>(3+4*(i*boxes+j)) = (float)stdv1[0];
        vector.at<float>(4+4*(i*boxes+j)) = (float)stdv2[0];


      }
    }
    return vector;
  }
  if (type == 2) {
    cv::Mat vector = cv::Mat::zeros(1, boxes*boxes+1, CV_32F);
    if (!(this->overlaps_with(other))) {
      return vector;
    }


    cv::Mat tile_p_image_1;
    cv::Mat tile_p_image_2;
    tile_p_image_1 = this->get_tile_data(Resolution::FULL); //cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
    tile_p_image_2 = other->get_tile_data(Resolution::FULL); //cv::imread(other->filepath, CV_LOAD_IMAGE_UNCHANGED);


    std::pair<cv::Point2f, cv::Point2f> tile_1_bounds = this->get_bbox();
    std::pair<cv::Point2f, cv::Point2f> tile_2_bounds = other->get_bbox();

    int nrows = std::min(tile_1_bounds.second.y, tile_2_bounds.second.y) - std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);
    int ncols = std::min(tile_1_bounds.second.x, tile_2_bounds.second.x) - std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
    //printf("rows = %d, cols = %d\n", nrows, ncols);
    if ((nrows <= 0) || (ncols <= 0) ) {
      return vector;
    }
    int offset_x = std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
    int offset_y = std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);
    cv::Mat transform_1 = cv::Mat::zeros(nrows, ncols, CV_8UC1);
    cv::Mat transform_2 = cv::Mat::zeros(nrows, ncols, CV_8UC1);

    // make the transformed images in the same size with the same cells in the same locations
    for (int _y = 0; _y < tile_p_image_1.rows; _y++) {
      for (int _x = 0; _x < tile_p_image_1.cols; _x++) {
        cv::Point2f p = cv::Point2f(_x, _y);
        cv::Point2f transformed_p = this->rigid_transform(p);

        int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
        int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
        if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
          transform_1.at<unsigned char>(y_c, x_c) +=
             tile_p_image_1.at<unsigned char>(_y, _x);
        }
      }
    }

    for (int _y = 0; _y < tile_p_image_2.rows; _y++) {
      for (int _x = 0; _x < tile_p_image_2.cols; _x++) {
        cv::Point2f p = cv::Point2f(_x, _y);
        cv::Point2f transformed_p = other->rigid_transform(p);

        int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
        int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
        if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
          transform_2.at<unsigned char>(y_c, x_c) +=
             tile_p_image_2.at<unsigned char>(_y, _x);
        }
      }
    }

    // clear any location which only has a value for one of them
    // note that the transforms are the same size
    for (int _y = 0; _y < transform_1.rows; _y++) {
      for (int _x = 0; _x < transform_1.cols; _x++) {
        if (transform_2.at<unsigned char>(_y, _x) == 0) {
         transform_1.at<unsigned char>(_y, _x) = 0;
        }
        else if (transform_1.at<unsigned char>(_y, _x) == 0) {
         transform_2.at<unsigned char>(_y, _x) = 0;
        }
      }
    }

    vector.at<float>(0) = ((float)nrows*ncols)/(boxes*boxes);

    for (int i = 0; i < boxes; i++) {
      for (int j = 0; j < boxes; j++) {
        int start_x = i*ncols/boxes;
        int start_y = j*nrows/boxes;
        int end_x = std::min((i+1)*ncols/boxes, ncols-1);
        int end_y = std::min((j+1)*nrows/boxes, nrows-1);
        cv::Mat tmp1 = transform_1(cv::Rect(start_x,start_y,end_x-start_x,end_y-start_y));
        cv::Mat tmp2 = transform_2(cv::Rect(start_x,start_y,end_x-start_x,end_y-start_y));
        cv::Mat result_CCOEFF_NORMED;
        cv::matchTemplate(tmp1, tmp2, result_CCOEFF_NORMED, CV_TM_CCOEFF_NORMED);
        vector.at<float>(1+(i*boxes+j)) = result_CCOEFF_NORMED.at<float>(0,0);
      }
    }
    return vector;
  }

  printf("Error reached end of get_feature_vector without returning anything\n");
  exit(0);
}

void tfk::Tile::release_full_image() {
    full_image_lock->lock();
    full_image.release();
    has_full_image = false;
    full_image_lock->unlock(); 
}

void tfk::Tile::release_2d_keypoints() {

    this->p_kps->clear();
    std::vector<cv::KeyPoint>().swap(*(this->p_kps));
    this->p_kps_fallback->clear();
    std::vector<cv::KeyPoint>().swap(*(this->p_kps_fallback));

    this->p_kps_desc->release();
    this->p_kps_desc_fallback->release();

    //// clean up the memory
    //for (auto del : known_set) {
    //  tile_data_t *a_tile = &(p_sec_data->tiles[del]);
    //  a_tile->p_kps->clear();
    //  std::vector<cv::KeyPoint>().swap(*(a_tile->p_kps));
    //  ((a_tile->p_kps_desc))->release();
    //}
}


std::vector<cv::Point2f> tfk::Tile::get_corners() {

  std::vector<cv::Point2f> post_corners;

  double dx = this->shape_dx;
  double dy = this->shape_dy;

  cv::Point2f corners[4];
  corners[0] = cv::Point2f(0.0,0.0);
  corners[1] = cv::Point2f(dx,0.0);
  corners[2] = cv::Point2f(0.0,dy);
  corners[3] = cv::Point2f(dx,dy);

  for (int i = 0; i < 4; i++) {
    post_corners.push_back(this->rigid_transform(corners[i]));
  }

  return post_corners;
}

// format is min_x,min_y , max_x,max_y
std::pair<cv::Point2f, cv::Point2f> tfk::Tile::get_bbox() {

  std::vector<cv::Point2f> corners = this->get_corners();
  float min_x = corners[0].x;
  float max_x = corners[0].x;
  float min_y = corners[0].y;
  float max_y = corners[0].y;
  for (int i = 1; i < corners.size(); i++) {
    if (corners[i].x < min_x) min_x = corners[i].x;
    if (corners[i].x > max_x) max_x = corners[i].x;

    if (corners[i].y < min_y) min_y = corners[i].y;
    if (corners[i].y > max_y) max_y = corners[i].y;
  }
  return std::make_pair(cv::Point2f(min_x,min_y), cv::Point2f(max_x, max_y));
}

void tfk::Tile::get_3d_keypoints(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Mat>& desc) {
  if (this->p_kps_3d->size() <= 0) return;

  for (int pt_idx = 0; pt_idx < this->p_kps_3d->size(); ++pt_idx) {
    //if (this->ignore[pt_idx]) continue;
    cv::Point2f pt = this->rigid_transform((*(this->p_kps_3d))[pt_idx].pt);
    cv::KeyPoint kpt = (*(this->p_kps_3d))[pt_idx];
    kpt.pt = pt;
    keypoints.push_back(kpt);
    desc.push_back(this->p_kps_desc_3d->row(pt_idx).clone());
  }
}


void tfk::Tile::write_wafer(FILE* wafer_file, int section_id, int base_section) {
  fprintf(wafer_file, "\t\t\"bbox\": [\n");

  fprintf(wafer_file,
      "\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f\n],",
      this->x_start+this->offset_x, (this->x_finish+this->offset_x),
      this->y_start+this->offset_y, (this->y_finish+this->offset_y));
  fprintf(wafer_file, "\t\t\"height\": %d,\n",SIFT_D1_SHIFT_3D);
  fprintf(wafer_file, "\t\t\"layer\": %d,\n",section_id + base_section+1);
  fprintf(wafer_file, "\t\t\"maxIntensity\": %f,\n",255.0);
  fprintf(wafer_file, "\t\t\"mfov\": %d,\n",
      this->mfov_id);
  fprintf(wafer_file, "\t\t\"minIntensity\": %f,\n",
      0.0);
  fprintf(wafer_file, "\t\t\"mipmapLevels\": {\n");
  fprintf(wafer_file, "\t\t\"0\": {\n");
  fprintf(wafer_file, "\t\t\t\"imageUrl\": \"%s\"\n", this->filepath.c_str());
  fprintf(wafer_file, "\t\t\t}\n");
  fprintf(wafer_file, "\t\t},\n");
  fprintf(wafer_file, "\t\t\"tile_index\": %d,\n",
      this->index);
  fprintf(wafer_file, "\t\t\"transforms\": [\n");
  // {'className': 'mpicbg.trakem2.transform.AffineModel2D', 'dataString': '0.1 0.0 0.0 0.1 0.0 0.0'}

  fprintf(wafer_file, "\t\t\t{\n");
  fprintf(wafer_file,
      "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.AffineModel2D\",\n");
  fprintf(wafer_file,
      "\t\t\t\t\"dataString\": \"%f %f %f %f %f %f\"\n", this->a00,
      this->a10, this->a01, this->a11, this->x_start+this->offset_x, this->y_start+this->offset_y);
  //#ifdef ALIGN3D
  //if (true) {
  //#else
  //if (false) {
  //#endif 
  //fprintf(wafer_file,
  //    "\t\t\t},\n");

  //fprintf(wafer_file, "\t\t\t{\n");
  //fprintf(wafer_file,
  //    "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.PointsTransformModel\",\n");
  //fprintf(wafer_file,
  //    "\t\t\t\t\"dataString\": \"%s\"\n", get_point_transform_string(graph, vd).c_str());
  //fprintf(wafer_file,
  //    "\t\t\t}\n");
  //} else {
   fprintf(wafer_file,
      "\t\t\t}\n");
  //}

  fprintf(wafer_file,
      "\t\t],\n");
  fprintf(wafer_file,
      "\t\t\"width\":%d\n",SIFT_D2_SHIFT_3D);
}

void tfk::Tile::local2DAlignUpdateLimited(std::set<Tile*>* active_set) {
  //if (this->bad_2d_alignment) return;

  if (active_set->find(this) == active_set->end()) return;

  //std::vector<edata>& edges = graph->edgeData[vid];
  double global_learning_rate = 0.4;
  if (this->edges.size() == 0) return;
  if (this->edges.size() == 0) return;

  double learning_rate = global_learning_rate;
  double grad_error_x = 0.0;
  double grad_error_y = 0.0;
  double weight_sum = 1.0;

  for (int i = 0; i < this->edges.size(); i++) {
    std::vector<cv::Point2f>* v_points = this->edges[i].v_points;
    std::vector<cv::Point2f>* n_points = this->edges[i].n_points;
    //vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
    Tile* neighbor = (Tile*) this->edges[i].neighbor_tile;
    if (active_set->find(neighbor) == active_set->end()) continue;

    double curr_weight = 1.0/v_points->size();

    for (int j = 0; j < v_points->size(); j++) {
      cv::Point2f ptx1 = this->rigid_transform((*v_points)[j]); //transform_point(vertex_data, (*v_points)[j]);
      cv::Point2f ptx2 = neighbor->rigid_transform((*n_points)[j]);//transform_point(neighbor_vertex, (*n_points)[j]);

      double delta_x = ptx2.x - ptx1.x;
      double delta_y = ptx2.y - ptx1.y;
      grad_error_x += 2 * delta_x * curr_weight;
      grad_error_y += 2 * delta_y * curr_weight;
      weight_sum += curr_weight;
    }
  }

  //printf("gradient %f %f\n", grad_error_x, grad_error_y);

  // update the gradients.
  if (weight_sum > 0) {
    this->offset_x += grad_error_x*learning_rate/(weight_sum);
    this->offset_y += grad_error_y*learning_rate/(weight_sum);
  }
}

void tfk::Tile::local2DAlignUpdate() {
  if (this->bad_2d_alignment) return;
  //std::vector<edata>& edges = graph->edgeData[vid];
  double global_learning_rate = 0.4;
  if (this->edges.size() == 0) return;
  if (this->edges.size() == 0) return;

  double learning_rate = global_learning_rate;
  double grad_error_x = 0.0;
  double grad_error_y = 0.0;
  double weight_sum = 1.0;

  for (int i = 0; i < this->edges.size(); i++) {
    std::vector<cv::Point2f>* v_points = this->edges[i].v_points;
    std::vector<cv::Point2f>* n_points = this->edges[i].n_points;
    //vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
    Tile* neighbor = (Tile*) this->edges[i].neighbor_tile;
    if (neighbor->bad_2d_alignment) continue;
    if (this->ideal_offsets.find(neighbor->tile_id) == this->ideal_offsets.end() &&
        neighbor->ideal_offsets.find(this->tile_id) == neighbor->ideal_offsets.end()) continue;
    


      cv::Point2f a_point = cv::Point2f(this->x_start+this->offset_x,
                                        this->y_start+this->offset_y);
      cv::Point2f b_point = cv::Point2f(neighbor->x_start+neighbor->offset_x,
                                        neighbor->y_start+neighbor->offset_y);
      cv::Point2f delta = a_point-b_point;
                              // c_a_point - c_b_point - a_point + b_point
                              // c_a_point - a_point + b_point-c_b_point
      if (v_points->size() == 0 && n_points->size() == 0) continue; 
      cv::Point2f deviation;
      if (this->ideal_offsets.find(neighbor->tile_id) != this->ideal_offsets.end()) {
        deviation = this->ideal_offsets[neighbor->tile_id] - delta;
      } else {
        //          c_b_point - c_a_point - b_point + a_point
        //          a_point - b_point + c_b_point - c_a_point
        //          b_point - a_point + c_a_point - c_b_point


        //            c_b_point - c_a_point + b_point - a_point
        //            c_a_point - c_b_point + b_point - a_point
        //            c_a_point - a_point + b_point - c_b_point
        
        //            - c_b_point + c_a_point - a_point + b_point
        deviation = -1*neighbor->ideal_offsets[this->tile_id] - delta;
      }

      weight_sum += 1;
      if (std::abs(deviation.x) > 2.0 || true) {
        grad_error_x += 2*deviation.x;
      }
      if (std::abs(deviation.y) > 2.0 || true ) {
        grad_error_y += 2*deviation.y;
      }
      continue;
    //double curr_weight = 1.0/v_points->size();

    //for (int j = 0; j < v_points->size(); j++) {
    //  cv::Point2f ptx1 = this->rigid_transform((*v_points)[j]); //transform_point(vertex_data, (*v_points)[j]);
    //  cv::Point2f ptx2 = neighbor->rigid_transform((*n_points)[j]);//transform_point(neighbor_vertex, (*n_points)[j]);

    //  double delta_x = ptx2.x - ptx1.x;
    //  double delta_y = ptx2.y - ptx1.y;
    //  grad_error_x += 2 * delta_x * curr_weight;
    //  grad_error_y += 2 * delta_y * curr_weight;
    //  weight_sum += curr_weight;
    //}
  }

  //printf("gradient %f %f\n", grad_error_x, grad_error_y);

  // update the gradients.
  this->offset_x += grad_error_x*learning_rate/(weight_sum);
  this->offset_y += grad_error_y*learning_rate/(weight_sum);
}


void tfk::Tile::make_symmetric(int phase, std::vector<Tile*>& tile_list) {
  if (phase == 0) {
    if (this->bad_2d_alignment) return;
    for (int i = 0; i < tile_list.size(); i++) {
      Tile* other = tile_list[i];
      if (other->bad_2d_alignment) continue;
      for (int j = 0; j < other->edges.size(); j++) {
        edata edge = other->edges[j];
        if (edge.neighbor_id == this->tile_id) {
          edata edge2;
          edge2.v_points = edge.n_points;
          edge2.n_points = edge.v_points;
          edge2.neighbor_id = other->tile_id;
          edge2.neighbor_tile = other;
          edge2.weight = 1.0;
          //printf("adding symmetricn edge %d %d\n", this->tile_id, other->tile_id);
          this->add_edges.push_back(edge2);
        }
      }
    }
  } else if (phase == 1) {
    if (this->bad_2d_alignment) return;
    for (int i = 0; i < this->add_edges.size(); i++) {
      this->edges.push_back(this->add_edges[i]);
    }
  }
}


void tfk::Tile::insert_matches(Tile* neighbor, std::vector<cv::Point2f>& points_a, std::vector<cv::Point2f>& points_b) {
  std::vector<cv::Point2f>* vedges = new std::vector<cv::Point2f>();
  std::vector<cv::Point2f>* nedges = new std::vector<cv::Point2f>();

  for (int i = 0; i < points_a.size(); i++) {
    vedges->push_back(cv::Point2f(points_a[i]));
  }

  for (int i = 0; i < points_b.size(); i++) {
    nedges->push_back(cv::Point2f(points_b[i]));
  }

  edata edge1;
  edge1.v_points = vedges;
  edge1.n_points = nedges;
  edge1.neighbor_id = neighbor->tile_id;
  edge1.neighbor_tile = (void*) neighbor;
  edge1.weight = 1.0;

  this->edges.push_back(edge1);



  //this->insertEdge(atile_id, edge1);
  // make bi-directional later.
  //edata edge2;
  //edge2.v_points = nedges;
  //edge2.n_points = vedges;
  //edge2.neighbor_id = atile_id;
  //edge2.weight = weight;
  //this->insertEdge(btile_id, edge2);
}

tfk::Tile::Tile(int section_id, int tile_id, int index, std::string filepath,
    int x_start, int x_finish, int y_start, int y_finish) {
  this->section_id = section_id;
  this->tile_id = tile_id;
  this->index = index;
  this->filepath = filepath;
  this->x_start = x_start;
  this->x_finish = x_finish;
  this->y_start = y_start;
  this->y_finish = y_finish;
  this->offset_x = 0.0;
  this->offset_y = 0.0;
  this->image_data_replaced = false;
}


cv::Point2f tfk::Tile::rigid_transform(cv::Point2f pt) {
  cv::Point2f pt2 = cv::Point2f(pt.x+this->offset_x+this->x_start, pt.y+this->offset_y+this->y_start);
  return pt2;
}

bool tfk::Tile::overlaps_with(std::pair<cv::Point2f, cv::Point2f> bbox) {
    int x1_start = this->x_start;
    int x1_finish = this->x_finish;
    int y1_start = this->y_start;
    int y1_finish = this->y_finish;

    int x2_start = bbox.first.x;
    int x2_finish = bbox.second.x;
    int y2_start = bbox.first.y;
    int y2_finish = bbox.second.y;

    bool res = false;
    if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
        (y1_start < y2_finish) && (y1_finish > y2_start)) {
        res = true;
    }
    return res;
}


bool tfk::Tile::overlaps_with(Tile* other) {
    int x1_start = this->x_start;
    int x1_finish = this->x_finish;
    int y1_start = this->y_start;
    int y1_finish = this->y_finish;

    int x2_start = other->x_start;
    int x2_finish = other->x_finish;
    int y2_start = other->y_start;
    int y2_finish = other->y_finish;

    bool res = false;
    if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
        (y1_start < y2_finish) && (y1_finish > y2_start)) {
        res = true;
    }
    return res;
}

tfk::Tile::Tile(TileData& tile_data) {
    //tile_data_t *p_cur_tile = &(p_sec_data->tiles[p_sec_data->n_tiles]);
    //p_sec_data->n_tiles++;
    this->has_full_image = false;
    this->full_image_lock = new std::mutex();
    this->image_data_replaced = false;
    this->shape_dx = tile_data.x_finish() - tile_data.x_start();
    this->shape_dy = tile_data.y_finish() - tile_data.y_start();
    this->bad_2d_alignment = false;
    this->section_id = tile_data.section_id();
    this->mfov_id = tile_data.tile_mfov();
    this->index = tile_data.tile_index();
    this->x_start = tile_data.x_start();
    this->x_finish = tile_data.x_finish();
    this->y_start = tile_data.y_start();
    this->y_finish = tile_data.y_finish();
    this->offset_x = 0.0;
    this->offset_y = 0.0;
    this->filepath = tile_data.tile_filepath();
    this->p_image = new cv::Mat();
    this->p_kps = new std::vector<cv::KeyPoint>();
    this->p_kps_desc = new cv::Mat();
    this->p_kps_fallback = new std::vector<cv::KeyPoint>();
    this->p_kps_desc_fallback = new cv::Mat();

    this->p_kps_3d = new std::vector<cv::KeyPoint>();
    this->ignore = NULL;
    this->p_kps_desc_3d = new cv::Mat();
    this->level = 0;
    this->bad = false;

    this->a00 = 1.0;
    this->a10 = 0.0;
    this->a01 = 0.0;
    this->a11 = 1.0;

    this->tile_data = tile_data;


    if (tile_data.has_ignore()) {
      bool *ignore = (bool *) malloc(sizeof(bool));
      *ignore = tile_data.ignore();
      this->ignore = ignore;
    }
    if (tile_data.has_a00()) {
      this->a00 = tile_data.a00();
    }
    if (tile_data.has_a10()) {
      this->a10 = tile_data.a10();
    }
    if (tile_data.has_a01()) {
      this->a10 = tile_data.a01();
    }
    if (tile_data.has_a11()) {
      this->a11 = tile_data.a11();
    }
    if (tile_data.has_level()) {
      this->level = tile_data.level();
    }
    if (tile_data.has_bad()) {
      this->bad = tile_data.bad();
    }
}

void tfk::Tile::recompute_3d_keypoints(std::vector<cv::KeyPoint>& atile_all_kps,
                                       std::vector<cv::Mat>& atile_all_kps_desc,
                                       tfk::params sift_parameters) {

  cv::Mat local_p_image;
  local_p_image.create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  local_p_image = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);


  //this->p_kps_3d = new std::vector<cv::KeyPoint>();

  int rows = local_p_image.rows;
  int cols = local_p_image.cols;

  ASSERT((rows % SIFT_D1_SHIFT_3D) == 0);
  ASSERT((cols % SIFT_D2_SHIFT_3D) == 0);

  std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
  cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];
  int n_sub_images;


  cv::Ptr<cv::Feature2D> p_sift;
  p_sift = new cv::xfeatures2d::SIFT_Impl(
            sift_parameters.num_features,  // num_features --- unsupported.
            sift_parameters.num_octaves,  // number of octaves
            sift_parameters.contrast_threshold,  // contrast threshold.
            sift_parameters.edge_threshold,  // edge threshold.
            sift_parameters.sigma);  // sigma.

    int max_rows = rows / SIFT_D1_SHIFT_3D;
    int max_cols = cols / SIFT_D2_SHIFT_3D;
    n_sub_images = max_rows * max_cols;
        cv::Mat sub_im_mask = cv::Mat::ones(0,0,
            CV_8UC1);
        int sub_im_id = 0;
        // Detect the SIFT features within the subimage.
        fasttime_t tstart = gettime();
        p_sift->detectAndCompute((local_p_image), sub_im_mask, v_kps[sub_im_id],
            m_kps_desc[sub_im_id], false);

        fasttime_t tend = gettime();
        totalTime += tdiff(tstart, tend);
    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    int point_count_3d = 0;
    for (int _i = 0; _i < n_sub_images; _i++) {
        for (int _j = 0; _j < v_kps[_i].size(); _j++) {
            cv::Point2f pt = this->rigid_transform(v_kps[_i][_j].pt);
            cv::KeyPoint kpt = v_kps[_i][_j];
            kpt.pt = pt;
            atile_all_kps.push_back(kpt);
            atile_all_kps_desc.push_back(m_kps_desc[_i].row(_j).clone());
            point_count_3d++;
        }
    }

  //cv::Mat m_kps_desc_filtered = m_kps_desc[0].clone();
  //*(this)->p_kps_desc_3d = m_kps_desc[0].clone();

  //atile_all_kps_desc.push_back(m_kps_desc[0].clone());

  //printf("Number of 3d points is %d\n", point_count_3d);
  local_p_image.release();
}


void tfk::Tile::compute_sift_keypoints3d(bool recomputation) {
  //(*this->p_image).create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  //(*this->p_image) = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);

  cv::Mat tmp_image;
  tmp_image.create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  tmp_image = this->get_tile_data(Resolution::FULL); //cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);

  //(*this->p_image).create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);

  float scale_x = 1.0/8;
  float scale_y = 1.0/8;

  //float scale_x = 1.0;
  //float scale_y = 1.0;
  cv::resize(tmp_image, (*this->p_image), cv::Size(), scale_x,scale_y,CV_INTER_AREA);


  this->p_kps_3d = new std::vector<cv::KeyPoint>();

  //int rows = this->p_image->rows;
  //int cols = this->p_image->cols;
  //ASSERT((rows % SIFT_D1_SHIFT_3D) == 0);
  //ASSERT((cols % SIFT_D2_SHIFT_3D) == 0);
  cv::Ptr<cv::Feature2D> p_sift;
  std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
  cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];
  int n_sub_images;


  if (true || !recomputation) {
  // NOTE(TFK): I need to check these parameters against the prefix_ cached ones.
  p_sift = new cv::xfeatures2d::SIFT_Impl(
            1,  // num_features --- unsupported.
            6,  // number of octaves
            //24,  // number of octaves
            CONTRAST_THRESH_3D,  // contrast threshold.
            EDGE_THRESH_3D,  // edge threshold.
            1.6*2);  // sigma.
 } else {

  p_sift = new cv::xfeatures2d::SIFT_Impl(
            16,  // num_features --- unsupported.
            6,  // number of octaves
            CONTRAST_THRESH_3D,  // contrast threshold.
            EDGE_THRESH_3D,  // edge threshold.
            1.6);  // sigma.

 }

    //int max_rows = rows / SIFT_D1_SHIFT_3D;
    //int max_cols = cols / SIFT_D2_SHIFT_3D;
    int max_rows = 1;
    int max_cols = 1;
    n_sub_images = max_rows * max_cols;
        cv::Mat sub_im_mask = cv::Mat::ones(0,0,
            CV_8UC1);
        int sub_im_id = 0;
        // Detect the SIFT features within the subimage.
        fasttime_t tstart = gettime();
        p_sift->detectAndCompute((*this->p_image), sub_im_mask, v_kps[sub_im_id],
            m_kps_desc[sub_im_id], false);

        fasttime_t tend = gettime();
        totalTime += tdiff(tstart, tend);
    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    int point_count_3d = 0;
    for (int _i = 0; _i < n_sub_images; _i++) {
        for (int _j = 0; _j < v_kps[_i].size(); _j++) {
            v_kps[_i][_j].pt.x /= scale_x;
            v_kps[_i][_j].pt.y /= scale_y;
            (*this->p_kps_3d).push_back(v_kps[_i][_j]);
            point_count_3d++;
        }
    }

  //cv::Mat m_kps_desc_filtered = m_kps_desc[0].clone();
  *(this)->p_kps_desc_3d = m_kps_desc[0].clone();

  //printf("Number of 3d points is %d\n", point_count_3d);
  this->p_image->release();

}

cv::Mat tfk::Tile::get_tile_data(Resolution res) {

  std::string thumbnailpath = std::string(this->filepath);
  thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
  thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");

  switch(res) {

    case THUMBNAIL: {
      cv::Mat tmp = this->get_tile_data(Resolution::FULL);//cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      cv::Mat ret;
      cv::resize(tmp, ret, cv::Size(), 0.1,0.1,CV_INTER_AREA);
      //cv::Mat src = cv::imread(thumbnailpath, CV_LOAD_IMAGE_GRAYSCALE);
      //cv::Mat dst;
      ////cv::equalizeHist( src, dst );
      
      //int scale = 1;
      //cv::Laplacian(src, dst, CV_8U, 3, scale, 0, cv::BORDER_DEFAULT);
      //return src;
      return ret;
      //return dst;//cv::imread(thumbnailpath, CV_LOAD_IMAGE_GRAYSCALE);
      break;
    }
    case THUMBNAIL2: {
      cv::Mat src = cv::imread(thumbnailpath, CV_LOAD_IMAGE_GRAYSCALE);
      return src;
      //cv::Mat dst;
      ////cv::equalizeHist( src, dst );
      //int scale = 1;
      //cv::Laplacian(src, dst, CV_8U, 3, scale, 0, cv::BORDER_DEFAULT);

      //return dst;//cv::imread(thumbnailpath, CV_LOAD_IMAGE_GRAYSCALE);
      break;
    }

    case FILEIOTEST: {
      std::vector<int> params;
      params.push_back(CV_IMWRITE_JPEG_QUALITY);
      params.push_back(70);
      //std::string new_path = this->filepath + "_.jpg";
      std::string new_path;
      //if (this->tile_id % 3 == 0) {
      //  new_path = this->filepath.replace(0,5, "/ebs/");
      //} else if (this->tile_id % 3 == 1) {
      //  new_path = this->filepath.replace(0,5, "/ebs2/");
      //} else if (this->tile_id % 3 == 2) {
        new_path = this->filepath.replace(0,5, "/efs/");
      //}
      new_path = this->filepath + "_.jpg";
      printf("New %s\n", new_path.c_str());

      cv::Mat full_image = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      cv::imwrite(new_path, full_image, params);

      //cv::Mat full_image = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);
      //cv::Mat full_image = cv::imread(this->filepath, CV_LOAD_IMAGE_GRAYSCALE);
      return full_image;
      //return cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      break;
    }

    case FULL: {
      full_image_lock->lock();
      if (/*true || */!has_full_image) {
      //  std::string new_path = this->filepath.replace(0,5,"/ebs/");
      std::string new_path;
      //if (this->tile_id % 3 == 0) {
      //  new_path = this->filepath.replace(0,5, "/ebs/");
      //} else if (this->tile_id % 3 == 1) {
      //  new_path = this->filepath.replace(0,5, "/ebs2/");
      //} else if (this->tile_id % 3 == 2) {
        new_path = this->filepath.replace(0,5, "/efs/");
      //}
      new_path = this->filepath + "_.jpg";

        //printf("%s\n", path.c_str());
        //std::string new_path = this->filepath + "_.jpg";
        //printf("%s\n", new_path.c_str());
        full_image = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);
        //printf("image rows %d and cols %d\n", full_image.rows, full_image.cols);
        //full_image = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
        //printf("%s\n", new_path.c_str());
        //cv::imwrite(new_path,full_image);
        //full_image = cv::imread(path, CV_LOAD_IMAGE_UNCHANGED);
        has_full_image = true; //uncomment for cashing
      }
      cv::Mat ret = full_image.clone();
      full_image_lock->unlock();
      //full_image.release(); // remove for caching
      //printf("image rows %d and cols %d\n", ret.rows, ret.cols);
      return ret;
      //return cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      break;
    }
    case PERCENT30: {
      cv::Mat tmp = this->get_tile_data(Resolution::FULL);//cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      cv::Mat ret;
      cv::resize(tmp, ret, cv::Size(), 0.3,0.3,CV_INTER_AREA);
      //tmp.release();
      return ret;
      break;
    }
    default: {
      printf("Error in get_tile_data, invalid resolution specified, got %d\n", res);
      exit(1);
      return cv::Mat();
    }
  }
}

void tfk::Tile::compute_sift_keypoints2d_params(tfk::params params,
    std::vector<cv::KeyPoint>& local_keypoints, cv::Mat& local_desc) {
    compute_sift_keypoints2d_params(params, local_keypoints, local_desc, this);
}

void tfk::Tile::compute_sift_keypoints2d_params(tfk::params params,
std::vector<cv::KeyPoint>& local_keypoints, cv::Mat& local_desc, Tile* other_tile) {
  //printf("computing sift keypoints 2d with params\n");

  cv::Mat _tmp_image = this->get_tile_data(FULL);
  //_tmp_image.create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  //_tmp_image = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);

  cv::Mat local_p_image;

  int my_x_start = this->x_start;
  int my_x_finish = this->x_finish;

  int other_x_start = other_tile->x_start - 50;
  int other_x_finish = other_tile->x_finish + 50;

  while (my_x_start < other_x_start) {
    my_x_start += 1;
  }
  while (my_x_finish > other_x_finish) {
    my_x_finish -= 1;
  }

  int my_y_start = this->y_start;
  int my_y_finish = this->y_finish;

  int other_y_start = other_tile->y_start - 100;
  int other_y_finish = other_tile->y_finish + 100;

  while (my_y_start < other_y_start) {
    my_y_start += 1;
  }
  while (my_y_finish > other_y_finish) {
    my_y_finish -= 1;
  }

  int new_x_start = my_x_start - ((int)this->x_start);
  int new_x_finish = ((int)this->x_finish) - my_x_finish;

  int new_y_start = my_y_start - ((int)this->y_start);
  int new_y_finish = ((int)this->y_finish) - my_y_finish;

  if (new_x_start < 0 || new_y_start < 0) printf("The starts are less than zero!\n");
  if (_tmp_image.cols-new_x_finish < 0 || _tmp_image.rows - new_y_finish < 0) printf("The ends are less than zero!\n");
  if (_tmp_image.cols-new_x_finish > _tmp_image.cols || _tmp_image.rows - new_y_finish > _tmp_image.rows) printf("The ends are greater than dims!\n");

  cv::Mat tmp_image = _tmp_image(cv::Rect(new_x_start, new_y_start,
                                          _tmp_image.cols - new_x_finish-new_x_start,
                                          _tmp_image.rows - new_y_finish-new_y_start));

  float scale_x = params.scale_x;
  float scale_y = params.scale_y;
  cv::resize(tmp_image, local_p_image, cv::Size(), scale_x,scale_y,CV_INTER_AREA);
  _tmp_image.release();

  int rows = local_p_image.rows;
  int cols = local_p_image.cols;

  cv::Ptr<cv::Feature2D> p_sift;

  std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
  cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];

  int n_sub_images = 1;
  if ((this->tile_id > MFOV_BOUNDARY_THRESH)) {
    p_sift = new cv::xfeatures2d::SIFT_Impl(
            params.num_features,  // num_features --- unsupported.
            params.num_octaves,  // number of octaves
            params.contrast_threshold,  // contrast threshold.
            params.edge_threshold,  // edge threshold.
            params.sigma);  // sigma.

    // THEN: This tile is on the boundary, we need to compute SIFT features
    // on the entire section.
    //int max_rows = rows / SIFT_D1_SHIFT;
    //int max_cols = cols / SIFT_D2_SHIFT;
    int max_rows = 1;
    int max_cols = 1;
    n_sub_images = max_rows * max_cols;

    cilk_for (int cur_d1 = 0; cur_d1 < rows; cur_d1 += SIFT_D1_SHIFT) {
      cilk_for (int cur_d2 = 0; cur_d2 < cols; cur_d2 += SIFT_D2_SHIFT) {
        //printf("cur_d2 is %d cur_d1 is %d\n", cur_d2, cur_d1);
        // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
        cv::Mat sub_im = local_p_image(cv::Rect(cur_d2, cur_d1,
            cols, rows));

        // Mask for subimage
        cv::Mat sum_im_mask = cv::Mat::ones(rows, cols,
            CV_8UC1);

        // Compute a subimage ID, refering to a tile within larger
        //   2d image.
        //int cur_d1_id = 0;//cur_d1 / SIFT_D1_SHIFT;
        //int cur_d2_id = 0;//cur_d2 / SIFT_D2_SHIFT;
        int sub_im_id = 0;//cur_d1_id * max_cols + cur_d2_id;

        // Detect the SIFT features within the subimage.
        fasttime_t tstart = gettime();
        p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
            m_kps_desc[sub_im_id]);

        fasttime_t tend = gettime();
        totalTime += tdiff(tstart, tend);

        for (size_t i = 0; i < v_kps[sub_im_id].size(); i++) {
          v_kps[sub_im_id][i].pt.x += cur_d2;
          v_kps[sub_im_id][i].pt.y += cur_d1;
        }
      }
    }
    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    for (int i = 0; i < n_sub_images; i++) {
        for (int j = 0; j < v_kps[i].size(); j++) {
            v_kps[i][j].pt.x /= scale_x;
            v_kps[i][j].pt.y /= scale_y;
            v_kps[i][j].pt.x += new_x_start;
            v_kps[i][j].pt.y += new_y_start;
            local_keypoints.push_back(v_kps[i][j]);
        }
    }

  } else {
    printf("Assert false becasue this is unsupported code path.\n");
    assert(false);
  }

  cv::vconcat(m_kps_desc, n_sub_images, local_desc);
  local_p_image.release();
}


void tfk::Tile::compute_sift_keypoints2d() {
  //printf("computing sift keypoints 2d\n");

  cv::Mat tmp_image;
  tmp_image.create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  tmp_image = this->get_tile_data(Resolution::FULL);//cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);

  //(*this->p_image).create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);

  float scale_x = 0.2;
  float scale_y = 0.2;
  cv::resize(tmp_image, (*this->p_image), cv::Size(), scale_x,scale_y,CV_INTER_AREA);

  //(*this->p_image) = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);



  //printf("Dimensiosn are %d %d\n", this->p_image->rows, this->p_image->cols);


  int rows = this->p_image->rows;
  int cols = this->p_image->cols;

  //ASSERT((rows % SIFT_D1_SHIFT) == 0);
  //ASSERT((cols % SIFT_D2_SHIFT) == 0);

  cv::Ptr<cv::Feature2D> p_sift;

  std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
  cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];

  int n_sub_images;
  if ((this->tile_id > MFOV_BOUNDARY_THRESH)) {
    //p_sift = new cv::xfeatures2d::SIFT_Impl(
    //        1,  // num_features --- unsupported.
    //        6,  // number of octaves
    //        0.01,  // contrast threshold.
    //        10,  // edge threshold.
    //        1.2);  // sigma.

    p_sift = new cv::xfeatures2d::SIFT_Impl(
            2,  // num_features --- unsupported.
            2,  // number of octaves
            //0.04,  // contrast threshold.
            0.02,  // contrast threshold.
            5,  // edge threshold.
            1.2);  // sigma.

    // THEN: This tile is on the boundary, we need to compute SIFT features
    // on the entire section.
    //int max_rows = rows / SIFT_D1_SHIFT;
    //int max_cols = cols / SIFT_D2_SHIFT;
    int max_rows = 1;
    int max_cols = 1;
    n_sub_images = max_rows * max_cols;

    cilk_for (int cur_d1 = 0; cur_d1 < rows; cur_d1 += SIFT_D1_SHIFT) {
      cilk_for (int cur_d2 = 0; cur_d2 < cols; cur_d2 += SIFT_D2_SHIFT) {
        //printf("cur_d2 is %d cur_d1 is %d\n", cur_d2, cur_d1);
        // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
        cv::Mat sub_im = (*this->p_image)(cv::Rect(cur_d2, cur_d1,
            cols, rows));

        // Mask for subimage
        cv::Mat sum_im_mask = cv::Mat::ones(rows, cols,
            CV_8UC1);

        // Compute a subimage ID, refering to a tile within larger
        //   2d image.
        //int cur_d1_id = 0;//cur_d1 / SIFT_D1_SHIFT;
        //int cur_d2_id = 0;//cur_d2 / SIFT_D2_SHIFT;
        int sub_im_id = 0;//cur_d1_id * max_cols + cur_d2_id;

        // Detect the SIFT features within the subimage.
        fasttime_t tstart = gettime();
        p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
            m_kps_desc[sub_im_id]);

        fasttime_t tend = gettime();
        totalTime += tdiff(tstart, tend);

        for (size_t i = 0; i < v_kps[sub_im_id].size(); i++) {
          v_kps[sub_im_id][i].pt.x += cur_d2;
          v_kps[sub_im_id][i].pt.y += cur_d1;
        }
      }
    }

    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    for (int i = 0; i < n_sub_images; i++) {
        for (int j = 0; j < v_kps[i].size(); j++) {
            v_kps[i][j].pt.x /= scale_x;
            v_kps[i][j].pt.y /= scale_y;
            (*this->p_kps).push_back(v_kps[i][j]);
        }
    }

  } else {
    p_sift = new cv::xfeatures2d::SIFT_Impl(
            4,  // num_features --- unsupported.
            6,  // number of octaves
            CONTRAST_THRESH,  // contrast threshold.
            EDGE_THRESH_2D,  // edge threshold.
            1.6);  // sigma.

    // ELSE THEN: This tile is in the interior of the MFOV. Only need to
    //     compute features along the boundary.
    n_sub_images = 4;

    // BEGIN TOP SLICE
    {
      // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
      cv::Mat sub_im = (*this->p_image)(cv::Rect(
          0, 0, SIFT_D2_SHIFT_3D, OVERLAP_2D));

      // Mask for subimage
      cv::Mat sum_im_mask = cv::Mat::ones(OVERLAP_2D, SIFT_D2_SHIFT_3D, CV_8UC1);
      int sub_im_id = 0;

      // Detect the SIFT features within the subimage.
      fasttime_t tstart = gettime();
      p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
          m_kps_desc[sub_im_id]);
      fasttime_t tend = gettime();
      totalTime += tdiff(tstart, tend);
      for (size_t i = 0; i < v_kps[sub_im_id].size(); i++) {
          v_kps[sub_im_id][i].pt.x += 0;  // cur_d2;
          v_kps[sub_im_id][i].pt.y += 0;  // cur_d1;
      }
    }
    // END TOP SLICE

    // BEGIN LEFT SLICE
    {
      // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
      cv::Mat sub_im = (*this->p_image)(cv::Rect(
          0, OVERLAP_2D, OVERLAP_2D, SIFT_D1_SHIFT_3D-OVERLAP_2D));

      // Mask for subimage
      cv::Mat sum_im_mask = cv::Mat::ones(SIFT_D1_SHIFT_3D-OVERLAP_2D, OVERLAP_2D,
          CV_8UC1);
      int sub_im_id = 1;
      // Detect the SIFT features within the subimage.
      fasttime_t tstart = gettime();
      p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
         m_kps_desc[sub_im_id]);
      fasttime_t tend = gettime();
      totalTime += tdiff(tstart, tend);
      for (int i = 0; i < v_kps[sub_im_id].size(); i++) {
          v_kps[sub_im_id][i].pt.x += 0;  // cur_d2;
          v_kps[sub_im_id][i].pt.y += OVERLAP_2D;  // cur_d1;
      }
    }
    // END LEFT SLICE

    // BEGIN RIGHT SLICE
    {
      // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
      cv::Mat sub_im = (*this->p_image)(cv::Rect(
          SIFT_D2_SHIFT_3D-OVERLAP_2D, OVERLAP_2D, OVERLAP_2D, SIFT_D1_SHIFT_3D-OVERLAP_2D));

      // Mask for subimage
      cv::Mat sum_im_mask = cv::Mat::ones(SIFT_D1_SHIFT_3D-OVERLAP_2D, OVERLAP_2D,
          CV_8UC1);
      int sub_im_id = 2;
      // Detect the SIFT features within the subimage.
      fasttime_t tstart = gettime();

      p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
          m_kps_desc[sub_im_id]);
      fasttime_t tend = gettime();
      totalTime += tdiff(tstart, tend);
      for (int i = 0; i < v_kps[sub_im_id].size(); i++) {
          v_kps[sub_im_id][i].pt.x += SIFT_D2_SHIFT_3D-OVERLAP_2D;  // cur_d2;
          v_kps[sub_im_id][i].pt.y += OVERLAP_2D;  // cur_d1;
      }
    }
    // END RIGHT SLICE

    // BEGIN BOTTOM SLICE
    {
      // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
      cv::Mat sub_im = (*this->p_image)(cv::Rect(
          OVERLAP_2D, SIFT_D1_SHIFT_3D-OVERLAP_2D, SIFT_D2_SHIFT_3D-OVERLAP_2D, OVERLAP_2D));

      // Mask for subimage
      cv::Mat sum_im_mask = cv::Mat::ones(OVERLAP_2D, SIFT_D2_SHIFT_3D-OVERLAP_2D,
          CV_8UC1);
      // Compute a subimage ID, refering to a tile within larger
      //   2d image.
      int sub_im_id = 3;
      // Detect the SIFT features within the subimage.
      fasttime_t tstart = gettime();

      p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
          m_kps_desc[sub_im_id]);
      fasttime_t tend = gettime();
      totalTime += tdiff(tstart, tend);
      for (int i = 0; i < v_kps[sub_im_id].size(); i++) {
          v_kps[sub_im_id][i].pt.x += OVERLAP_2D;  // cur_d2;
          v_kps[sub_im_id][i].pt.y += (SIFT_D1_SHIFT_3D-OVERLAP_2D);  // cur_d1;
      }
    }
    // END BOTTOM SLICE

    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    for (int i = 0; i < n_sub_images; i++) {
        for (size_t j = 0; j < v_kps[i].size(); j++) {
            (*this->p_kps).push_back(v_kps[i][j]);
        }
    }
  }

  cv::vconcat(m_kps_desc, n_sub_images, *(this->p_kps_desc));
  this->p_image->release();
}

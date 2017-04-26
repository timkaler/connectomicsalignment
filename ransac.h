static simple_mutex_t mutex = 0;

void updateAffineTransform(vdata* vertex, cv::Mat& transform) {
     

  cv::Mat A(3, 3, cv::DataType<double>::type);

  //vdata tmp;
  //tmp.a00 = warp_mat.at<double>(0, 0); 
  //tmp.a01 = warp_mat.at<double>(0, 1);
  //tmp.offset_x = warp_mat.at<double>(0, 2);
  //tmp.a10 = warp_mat.at<double>(1, 0); 
  //tmp.a11 = warp_mat.at<double>(1, 1); 
  //tmp.offset_y = warp_mat.at<double>(1, 2);
  //tmp.start_x = 0.0;
  //tmp.start_y = 0.0;


  A.at<double>(0,0) = vertex->a00;
  A.at<double>(0,1) = vertex->a01;
  A.at<double>(0,2) = vertex->offset_x + vertex->start_x;
  A.at<double>(1,0) = vertex->a10;
  A.at<double>(1,1) = vertex->a11;
  A.at<double>(1,2) = vertex->offset_y + vertex->start_y;
  A.at<double>(2,0) = 0.0;
  A.at<double>(2,1) = 0.0;
  A.at<double>(2,2) = 1.0;

  cv::Mat B = transform*A;


  vertex->a00 = B.at<double>(0,0);
  vertex->a01 = B.at<double>(0,1);
  vertex->a10 = B.at<double>(1,0);
  vertex->a11 = B.at<double>(1,1);
  vertex->offset_x = B.at<double>(0,2);
  vertex->offset_y = B.at<double>(1,2);

          double start_x = 0.0;
          double start_y = 0.0;
          double end_x = 0.0;
          double end_y = 0.0;



          //double post_start_x1 = transform.at<double>(0,0)*start_x + transform.at<double>(0,1)*start_y;
          //double post_start_x2 = transform.at<double>(0,0)*start_x + transform.at<double>(0,1)*end_y;
          //double post_start_y1 = transform.at<double>(1,1)*start_y + transform.at<double>(1,0)*start_x;
          //double post_start_y2 = transform.at<double>(1,1)*start_y + transform.at<double>(1,0)*end_x;


          //double post_end_x1 = transform.at<double>(0,0)*end_x + transform.at<double>(0,1)*start_y;
          //double post_end_x2 = transform.at<double>(0,0)*end_x + transform.at<double>(0,1)*end_y;
          //double post_end_y1 = transform.at<double>(1,1)*end_y + transform.at<double>(1,0)*start_x;
          //double post_end_y2 = transform.at<double>(1,1)*end_y + transform.at<double>(1,0)*end_x;



          //if (post_start_y1 > post_start_y2) {
          //  start_y = post_start_y1;
          //} else {
          //  start_y = post_start_y2;
          //}

          //if (post_start_x1 > post_start_x2) {
          //  start_x = post_start_x1;
          //} else {
          //  start_x = post_start_x2;
          //}



          //if (post_end_y1 < post_end_y2) {
          //  end_y = post_end_y1;
          //} else {
          //  end_y = post_end_y2;
          //}

          //if (post_end_x1 < post_end_x2) {
          //  end_x = post_end_x1;
          //} else {
          //  end_x = post_end_x2;
          //}







          ////merged_graph->getVertexData(v)->start_x = best_vertex_data.a00*start_x + best_vertex_data.a01*start_y;
          ////merged_graph->getVertexData(v)->start_y = best_vertex_data.a11*start_y + best_vertex_data.a10*start_x;
          vertex->start_x = start_x;// + transform.at<double>(0,2);
          vertex->start_y = start_y;// + transform.at<double>(1,2);
          vertex->end_x = end_x;// + transform.at<double>(0,2);
          vertex->end_y = end_y;// + transform.at<double>(1,2);

  



}

void updateAffineTransformOld(vdata* vertex, vdata* transform) {
      
  //std::cout << warp_mat << std::endl;
  vdata tmp;
  tmp.a00 = vertex->a00*transform->a00 + vertex->a10*transform->a01;
  tmp.a01 = vertex->a01*transform->a00 + vertex->a11*transform->a01;
  tmp.a10 = vertex->a00*transform->a10 + vertex->a10*transform->a11;
  tmp.a11 = vertex->a01*transform->a10 + vertex->a11*transform->a11;
  tmp.offset_x = vertex->offset_x*transform->a00 + vertex->offset_y*transform->a01 + transform->offset_x;
  tmp.offset_y = vertex->offset_x*transform->a10 + vertex->offset_y*transform->a11 + transform->offset_y; 
  //tmp.start_x = 0.0;
  //tmp.start_y = 0.0;

          double start_x = vertex->start_x;
          double start_y = vertex->start_y;
          double end_x = vertex->end_x;
          double end_y = vertex->end_y;


          double post_start_x1 = transform->a00*start_x + transform->a01*start_y;
          double post_start_x2 = transform->a00*start_x + transform->a01*end_y;
          double post_start_y1 = transform->a11*start_y + transform->a10*start_x;
          double post_start_y2 = transform->a11*start_y + transform->a10*end_x;


          double post_end_x1 = transform->a00*end_x + transform->a01*start_y;
          double post_end_x2 = transform->a00*end_x + transform->a01*end_y;
          double post_end_y1 = transform->a11*end_y + transform->a10*start_x;
          double post_end_y2 = transform->a11*end_y + transform->a10*end_x;



          if (post_start_y1 > post_start_y2) {
            start_y = post_start_y1;
          } else {
            start_y = post_start_y2;
          }

          if (post_start_x1 > post_start_x2) {
            start_x = post_start_x1;
          } else {
            start_x = post_start_x2;
          }



          if (post_end_y1 < post_end_y2) {
            end_y = post_end_y1;
          } else {
            end_y = post_end_y2;
          }

          if (post_end_x1 < post_end_x2) {
            end_x = post_end_x1;
          } else {
            end_x = post_end_x2;
          }







          //merged_graph->getVertexData(v)->start_x = best_vertex_data.a00*start_x + best_vertex_data.a01*start_y;
          //merged_graph->getVertexData(v)->start_y = best_vertex_data.a11*start_y + best_vertex_data.a10*start_x;
          vertex->start_x = start_x;
          vertex->start_y = start_y;
          vertex->end_x = end_x;
          vertex->end_y = end_y;
          //merged_graph->getVertexData(v)->start_x += merged_graph->getVertexData(v)->offset_x;
          //merged_graph->getVertexData(v)->start_y += merged_graph->getVertexData(v)->offset_y;
          //merged_graph->getVertexData(v)->end_x += merged_graph->getVertexData(v)->offset_x;
          //merged_graph->getVertexData(v)->end_y += merged_graph->getVertexData(v)->offset_y;
          //merged_graph->getVertexData(v)->offset_x = best_vertex_data.offset_x;
          //merged_graph->getVertexData(v)->offset_y = best_vertex_data.offset_y;
 





  vertex->a00 = tmp.a00;
  vertex->a01 = tmp.a01;
  vertex->a10 = tmp.a10;
  vertex->a11 = tmp.a11;
  vertex->offset_x = tmp.offset_x;
  vertex->offset_y = tmp.offset_y;

  //printf("tmp values are %f %f %f %f %f %f\n", tmp.a00, tmp.a01, tmp.a10, tmp.a11, tmp.offset_x, tmp.offset_y);

}


vdata getAffineTransform(std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
  cv::Mat warp_mat = cv::getAffineTransform(pts1, pts2);
  //std::cout << warp_mat << std::endl;

  vdata tmp;
  tmp.a00 = warp_mat.at<double>(0, 0); 
  tmp.a01 = warp_mat.at<double>(0, 1);
  tmp.offset_x = warp_mat.at<double>(0, 2);
  tmp.a10 = warp_mat.at<double>(1, 0); 
  tmp.a11 = warp_mat.at<double>(1, 1); 
  tmp.offset_y = warp_mat.at<double>(1, 2);
  tmp.start_x = 0.0;
  tmp.start_y = 0.0;


  //printf("tmp values are %f %f %f %f %f %f\n", tmp.a00, tmp.a01, tmp.a10, tmp.a11, tmp.offset_x, tmp.offset_y);

  return tmp;
}


double get_angle(cv::Point2f p1, cv::Point2f p2) {
    double angle = atan2(p2.y - p1.y, p2.x - p1.x) * 180.0 / CV_PI;

    if (angle < 0) {
        angle = angle + 360;
    }
    //printf("get_angle: %f\n", angle);
    return angle;
}

vdata tfk_simple_ransac_strict_ret_affine(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {

//
//  int angle_bins[360/20];
//  for (int i = 0; i < 360/20; i++) {
//    angle_bins[i] = 0;
//  }
//
//  // filter based on angle.
//  for (int i = 0; i < pre_match_points_a.size(); i++) {
//    int angle = get_angle(pre_match_points_a[i], pre_match_points_b[i])/20;
//    angle_bins[angle]++;
//  }
//
//  int max_angle_index = 0;
//  for (int i = 0; i < 360/20; i++) {
//    if (angle_bins[i] > angle_bins[max_angle_index]) max_angle_index = i;
//  }
//
//  printf("The max angle index is %d\n", max_angle_index);
//
//
//  for (int i = 0; i < pre_match_points_a.size(); i++) {
//    int angle = get_angle(pre_match_points_a[i], pre_match_points_b[i])/20;
//    //if (angle == max_angle_index) {
//      match_points_a.push_back(pre_match_points_a[i]); 
//      match_points_b.push_back(pre_match_points_b[i]); 
//    //}
//  }

  printf("After filtering the number of match points is %d\n", match_points_a.size());

  double best_dx = 0.0;
  double best_dy = 0.0;

  double best_a00 = 1.0;
  double best_a01 = 0.0;
  double best_a10 = 0.0;
  double best_a11 = 1.0;

  vdata best_vertex_data;

  int maxInliers = 0;
  int prevMaxInliers = 0;
  double thresh = 1.0;

  int num_iterations = 0;
  std::mt19937 g1 (1);  // mt19937 is a standard mersenne_twister_engine
  std::uniform_int_distribution<int> distribution(0,match_points_a.size()); 
  //for (; thresh <= _thresh || maxInliers < 10; thresh += 1.0) {
  for (thresh = _thresh; num_iterations < 1;) {
    num_iterations++;
    int limit = 100000;
    bool random = false;
    if (match_points_a.size() < limit) {
      limit = match_points_a.size();
      random = true;
    }

    uint64_t the_limit = match_points_a.size()*match_points_a.size()*match_points_a.size();


    //for (int angle = 0; angle < 18; angle++) {
    //for (int point1_index = 0; point1_index < match_points_a.size(); point1_index++) {
    //  if (maxInliers > 0.1*match_points_a.size()) break;
    //  int point1_angle = get_angle(match_points_a[point1_index], match_points_b[point1_index])/20;
    //  if (point1_angle != angle) {
    //    continue;
    //  }
    //  for (int point2_index = point1_index+1; point2_index < match_points_a.size(); point2_index++) {

    //    int point2_angle = get_angle(match_points_a[point2_index], match_points_b[point2_index]);
    //    if (point2_angle != angle) {
    //      continue;
    //    }
    //    for (int point3_index = point2_index+1; point3_index < match_points_a.size();
    //         point3_index++) {
    //      int point3_angle = get_angle(match_points_a[point3_index], match_points_b[point3_index]);
    //      if (point3_angle != angle) {
    //        continue;
    //      }
    for (int _j = 0; _j < the_limit/10000 + 1; _j++) {
    if (maxInliers > 0.1*match_points_a.size() && maxInliers > 12) { 
      printf("Max inliers is fraction %f breaking\n", maxInliers*1.0/match_points_a.size());
      break;
    }
    if (_j > 3 && maxInliers <= 0.1*match_points_a.size()) break;
 
    //if (maxInliers < 0.05*match_points_a.size()*_j) break;
    cilk_for (int _i = 0; _i < 10000; _i++) {      
      int i1 = distribution(g1);
      int i2 = distribution(g1);
      int i3 = distribution(g1);
      //int i1 = point1_index;
      //int i2 = point2_index;
      //int i3 = point3_index;
      std::vector<cv::Point2f> pts1;
      std::vector<cv::Point2f> pts2;
      pts1.push_back(match_points_a[i1]);
      pts1.push_back(match_points_a[i2]);
      pts1.push_back(match_points_a[i3]);
      pts2.push_back(match_points_b[i1]);
      pts2.push_back(match_points_b[i2]);
      pts2.push_back(match_points_b[i3]);
      vdata tmp_vertex_data = getAffineTransform(pts1, pts2);
      vdata tmp = tmp_vertex_data;
    //if (std::abs(tmp_vertex_data.a00) > 1.1 || tmp_vertex_data.a00 < 0.9 || std::abs(tmp_vertex_data.a01) > 0.1 || std::abs(tmp_vertex_data.a10) > 0.1 || std::abs(tmp_vertex_data.a11 > 1.1) || tmp_vertex_data.a11 < 0.9) {
    //  continue;
    //}

      cv::Point2f test_point_a = pts1[0];
      cv::Point2f test_point_b = pts2[0];

      cv::Point2f test_point_a_transformed = transform_point(&tmp_vertex_data, test_point_a);

      if(std::abs(test_point_a_transformed.x - test_point_b.x) > 1e-2) { 
        //printf("wtf strange error %f %f\n", test_point_a_transformed.x, test_point_b.x);
      } else {

      //int tmp_inliers = 0;
      //for (int k = 0; k < 200; k++) {
      //  int i4 = distribution(g1);
      //  cv::Point2f point_a = transform_point(&tmp_vertex_data, match_points_a[i4]);      
      //  cv::Point2f point_b = match_points_b[i4];

      //  double ndx = point_b.x - point_a.x;
      //  double ndy = point_b.y - point_a.y;
      //  double dist = ndx*ndx+ndy*ndy;
      //  if (dist <= thresh*thresh) {
      //     tmp_inliers++;
      //  }
      //}
      //if (tmp_inliers < 5*2) continue;

      int inliers = 0;

      for (int j = 0; j < match_points_b.size(); j++) {

        cv::Point2f point_a = transform_point(&tmp_vertex_data, match_points_a[j]);      
        cv::Point2f point_b = match_points_b[j];

        double ndx = point_b.x - point_a.x;
        double ndy = point_b.y - point_a.y;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          inliers+=1; 
        }
      }
      simple_acquire(&mutex);
      if (inliers > maxInliers) {
        maxInliers = inliers;
        best_dx = tmp_vertex_data.offset_x;
        best_dy = tmp_vertex_data.offset_y;
        best_a00 = tmp_vertex_data.a00;
        best_a01 = tmp_vertex_data.a01;
        best_a10 = tmp_vertex_data.a10;
        best_a11 = tmp_vertex_data.a11;
      }
      simple_release(&mutex);
      }

        }
      //}
    //}


    }
    }
  //}

  best_vertex_data.offset_x = best_dx;
  best_vertex_data.offset_y = best_dy;
  best_vertex_data.a00 = best_a00;
  best_vertex_data.a01 = best_a01;
  best_vertex_data.a10 = best_a10;
  best_vertex_data.a11 = best_a11;
  best_vertex_data.start_x = 0.0;
  best_vertex_data.start_y = 0.0;
  best_vertex_data.start_x = 0.0;
  best_vertex_data.start_x = 0.0;
  best_vertex_data.start_x = 0.0;
/*
    if (maxInliers > 5000 && num_iterations < 10) {
      thresh = thresh*0.9 - 1.0;
      maxInliers = prevMaxInliers;
    } else {
      prevMaxInliers = maxInliers;
    }
*/
/*
    if (maxInliers > 500) {
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }
      return;
    }*/

   for (int j = 0; j < match_points_b.size(); j++) {
        cv::Point2f point_a = transform_point(&best_vertex_data, match_points_a[j]);      
        cv::Point2f point_b = match_points_b[j];
        double ndx = point_b.x - point_a.x;
        double ndy = point_b.y - point_a.y;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }

    printf("Best values are %f %f %f %f %f %f\n", best_vertex_data.a00, best_vertex_data.a01, best_vertex_data.a10, best_vertex_data.a11, best_vertex_data.offset_x, best_vertex_data.offset_y);
    printf("Number of inliers is %d and best_dx %f best_dy %f\n", maxInliers, best_vertex_data.offset_x, best_vertex_data.offset_y);
    return best_vertex_data;
    //return std::pair<double,double>(best_vertex_data.offset_x,best_vertex_data.offset_y);
}
std::pair<double,double> tfk_simple_ransac_strict_ret(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {

  double best_dx = 0.0;
  double best_dy = 0.0;
  int maxInliers = 0;
  int prevMaxInliers = 0;
  double thresh = 1.0;

  int num_iterations = 0;
  std::mt19937 g1 (1);  // mt19937 is a standard mersenne_twister_engine
  std::uniform_int_distribution<int> distribution(0,match_points_a.size()); 
  //for (; thresh <= _thresh || maxInliers < 10; thresh += 1.0) {
  for (thresh = _thresh; num_iterations < 1;) {
    num_iterations++;
    int limit = 100000;
    bool random = false;
    if (match_points_a.size() < limit) {
      limit = match_points_a.size();
      random = true;
    }
    for (int _i = 0; _i < match_points_a.size(); _i++) {
      int i = _i;//distribution(g1);
      double dx = match_points_b[i].x - match_points_a[i].x;
      double dy = match_points_b[i].y - match_points_a[i].y;
      int inliers = 0;
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          inliers++;
        }
      }
      if (inliers > maxInliers) {
        maxInliers = inliers;
        best_dx = dx;
        best_dy = dy;
      }
    }
/*
    if (maxInliers > 5000 && num_iterations < 10) {
      thresh = thresh*0.9 - 1.0;
      maxInliers = prevMaxInliers;
    } else {
      prevMaxInliers = maxInliers;
    }
*/
/*
    if (maxInliers > 500) {
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }
      return;
    }*/

   double error = 0.0;
   for (int j = 0; j < match_points_b.size(); j++) {

        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          error += dist;
          mask[j]=true;
        }
      }

    

    printf("Number of inliers is %d and best_dx %f best_dy %f error is %f\n", maxInliers, best_dx, best_dy, error);
    return std::pair<double,double>(best_dx,best_dy);
  }

  int test_count = 0;
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
          test_count++; 
        }
      }
  printf("The test count is %d\n", test_count);
}


void tfk_simple_ransac_strict(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {

  double best_dx = 0.0;
  double best_dy = 0.0;
  int maxInliers = 0;
  int prevMaxInliers = 0;
  double thresh = _thresh;

  int num_iterations = 0;
  std::mt19937 g1 (1);  // mt19937 is a standard mersenne_twister_engine
  std::uniform_int_distribution<int> distribution(0,match_points_a.size()); 
  //for (; thresh <= _thresh; thresh += 1.0) {
  for (thresh = _thresh; num_iterations < 1;) {
    num_iterations++;
    int limit = 100000;
    //if (match_points_a.size() < limit) limit = match_points_a.size();
    for (int _i = 0; _i < match_points_a.size(); _i++) {
      int i = _i;//distribution(g1);
      double dx = match_points_b[i].x - match_points_a[i].x;
      double dy = match_points_b[i].y - match_points_a[i].y;
      int inliers = 0;
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          inliers++;
        }
      }
      if (inliers > maxInliers) {
        maxInliers = inliers;
        best_dx = dx;
        best_dy = dy;
      }
    }
/*
    if (maxInliers > 5000 && num_iterations < 10) {
      thresh = thresh*0.9 - 1.0;
      maxInliers = prevMaxInliers;
    } else {
      prevMaxInliers = maxInliers;
    }
*/
/*
    if (maxInliers > 500) {
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }
      return;
    }*/
  }

  int test_count = 0;
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
          test_count++; 
        }
      }
  printf("The test count is %d\n", test_count);
}


int tfk_simple_ransac(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {

  double best_dx = 0.0;
  double best_dy = 0.0;
  int maxInliers = 0;
  int prevMaxInliers = 0;
  double thresh = _thresh;

    for (int i = 0; i < match_points_a.size(); i++) {
      double dx = match_points_b[i].x - match_points_a[i].x;
      double dy = match_points_b[i].y - match_points_a[i].y;
      //if (maxInliers > 0.2 * match_points_a.size() && maxInliers > 12) break; 
      int inliers = 0;
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          inliers++;
        }
      }
      if (inliers > maxInliers) {
        maxInliers = inliers;
        best_dx = dx;
        best_dy = dy;
      }
    }

    printf("number of inliers is %d fraction is %f\n", maxInliers, (1.0*maxInliers) / match_points_a.size());

      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }
      return maxInliers;
}

void tfk_simple_ransac_old(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask) {

  double best_dx = 0.0;
  double best_dy = 0.0;
  int maxInliers = 0;
  int prevMaxInliers = 0;
  double thresh = 1.0;

  int num_iterations = 0;
  for (; thresh <= _thresh || maxInliers < 4; thresh += 1.0) {
    num_iterations++;
    for (int i = 0; i < match_points_a.size(); i++) {
      double dx = match_points_b[i].x - match_points_a[i].x;
      double dy = match_points_b[i].y - match_points_a[i].y;
      int inliers = 0;
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          inliers++;
        }
      }
      if (inliers > maxInliers) {
        maxInliers = inliers;
        best_dx = dx;
        best_dy = dy;
      }
    }

    if (maxInliers > 5000 && num_iterations < 10) {
      thresh = thresh*0.9 - 1.0;
      maxInliers = prevMaxInliers;
    } else {
      prevMaxInliers = maxInliers;
    }

    if (maxInliers > 500) {
      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }
      return;
    }
  }

      // mark inliers
      for (int j = 0; j < match_points_b.size(); j++) {
        double ndx = match_points_b[j].x - match_points_a[j].x - best_dx;
        double ndy = match_points_b[j].y - match_points_a[j].y - best_dy;
        double dist = ndx*ndx+ndy*ndy;
        if (dist <= thresh*thresh) {
          mask[j]=true;
        }
      }

}



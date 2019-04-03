// Copyright 2016 - Supertech Research Group
//   Tim Kaler, Tao B. Schardl, Haoran Xu, Charles E. Leiserson, Alex Matveev.

#include "./match.h"

cv::Point2f transform_point(vdata* vertex, cv::Point2f point_local) {
  float new_x = point_local.x*vertex->a00 + point_local.y * vertex->a01 + vertex->offset_x + vertex->start_x;
  float new_y = point_local.x*vertex->a10 + point_local.y * vertex->a11 + vertex->offset_y + vertex->start_y;
  return cv::Point2f(new_x, new_y);
}





void match_features_brute_parallel(std::vector< cv::DMatch > &matches, cv::Mat &descs1, cv::Mat &descs2, float rod) {


  std::vector<bool> is_good(descs1.rows);
  std::vector<int> good_matches(descs1.rows);


  std::cout << "type is " << descs1.type() << std::endl;
  std::cout << "rows,cols is  " << descs1.rows << "," << descs1.cols << std::endl;

  cilk_for (int i = 0; i < descs1.rows; i++) {
    float max_value = FLT_MAX;
    float second_max_value = FLT_MAX;

    int max_index = 0;
    int second_max_index = 0;

    for (int j = 0; j < descs2.rows; j++) {
      float dot_product = 0.0;
      for (int k = 0; k < 128; k++) {
        float val = (descs1.at<float>(i,k) - descs2.at<float>(j,k));
        dot_product += val*val;//(descs1(i,k)-descs2(i,k));
      }

      dot_product = sqrt(dot_product);

      if (dot_product < max_value) {
        second_max_value = max_value;
        second_max_index = max_index;
        max_value = dot_product;
        max_index = j;
      } else if (dot_product < second_max_value) {
        second_max_value = dot_product;
        second_max_index = j;
      } 
    }


      if (max_value < second_max_value * rod) {
        is_good[i] = true;
        good_matches[i] = max_index;
      } else {
        is_good[i] = false;
        good_matches[i] = -1;
      }
  }

  for (int i = 0; i < descs1.rows; i++) {
    if (is_good[i]) {
      matches.push_back(cv::DMatch(i, good_matches[i], 1.0));
    }
  }
  printf("there are %zu good matches", matches.size());
}



// Helper method to match the features of two tiles.
void match_features(std::vector< cv::DMatch > &matches, cv::Mat &descs1, cv::Mat &descs2, float rod, bool brute) {
  std::vector< std::vector < cv::DMatch > > raw_matches;
  if (true || descs1.rows + descs1.cols > descs2.rows + descs2.cols) {
    //if (descs1.rows +descs1.cols > 0 && false) {
    ////cv::BFMatcher matcher(cv::NORM_L2, false);
    //cv::Ptr<cv::flann::IndexParams> index_params = new cv::flann::KDTreeIndexParams(4);
    ////cv::Ptr<cv::flann::SearchParams> search_params = new cv::flann::SearchParams(64, 0.0001,true);
    //cv::FlannBasedMatcher matcher;
    //matcher.knnMatch(descs1, descs2,
    //                 raw_matches,
    //                 2);
    //} else {
    if (brute) {
    cv::BFMatcher matcher(cv::NORM_L2, false);



    //static const cv::Ptr<cv::flann::IndexParams> index_params = new cv::flann::KDTreeIndexParams(16);
    //static const cv::Ptr<cv::flann::SearchParams> search_params = new cv::flann::SearchParams(128, 0, false);
    //static const cv::Ptr<cv::flann::IndexParams> index_params = new cv::flann::KDTreeIndexParams(4);
    //static const cv::Ptr<cv::flann::SearchParams> search_params = new cv::flann::SearchParams(32, 1.0, true);
    //cv::FlannBasedMatcher matcher(index_params, search_params);
    //cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    //cv::FlannBasedMatcher matcher;
    matcher.knnMatch(descs1, descs2,
                     raw_matches,
                     2);
    } else {

      //std::vector<std::vector< std::vector < cv::DMatch > > > raw_matches_tmp (descs1.rows/1000 + 4);

      cv::FlannBasedMatcher matcher;
      matcher.knnMatch(descs1, descs2,
                       raw_matches,
                       2);
    }



    matches.reserve(raw_matches.size());
    //}
    // Apply ratio test
    for (size_t i = 0; i < raw_matches.size(); i++) {
      if(raw_matches[i].size() >= 2){
      if (raw_matches[i][0].distance <
          (rod * raw_matches[i][1].distance)) {
        matches.push_back(raw_matches[i][0]);
      }
      }
    }
  } else {
    cv::BFMatcher matcher(cv::NORM_L2, false);
    //cv::FlannBasedMatcher matcher;
    matcher.knnMatch(descs2, descs1,
                     raw_matches,
                     2);

    matches.reserve(raw_matches.size());
    // Apply ratio test
    for (size_t i = 0; i < raw_matches.size(); i++) {
      if(raw_matches[i].size() >= 2){
      if (raw_matches[i][0].distance <
          (rod * raw_matches[i][1].distance)) {
        matches.push_back(cv::DMatch(raw_matches[i][0].trainIdx, raw_matches[i][0].queryIdx, raw_matches[i][0].distance));
      }
      }
    }
  }
}


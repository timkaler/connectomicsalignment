// Copyright 2016 - Supertech Research Group
//   Tim Kaler, Tao B. Schardl, Haoran Xu, Charles E. Leiserson, Alex Matveev.

#include "./match.h"

cv::Point2f transform_point(vdata* vertex, cv::Point2f point_local) {
  float new_x = point_local.x*vertex->a00 + point_local.y * vertex->a01 + vertex->offset_x + vertex->start_x;
  float new_y = point_local.x*vertex->a10 + point_local.y * vertex->a11 + vertex->offset_y + vertex->start_y;
  return cv::Point2f(new_x, new_y);
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

      cv::FlannBasedMatcher matcher;
      matcher.knnMatch(descs1, descs2,
                       raw_matches,
                       2);
    }

    //}
    matches.reserve(raw_matches.size());
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

void alternative_match_features(std::vector< cv::DMatch > &matches, cv::Mat &descs1, cv::Mat &descs2, float rod) {
  std::vector< std::vector < cv::DMatch > > raw_matches;
  if (true || descs1.rows + descs1.cols > descs2.rows + descs2.cols) {
    //cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    cv::FlannBasedMatcher matcher;
    //cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    matcher.knnMatch(descs1, descs2,
                     raw_matches,
                     2);

    matches.reserve(raw_matches.size());
    // Apply ratio test
    for (size_t i = 0; i < raw_matches.size(); i++) {
      if (raw_matches[i][0].distance <
          (rod * raw_matches[i][1].distance)) {
        matches.push_back(raw_matches[i][0]);
      }
    }
  } else {
    //cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    //cv::FlannBasedMatcher matcher;
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    matcher.knnMatch(descs2, descs1,
                     raw_matches,
                     2);

    matches.reserve(raw_matches.size());
    // Apply ratio test
    for (size_t i = 0; i < raw_matches.size(); i++) {
      if (raw_matches[i][0].distance <
          (rod * raw_matches[i][1].distance)) {
        matches.push_back(cv::DMatch(raw_matches[i][0].trainIdx, raw_matches[i][0].queryIdx, raw_matches[i][0].distance));
      }
    }
  }
}

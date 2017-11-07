

#ifndef TFKRANSAC
#define TFKRANSAC
#include "./common.h"
#include "./match.h"
#include "./cilk_tools/Graph.h"

void updateAffineSectionTransform(vdata* _vertex, cv::Mat& transform);
void updateAffineTransform(vdata* vertex, cv::Mat& transform);
void updateAffineTransformOld(vdata* vertex, vdata* transform);



vdata getAffineTransform(std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2);



std::pair<double,double> tfk_simple_ransac_strict_ret(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask);

vdata tfk_simple_ransac_strict_ret_affine(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask);

void tfk_simple_ransac_strict(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask);


int tfk_simple_ransac(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask);

void tfk_simple_ransac_old(std::vector<cv::Point2f>& match_points_a,
    std::vector<cv::Point2f>& match_points_b, double _thresh, bool* mask);

#endif

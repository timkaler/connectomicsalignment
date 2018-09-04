/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc/types_c.h>

#ifndef STACK_HELPERS_H
#define STACK_HELPERS_H


namespace cv {
  bool computeAffineTFK(const std::vector<cv::Point2f> &srcPoints, const std::vector<cv::Point2f> &dstPoints, Mat &transf);
}

inline std::string matchPadTo(std::string str, const size_t num, const char paddingChar = '0');

cv::Mat colorize(cv::Mat c1, cv::Mat c2, cv::Mat c3, unsigned int channel = 0);


cv::Mat apply_heatmap_to_grayscale(cv::Mat* gray, cv::Mat* heat_floats, int nrows, int ncols);

// Helper method to check if a key point is inside a given bounding
// box.
__attribute__((const))
bool bbox_contains(float pt_x, float pt_y,
                          int x_start, int x_finish,
                          int y_start, int y_finish);


float Dot(cv::Point2f a, cv::Point2f b);

double computeTriangleArea(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);

double c_huber(double value, double target, double sigma, double d_value_dx, double d_value_dy,
               double* d_huber_dx, double* d_huber_dy);


double c_reglen(double vx, double vy, double d_vx_dx, double d_vy_dy,
                double* d_reglen_dx, double* d_reglen_dy);


double crosslink_mesh_derivs(std::vector<cv::Point2f>* mesh1,
                             cv::Point2f* d_cost_d_mesh1,
                             int* indices1, double* barys1,
                             double all_weight, double sigma, cv::Point2f dest_p);


double internal_mesh_derivs(std::vector<cv::Point2f>* mesh, cv::Point2f* d_cost_d_mesh,
                            std::pair<int, int> edge_indices,
                            double rest_length, double all_weight,
                            double sigma);


double area_mesh_derivs(std::vector<cv::Point2f>* mesh, cv::Point2f* d_cost_d_mesh,
                        int* triangle_indices, double rest_area, double all_weight);



void Barycentric(cv::Point2f p, cv::Point2f a, cv::Point2f b, cv::Point2f c,
   float &u, float &v, float &w);


#endif

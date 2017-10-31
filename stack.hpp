#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <mutex>

#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/types_c.h>
#include <hdf5.h>
#include <opencv2/hdf/hdf5.hpp>
#include "cilk_tools/Graph.h"
//#include "cilk_tools/engine.h"
#include "./common.h"
#include "./fasttime.h"
#include "AlignData.pb.h"
#include "./match.h"
#include "./ransac.h"
#include "cilk_tools/engine.h"
//#include "./meshoptimize.h"
#ifndef ALIGNSTACK
#define ALIGNSTACK

void updateTile2DAlign(int vid, void* scheduler_void);
static bool STORE_ALIGN_RESULTS = false;
static double totalTime = 0;
static float CONTRAST_THRESH = 0.04;
static float CONTRAST_THRESH_3D = 0.04;
static float EDGE_THRESH_3D = 5.0;
static float EDGE_THRESH_2D = 5.0;
static cv::Mat colorize(cv::Mat c1, cv::Mat c2, cv::Mat c3, unsigned int channel = 0) {
    std::vector<cv::Mat> channels;
    channels.push_back(c1);
    channels.push_back(c2);
    channels.push_back(c3);
    cv::Mat color;
    cv::merge(channels, color);
    return color;
}


static cv::Mat apply_heatmap_to_grayscale(cv::Mat* gray, cv::Mat* heat_floats, int nrows, int ncols) {
  cv::Mat c1,c2,c3;
  c1.create(nrows, ncols, CV_8UC1);
  c2.create(nrows, ncols, CV_8UC1);
  c3.create(nrows, ncols, CV_8UC1);

  for (int x = 0; x < gray->size().width; x++) {
    for (int y = 0; y < gray->size().height; y++) {
       float g = (1.0*gray->at<unsigned char>(y,x))/255;
       float h = 0.5 + 0.5*(heat_floats->at<float>(y,x)*heat_floats->at<float>(y,x));

       c1.at<unsigned char>(y,x) = (unsigned char) (g*255*(1-h));
       c2.at<unsigned char>(y,x) = (unsigned char) (g*255*(1-h));
       c3.at<unsigned char>(y,x) = (unsigned char) (g*255*((h)));
    }
  }
  return colorize(c1,c2,c3);
}

// Helper method to check if a key point is inside a given bounding
// box.
__attribute__((const))
static bool bbox_contains(float pt_x, float pt_y,
                          int x_start, int x_finish,
                          int y_start, int y_finish) {
  // TRACE_1("  -- pt: (%f, %f)\n", pt.x, pt.y);
  // TRACE_1("  -- bbox: [(%d, %d), (%d, %d)]\n",
  //         x_start, y_start,
  //         x_finish, y_finish);
  return (pt_x >= x_start && pt_x <= x_finish) &&
    (pt_y >= y_start && pt_y <= y_finish);
}

static float Dot(cv::Point2f a, cv::Point2f b) {
  return a.x*b.x + a.y*b.y;
}

static double computeTriangleArea(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {


  double v01x = p2.x - p1.x;
  double v01y = p2.y - p1.y;

  double v02x = p3.x - p1.x;
  double v02y = p3.y - p1.y;

  double area = 0.5 * (v02x * v01y - v01x * v02y);
  return area;
  //double dx,dy;
  //dx = p1.x-p2.x;
  //dy = p1.y-p2.y;
  //double p1p2 = std::sqrt(dx*dx+dy*dy);

  //dx = p1.x-p3.x;
  //dy = p1.y-p3.y;
  //double p1p3 = std::sqrt(dx*dx+dy*dy);

  //dx = p2.x-p3.x;
  //dy = p2.y-p3.y;
  //double p2p3 = std::sqrt(dx*dx+dy*dy);

  //double p = (p1p2+p1p3+p2p3)/2;
  //double area = std::sqrt(p*(p-p1p2)*(p-p1p3)*(p-p2p3));
  //return area;
}

static double c_huber(double value, double target, double sigma, double d_value_dx, double d_value_dy,
               double* d_huber_dx, double* d_huber_dy) {
  double diff, a, b;

  diff = value - target;
  if (std::abs(diff) <= sigma) {
    a = (diff*diff)/2;
    d_huber_dx[0] = diff * d_value_dx;
    d_huber_dy[0] = diff * d_value_dy;
    return a;
  } else {
    b = sigma * (std::abs(diff) - sigma / 2);
    d_huber_dx[0] = sigma * d_value_dx;
    d_huber_dy[0] = sigma * d_value_dy;
    return b;
  }
}

static double c_reglen(double vx, double vy, double d_vx_dx, double d_vy_dy,
                double* d_reglen_dx, double* d_reglen_dy) {
  double sq_len, sqrt_len;
  double small_value = 0.0001;
  sq_len = vx * vx + vy * vy + small_value;
  sqrt_len = std::sqrt(sq_len);
  d_reglen_dx[0] = vx / sqrt_len;
  d_reglen_dy[0] = vy / sqrt_len;
  return sqrt_len;
}


static double crosslink_mesh_derivs(std::vector<cv::Point2f>* mesh1, std::vector<cv::Point2f>* mesh2,
                             cv::Point2f* d_cost_d_mesh1, cv::Point2f* d_cost_d_mesh2,
                             int* indices1, int* indices2, double* barys1, double* barys2,
                             double all_weight, double sigma) {
  double px, py, qx, qy;
  int pidx0, pidx1, pidx2;
  int qidx0, qidx1, qidx2;
  double pb0, pb1, pb2;
  double qb0, qb1, qb2;
  double r, h;
  double dr_dx, dr_dy, dh_dx, dh_dy;
  double cost;

  cost = 0;

  pidx0 = indices1[0];
  pidx1 = indices1[1];
  pidx2 = indices1[2];

  pb0 = barys1[0];
  pb1 = barys1[1];
  pb2 = barys1[2];

  qidx0 = indices2[0];
  qidx1 = indices2[1];
  qidx2 = indices2[2];

  qb0 = barys2[0];
  qb1 = barys2[1];
  qb2 = barys2[2];

  px = (*mesh1)[pidx0].x * pb0 +
       (*mesh1)[pidx1].x * pb1 + 
       (*mesh1)[pidx2].x * pb2;

  py = (*mesh1)[pidx0].y * pb0 +
       (*mesh1)[pidx1].y * pb1 + 
       (*mesh1)[pidx2].y * pb2;

  qx = (*mesh2)[qidx0].x * qb0 +
       (*mesh2)[qidx1].x * qb1 + 
       (*mesh2)[qidx2].x * qb2;

  qy = (*mesh2)[qidx0].y * qb0 +
       (*mesh2)[qidx1].y * qb1 + 
       (*mesh2)[qidx2].y * qb2;

  r = c_reglen(px-qx, py-qy,1,1,&(dr_dx),&(dr_dy));
  h = c_huber(r, 0, sigma, dr_dx, dr_dy, &(dh_dx), &(dh_dy));

  cost += h * all_weight;
  dh_dx *= all_weight;
  dh_dy *= all_weight;

  // update derivs.
  d_cost_d_mesh1[pidx0].x += (float)1.0*(pb0 * dh_dx);
  d_cost_d_mesh1[pidx1].x += (float)1.0*(pb1 * dh_dx);
  d_cost_d_mesh1[pidx2].x += (float)1.0*(pb2 * dh_dx);

  d_cost_d_mesh1[pidx0].y += (float)(pb0 * dh_dy);
  d_cost_d_mesh1[pidx1].y += (float)(pb1 * dh_dy);
  d_cost_d_mesh1[pidx2].y += (float)(pb2 * dh_dy);

  d_cost_d_mesh2[qidx0].x -= (float)(qb0 * dh_dx);
  d_cost_d_mesh2[qidx1].x -= (float)(qb1 * dh_dx);
  d_cost_d_mesh2[qidx2].x -= (float)(qb2 * dh_dx);

  d_cost_d_mesh2[qidx0].y -= (float)(qb0 * dh_dy);
  d_cost_d_mesh2[qidx1].y -= (float)(qb1 * dh_dy);
  d_cost_d_mesh2[qidx2].y -= (float)(qb2 * dh_dy);

  return cost;
}

static double internal_mesh_derivs(std::vector<cv::Point2f>* mesh, cv::Point2f* d_cost_d_mesh,
                            std::pair<int, int> edge_indices,
                            double rest_length, double all_weight,
                            double sigma) {
  int idx1,idx2;
  double px,py,qx,qy;
  double r,h;
  double dr_dx, dr_dy, dh_dx, dh_dy;
  double cost;

  cost = 0;

  idx1 = edge_indices.first;
  idx2 = edge_indices.second;

  px = (*mesh)[idx1].x;
  py = (*mesh)[idx1].y;

  qx = (*mesh)[idx2].x; 
  qy = (*mesh)[idx2].y;

  r = c_reglen(px-qx, py-qy, 1, 1, &(dr_dx), &(dr_dy));
  h = c_huber(r, rest_length, sigma, dr_dx, dr_dy, &(dh_dx), &(dh_dy));

  cost += h * all_weight;
  dh_dx *= all_weight;
  dh_dy *= all_weight;

  // update derivs.
  d_cost_d_mesh[idx1].x += dh_dx;
  d_cost_d_mesh[idx1].y += dh_dy;
  d_cost_d_mesh[idx2].x -= dh_dx;
  d_cost_d_mesh[idx2].y -= dh_dy;

  return cost;
}

static double area_mesh_derivs(std::vector<cv::Point2f>* mesh, cv::Point2f* d_cost_d_mesh,
                        int* triangle_indices, double rest_area, double all_weight) {
  int idx0, idx1, idx2;
  double v01x, v01y, v02x, v02y, area, r_area;
  double cost, c, dc_da;

  cost = 0;

  idx0 = triangle_indices[0];
  idx1 = triangle_indices[1];
  idx2 = triangle_indices[2];

  v01x = (*mesh)[idx1].x - (*mesh)[idx0].x;
  v01y = (*mesh)[idx1].y - (*mesh)[idx0].y;

  v02x = (*mesh)[idx2].x - (*mesh)[idx0].x;
  v02y = (*mesh)[idx2].y - (*mesh)[idx0].y;

  area = 0.5 * (v02x * v01y - v01x * v02y);
  r_area = rest_area;
  if (area*r_area <= 0) {
    c = INFINITY;
    dc_da = 0;
  } else {
    double tmp = ((area - r_area) / area);
    c = all_weight * (tmp*tmp);
    dc_da = 2 * all_weight * r_area * (area - r_area) / (area*area*area);
  }
  cost += c;

  // update derivs
  d_cost_d_mesh[idx1].x += dc_da * 0.5 * (-1*v02y);
  d_cost_d_mesh[idx1].y += dc_da * 0.5 * (v02x);
  d_cost_d_mesh[idx2].x += dc_da * 0.5 * (v01y);
  d_cost_d_mesh[idx2].y += dc_da * 0.5 * (-1*v01x);

  // sum of negative of above.
  d_cost_d_mesh[idx0].x += dc_da * 0.5 * (v02y - v01y);
  d_cost_d_mesh[idx0].y += dc_da * 0.5 * (v01x - v02x);

  return cost;
}



static void Barycentric(cv::Point2f p, cv::Point2f a, cv::Point2f b, cv::Point2f c,
   float &u, float &v, float &w)
{
    cv::Point2f v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = Dot(v0, v0);
    float d01 = Dot(v0, v1);
    float d11 = Dot(v1, v1);
    float d20 = Dot(v2, v0);
    float d21 = Dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}


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



namespace tfk {

enum Resolution {THUMBNAIL, FULL};

class Tile {
  public:
   int section_id;
   int tile_id;
   int mfov_id;
   int index;
   std::string filepath;
   cv::Mat * p_image;
   double x_start;
   double x_finish;
   double y_start;
   double y_finish;


   double shape_dx;
   double shape_dy;

   std::vector<cv::KeyPoint>* p_kps;
   cv::Mat* p_kps_desc;

   std::vector<cv::KeyPoint>* p_kps_3d;
   cv::Mat* p_kps_desc_3d;

   double a00;
   double a10;
   double a11;
   double a01;
   double offset_x;
   double offset_y;

   bool* ignore;
   std::vector<renderTriangle>* mesh_triangles;
   int level;
   bool bad;

   std::vector<edata> edges;
   std::vector<edata> add_edges; //temporary.


   Tile(int section_id, int tile_id, int index, std::string filepath,
            int x_start, int x_finish, int y_start, int y_finish);

   Tile(TileData& tile_data);

   void compute_sift_keypoints2d();
   void compute_sift_keypoints3d();

   cv::Point2f rigid_transform(cv::Point2f pt);


   void get_3d_keypoints(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Mat>& desc);
   bool overlaps_with(std::pair<cv::Point2f, cv::Point2f> bbox);
   bool overlaps_with(Tile* other);
   void local2DAlignUpdate();
   void insert_matches(Tile* neighbor, std::vector<cv::Point2f>& points_a, std::vector<cv::Point2f>& points_b);
   void make_symmetric(int phase, std::vector<Tile*>& tile_list);
   void write_wafer(FILE* wafer_file, int section_id, int base_section);

   std::pair<cv::Point2f, cv::Point2f> get_bbox();
   std::vector<cv::Point2f> get_corners();
};


class Section {
  public:
    int section_id;
    int real_section_id;
    int n_tiles;
    int out_d1;
    int out_d2;
    cv::Mat* p_out;
    std::vector<cv::KeyPoint>* p_kps;
    std::string cached_2d_matches;
    std::string cached_3d_keypoints;

    std::vector<cv::Mat> affine_transforms;

    std::vector<Tile*> tiles;
    Graph* graph;

    double a00;
    double a10;
    double a11;
    double a01;
    double offset_x;
    double offset_y;

    std::vector<cv::Point2f>* mesh_orig;
    std::vector<cv::Point2f>* mesh_old;
    std::vector<cv::Point2f>* mesh;
    std::vector<std::pair<int,int> >* triangle_edges;
    std::vector<tfkTriangle>* triangles;
    cv::Point2f* gradients;
    cv::Point2f* gradients_with_momentum;
    double* rest_lengths;
    double* rest_areas;



    std::vector<tfkMatch> section_mesh_matches;
    cv::Mat coarse_transform;



    Section(int section_id);
    Section(SectionData& section_data);
    std::vector<int> get_all_close_tiles(int atile_id);
    void compute_keypoints_and_matches();
    void compute_tile_matches(int tile_id, Graph* graph);


    void coarse_affine_align(Section* neighbor);
    std::vector<cv::Point2f>* generate_hex_grid(double* bounding_box, double spacing);
    void construct_triangles();

    std::pair<cv::Point2f, cv::Point2f> get_bbox();

    cv::Point2f affine_transform(cv::Point2f pt);
    cv::Point2f elastic_transform(cv::Point2f pt);

    cv::Mat* read_tile(std::string filepath, Resolution res);


    bool overlaps_with(std::pair<cv::Point2f, cv::Point2f> bbox);

    std::pair<cv::Point2f, cv::Point2f> affine_transform_bbox(
        std::pair<cv::Point2f, cv::Point2f> bbox);
    void affine_transform_keypoints(std::vector<cv::KeyPoint>& keypoints);
    void get_elastic_matches_one(Section* neighbor);
    void get_elastic_matches(std::vector<Section*> neighbors);
    //std::vector<cv::Point2f>* generate_hex_grid(double* bounding_box, double spacing);
    void affine_transform_mesh();
    //void construct_triangles();

    void write_wafer(FILE* wafer_file, int base_section);


    std::pair<cv::Point2f, cv::Point2f> scale_bbox(std::pair<cv::Point2f, cv::Point2f> bbox,
        cv::Point2f scale);
    bool tile_in_render_box(Tile* tile, std::pair<cv::Point2f, cv::Point2f> bbox);

    cv::Mat render(std::pair<cv::Point2f, cv::Point2f> bbox, Resolution resolution);
    void render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename);
    cv::Point2f get_render_scale(Resolution resolution);

    void render_error(Section* neighbor, std::pair<cv::Point2f, cv::Point2f> bbox,
                      std::string filename);

    renderTriangle getRenderTriangle(tfkTriangle tri);
    std::tuple<bool, float, float, float> get_triangle_for_point(cv::Point2f pt);

    void apply_affine_transforms();

    void read_3d_keypoints(std::string filename);
    void save_3d_keypoints(std::string filename);
    void read_2d_graph(std::string filename);
    void save_2d_graph(std::string filename);

    bool section_data_exists();

    void read_tile_matches();
    void save_tile_matches();
};



class Stack {
 public:
    // member variables
    int mode;
    std::string input_filepath;
    std::string output_dirpath;
    int base_section;
    int n_sections;
    bool do_subvolume;
    int min_x;
    int min_y;
    int max_x;
    int max_y;

    // list of sections
    std::vector<Section*> sections;

    Graph* merged_graph;

    Stack(int base_section, int n_sections, std::string input_filepath,
               std::string output_dirpath);

    void init();
    void align_2d();
    void pack_graph();
    void unpack_graph();
    void coarse_affine_align();
    void elastic_align();

    void get_elastic_matches();
    void elastic_gradient_descent();

    void render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix);

    void render_error(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix);
};

} // end namespace tfk.
#endif // ALIGNSTACK

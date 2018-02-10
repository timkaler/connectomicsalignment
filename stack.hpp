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



namespace tfk {

enum Resolution {THUMBNAIL, FULL, PERCENT30, THUMBNAIL2};

typedef struct params {
    int num_features; // actually what size to start on but thats what we call it
    int num_octaves;
    float contrast_threshold;
    float edge_threshold;
    float sigma;
    Resolution res;
    float scale_x;
    float scale_y;
} params;

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

   bool bad_2d_alignment;
   double shape_dx;
   double shape_dy;

   std::vector<cv::KeyPoint>* p_kps;
   cv::Mat* p_kps_desc;

   std::vector<cv::KeyPoint>* p_kps_fallback;
   cv::Mat* p_kps_desc_fallback;

   void compute_sift_keypoints2d_params(tfk::params params,
      std::vector<cv::KeyPoint>& local_keypoints, cv::Mat& local_desc);
   void compute_sift_keypoints2d_params(tfk::params params,
      std::vector<cv::KeyPoint>& local_keypoints, cv::Mat& local_desc, Tile* other_tile);



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

   bool image_data_replaced;

   std::vector<edata> edges;
   std::vector<edata> add_edges; //temporary.


   Tile(int section_id, int tile_id, int index, std::string filepath,
            int x_start, int x_finish, int y_start, int y_finish);

   Tile(TileData& tile_data);

   void compute_sift_keypoints2d();
   void compute_sift_keypoints3d(bool recomputation = false);
   void compute_sift_keypoints_with_params(params p);

   cv::Point2f rigid_transform(cv::Point2f pt);

   void release_2d_keypoints();

   float error_tile_pair(Tile *other);

   void get_3d_keypoints(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Mat>& desc);

   void recompute_3d_keypoints(std::vector<cv::KeyPoint>& atile_all_kps,
                                       std::vector<cv::Mat>& atile_all_kps_desc,
                                       tfk::params sift_parameters);


   bool overlaps_with(std::pair<cv::Point2f, cv::Point2f> bbox);
   bool overlaps_with(Tile* other);
   void local2DAlignUpdate();
   void insert_matches(Tile* neighbor, std::vector<cv::Point2f>& points_a, std::vector<cv::Point2f>& points_b);
   void make_symmetric(int phase, std::vector<Tile*>& tile_list);
   void write_wafer(FILE* wafer_file, int section_id, int base_section);

   std::pair<cv::Point2f, cv::Point2f> get_bbox();
   std::vector<cv::Point2f> get_corners();

   cv::Mat get_tile_data(Resolution res);
};


class Section {
  public:
    int section_id;
    int real_section_id;
    int n_tiles;
    int out_d1;
    int out_d2;
    bool elastic_transform_ready;
    int num_tiles_replaced;
    int num_bad_2d_matches;
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

    std::vector<std::vector<tfkTriangle>* > triangles;

    //std::vector<tfkTriangle>* triangles;

    cv::Point2f* gradients;
    cv::Point2f* gradients_with_momentum;
    double* rest_lengths;
    double* rest_areas;

    std::set<int> replaced_tile_ids;


    std::vector<tfkMatch> section_mesh_matches;
    cv::Mat coarse_transform;



    Section(int section_id);
    Section(SectionData& section_data);
    std::vector<int> get_all_close_tiles(int atile_id);
    std::vector<Tile*> get_all_close_tiles(Tile* atile_id);
    void compute_keypoints_and_matches();
    void compute_tile_matches(Tile* a_tile);

    void compute_tile_matches_pair(Tile* a_tile, Tile* b_tile,
      std::vector< cv::KeyPoint >& a_tile_keypoints, std::vector <cv::KeyPoint>& b_tile_keypoints,
      cv::Mat& a_tile_desc, cv::Mat& b_tile_desc,
      std::vector< cv::Point2f > &filtered_match_points_a,
      std::vector< cv::Point2f > &filtered_match_points_b, float ransac_thresh);




    void recompute_keypoints();

    void get_3d_keypoints_for_box(std::pair<cv::Point2f, cv::Point2f> bbox,
        std::vector<cv::KeyPoint>& kps_in_box, cv::Mat& kps_desc_in_box, bool use_cached,
        tfk::params sift_parameters);

   void find_3d_matches_in_box(Section* neighbor,
       std::pair<cv::Point2f, cv::Point2f> sliding_bbox,
       std::vector<cv::Point2f>& test_filtered_match_points_a,
       std::vector<cv::Point2f>& test_filtered_match_points_b,
       bool use_cached, tfk::params sift_parameters);


   double compute_3d_error_in_box(Section* neighbor,
       std::pair<cv::Point2f, cv::Point2f> sliding_bbox,
       std::vector<cv::Point2f>& test_filtered_match_points_a,
       std::vector<cv::Point2f>& test_filtered_match_points_b);



    void coarse_affine_align(Section* neighbor);
    std::vector<cv::Point2f>* generate_hex_grid(double* bounding_box, double spacing);
    void construct_triangles();

    std::pair<cv::Point2f, cv::Point2f> get_bbox();

    cv::Point2f affine_transform(cv::Point2f pt);
    cv::Point2f elastic_transform(cv::Point2f pt);
    cv::Point2f affine_transform(cv::Mat A, cv::Point2f pt);
    cv::Point2f affine_transform_plusA(cv::Point2f pt, cv::Mat A);
    
	cv::Mat* read_tile(std::string filepath, Resolution res);


    bool overlaps_with(std::pair<cv::Point2f, cv::Point2f> bbox);

    std::pair<cv::Point2f, cv::Point2f> affine_transform_bbox(
        std::pair<cv::Point2f, cv::Point2f> bbox);
    std::pair<cv::Point2f, cv::Point2f> elastic_transform_bbox(
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
	bool tile_in_render_box_affine(cv::Mat A, Tile* tile, std::pair<cv::Point2f, cv::Point2f> bbox);




    cv::Mat render_affine(cv::Mat A, std::pair<cv::Point2f, cv::Point2f> bbox, Resolution resolution);
    cv::Mat render(std::pair<cv::Point2f, cv::Point2f> bbox, Resolution resolution);
    void render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename, Resolution res);
    cv::Point2f get_render_scale(Resolution resolution);

     std::pair<std::vector<std::pair<cv::Point2f, cv::Point2f>> , std::vector<std::pair<cv::Point2f, cv::Point2f>>> render_error(Section* neighbor, Section* other_neighbor, Section* other2_neighbor, std::pair<cv::Point2f, cv::Point2f> bbox,
                      std::string filename);

     //std::pair<std::vector<std::pair<cv::Point2f, cv::Point2f>> , std::vector<std::pair<cv::Point2f, cv::Point2f>>> 
    double render_error_affine(Section* neighbor, std::pair<cv::Point2f, cv::Point2f> bbox,
                      std::string filename, cv::Mat A);

    renderTriangle getRenderTriangle(tfkTriangle tri);
    std::tuple<bool, float, float, float> get_triangle_for_point(cv::Point2f pt);

    void apply_affine_transforms();
    void apply_affine_transforms(cv::Mat A);

    void read_3d_keypoints(std::string filename);
    void save_3d_keypoints(std::string filename);
    void read_2d_graph(std::string filename);
    void save_2d_graph(std::string filename);

    bool section_data_exists();

    void read_tile_matches();
    void save_tile_matches();

    void replace_bad_region(std::pair<cv::Point2f, cv::Point2f> bad_bbox,
                            Section* other_neighbor);

    // tile belongs to *this* section.
    void replace_bad_tile(Tile* tile, Section* other_neighbor);

    bool transformed_tile_overlaps_with(Tile* tile,
        std::pair<cv::Point2f, cv::Point2f> bbox);
    
    std::vector<std::tuple<int, double, int>> parameter_optimization(int trials, double threshold, std::vector<params> &ps);
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


    void recompute_alignment();

    void get_elastic_matches();
    void elastic_gradient_descent();

    void render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix, Resolution res);

    void render_error(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix);
};

} // end namespace tfk.
#endif // ALIGNSTACK

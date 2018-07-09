#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <mutex>
#include <thread>
#include <future>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc/types_c.h>
#include "cilk_tools/Graph.h"
//#include "cilk_tools/engine.h"
#include "./common.h"
#include "./fasttime.h"
#include "AlignData.pb.h"

#include "./match.h"
#include "./ransac.h"

#include "cilk_tools/engine.h"
#include "mrtask.hpp"
#include "mrparams.hpp"
#include "paramdb.hpp"
#include "mlbase.hpp"
#include "triangle_mesh.hpp"
#include "triangle.h"
//#include "./meshoptimize.h"

#include "matchtilestask.hpp"
#include "matchtilepairtask.hpp"

#ifndef ALIGNSTACK
#define ALIGNSTACK

void updateTile2DAlign(int vid, void* scheduler_void);
static bool STORE_ALIGN_RESULTS = false;
static double totalTime = 0;
static float CONTRAST_THRESH = 0.04;
static float CONTRAST_THRESH_3D = 0.04;
static float EDGE_THRESH_3D = 5.0;
static float EDGE_THRESH_2D = 5.0;


#define NUM_MRTASKS 2
#define MATCH_TILE_PAIR_TASK_ID 0
#define MATCH_TILES_TASK_ID 1


namespace tfk {

// forward declaration.
class MatchTilesTask;
enum Resolution {THUMBNAIL, FULL, PERCENT30, THUMBNAIL2, FILEIOTEST};

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
   std::map<int, cv::Point2f> ideal_offsets;



   std::map<int, cv::Point2f> ideal_offsets_first;
   std::map<int, cv::Point2f> ideal_offsets_second;
   bool both_passes;

   std::map<int, float> neighbor_correlations;
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
   bool tmp_bad_2d_alignment;
   double shape_dx;
   double shape_dy;
   std::pair<cv::Mat, cv::Mat> get_overlap_matrix(Tile* other, float scale);

   //MRTask* match_tiles_task;

   MatchTilesTask* match_tiles_task;


   std::vector<cv::KeyPoint>* p_kps;
   cv::Mat* p_kps_desc;

   std::vector<cv::KeyPoint>* p_kps_fallback;
   cv::Mat* p_kps_desc_fallback;

   void compute_sift_keypoints2d_params(tfk::params params,
      std::vector<cv::KeyPoint>& local_keypoints, cv::Mat& local_desc);
   void compute_sift_keypoints2d_params(tfk::params params,
      std::vector<cv::KeyPoint>& local_keypoints, cv::Mat& local_desc, Tile* other_tile);

   float compute_deviation(Tile* b_tile);

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

   TileData tile_data;


   bool has_full_image;
   bool has_percent30_image;
   cv::Mat full_image;
   cv::Mat percent30_image;
   std::mutex* full_image_lock;
   std::mutex* percent30_lock;

   bool image_data_replaced;

   std::vector<edata> edges;
   std::vector<edata> add_edges; //temporary.

   // an array of pointers
   MLBase* *ml_models;
   ParamDB* *paramdbs;

   std::map<Tile*, std::vector<float>> feature_vectors;
   std::map<Tile*, bool> ml_preds;

   std::map<Tile*, int> keypoints_in_overlap;
   std::map<Tile*, int> matched_keypoints_in_overlap;


   Tile(int section_id, int tile_id, int index, std::string filepath,
            int x_start, int x_finish, int y_start, int y_finish);

   Tile(TileData& tile_data);

   void release_full_image();

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
   void local2DAlignUpdateLimited(std::set<Tile*>* active_set);
   void insert_matches(Tile* neighbor, std::vector<cv::Point2f>& points_a, std::vector<cv::Point2f>& points_b);
   void make_symmetric(int phase, std::vector<Tile*>& tile_list);
   void write_wafer(FILE* wafer_file, int section_id, int base_section);

   std::pair<cv::Point2f, cv::Point2f> get_bbox();
   std::vector<cv::Point2f> get_corners();

   cv::Mat get_tile_data(Resolution res);
   cv::Mat get_feature_vector(Tile *other, int boxes, int type);
};


class Section {
  public:
    std::vector<Tile*> get_all_neighbor_tiles(tfk::Tile* tile);
    int section_id;
    int real_section_id;
    int n_tiles;
    int out_d1;
    int out_d2;
    bool elastic_transform_ready;
    int num_tiles_replaced;
    int num_bad_2d_matches;
    bool use_bbox_prefilter;
    std::pair<cv::Point2f, cv::Point2f> _bounding_box;
    cv::Mat* p_out;
    std::vector<cv::KeyPoint>* p_kps;
    std::string cached_2d_matches;
    std::string cached_3d_keypoints;

    std::vector<cv::Mat> affine_transforms;

    std::vector<Tile*> tiles;
    Graph* graph;

    std::mutex* section_mesh_matches_mutex;

    double a00;
    double a10;
    double a11;
    double a01;
    double offset_x;
    double offset_y;

    bool alignment2d_exists();
    void load_2d_alignment();
    void save_2d_alignment();

    //std::vector<cv::Point2f>* mesh_orig;
    std::vector<cv::Point2f>* mesh_orig_save;
    std::vector<cv::Point2f>* mesh_old;
    //std::vector<cv::Point2f>* mesh;

    //std::vector<std::vector<tfkTriangle>* > triangles;

    //std::vector<tfkTriangle>* triangles;

    cv::Point2f* gradients;
    cv::Point2f* gradients_with_momentum;
    double* rest_lengths;
    double* rest_areas;

    std::set<int> replaced_tile_ids;


    std::vector<tfkMatch> section_mesh_matches;
    cv::Mat coarse_transform;

    // an array of pointers
    MLBase* *ml_models;
    ParamDB* *paramdbs;


    bool load_elastic_mesh(Section* neighbor);
    void save_elastic_mesh(Section* neighbor);


    void compute_on_tile_neighborhood(tfk::Tile* tile);
    Section(int section_id);
    Section(SectionData& section_data, std::pair<cv::Point2f, cv::Point2f> bounding_box, bool use_bbox_prefilter);
    std::vector<int> get_all_close_tiles(int atile_id);
    std::vector<Tile*> get_all_close_tiles(Tile* atile_id);
    void compute_keypoints_and_matches();
    void compute_tile_matches(Tile* a_tile);
    void compute_tile_matches2(Tile* a_tile);

    void align_3d(Section* neighbor);


    cv::Point2f compute_tile_matches_pair(Tile* a_tile, Tile* b_tile,
      std::vector< cv::KeyPoint >& a_tile_keypoints, std::vector <cv::KeyPoint>& b_tile_keypoints,
      cv::Mat& a_tile_desc, cv::Mat& b_tile_desc,
      std::vector< cv::Point2f > &filtered_match_points_a,
      std::vector< cv::Point2f > &filtered_match_points_b, float ransac_thresh);

    void elastic_gradient_descent_section(Section* _neighbor);

    void align_2d();

    void recompute_keypoints();

    void get_3d_keypoints_for_box(std::pair<cv::Point2f, cv::Point2f> bbox,
        std::vector<cv::KeyPoint>& kps_in_box, cv::Mat& kps_desc_in_box, bool use_cached,
        tfk::params sift_parameters,
        std::vector<Tile*>& tiles_loaded, std::mutex& tiles_loaded_mutex, bool apply_transform);

   void find_3d_matches_in_box(Section* neighbor,
       std::pair<cv::Point2f, cv::Point2f> sliding_bbox,
       std::vector<cv::Point2f>& test_filtered_match_points_a,
       std::vector<cv::Point2f>& test_filtered_match_points_b,
       bool use_cached, tfk::params sift_parameters,
       std::vector<Tile*>& tiles_loaded, std::mutex& tiles_loaded_mutex);



  void find_3d_matches_in_box_cache(Section* neighbor,
    std::pair<cv::Point2f, cv::Point2f> sliding_bbox,
    std::vector<cv::Point2f>& test_filtered_match_points_a,
    std::vector<cv::Point2f>& test_filtered_match_points_b,
    bool use_cached, tfk::params sift_parameters, std::vector<Tile*>& tiles_loaded, std::mutex& tiles_loaded_mutex, std::vector<cv::KeyPoint>& prev_keypoints, cv::Mat& prev_desc,
              std::vector<cv::KeyPoint>& my_keypoints, cv::Mat& my_desc);

void get_elastic_matches_relative(Section* neighbor);




   TriangleMesh* triangle_mesh;


   double compute_3d_error_in_box(Section* neighbor,
       std::pair<cv::Point2f, cv::Point2f> sliding_bbox,
       std::vector<cv::Point2f>& test_filtered_match_points_a,
       std::vector<cv::Point2f>& test_filtered_match_points_b,
       std::vector<Tile*>& tiles_loaded, std::mutex& tiles_loaded_mutex);



    void coarse_affine_align(Section* neighbor);
    void construct_triangles();

    std::pair<cv::Point2f, cv::Point2f> get_bbox();

    cv::Point2f affine_transform(cv::Point2f pt);
    cv::Point2f elastic_transform(cv::Point2f pt);
    cv::Point2f elastic_transform(cv::Point2f pt, Triangle tri);
    cv::Point2f affine_transform(cv::Mat A, cv::Point2f pt);
    cv::Point2f affine_transform_plusA(cv::Point2f pt, cv::Mat A);
    
	cv::Mat* read_tile(std::string filepath, Resolution res);


    bool overlaps_with(std::pair<cv::Point2f, cv::Point2f> bbox);

    std::pair<cv::Point2f, cv::Point2f> affine_transform_bbox(
        std::pair<cv::Point2f, cv::Point2f> bbox);
    std::pair<cv::Point2f, cv::Point2f> elastic_transform_bbox(
        std::pair<cv::Point2f, cv::Point2f> bbox);
    void affine_transform_keypoints(std::vector<cv::KeyPoint>& keypoints);
    //std::vector<cv::Point2f>* generate_hex_grid(double* bounding_box, double spacing);
    void affine_transform_mesh();
    //void construct_triangles();

    void write_wafer(FILE* wafer_file, int base_section);


    std::pair<cv::Point2f, cv::Point2f> scale_bbox(std::pair<cv::Point2f, cv::Point2f> bbox,
        cv::Point2f scale);
    bool tile_in_render_box(Tile* tile, std::pair<cv::Point2f, cv::Point2f> bbox);
	bool tile_in_render_box_affine(cv::Mat A, Tile* tile, std::pair<cv::Point2f, cv::Point2f> bbox);




    cv::Mat render_affine(cv::Mat A, std::pair<cv::Point2f, cv::Point2f> bbox, Resolution resolution,
                          std::vector<Tile*>& tiles_loaded, std::mutex& tiles_loaded_mutex);
    cv::Mat render(std::pair<cv::Point2f, cv::Point2f> bbox, Resolution resolution);
    void render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename, Resolution res);
    cv::Point2f get_render_scale(Resolution resolution);

     std::pair<std::vector<std::pair<cv::Point2f, cv::Point2f>> , std::vector<std::pair<cv::Point2f, cv::Point2f>>> render_error(Section* neighbor, Section* other_neighbor, Section* other2_neighbor, std::pair<cv::Point2f, cv::Point2f> bbox,
                      std::string filename);

     //std::pair<std::vector<std::pair<cv::Point2f, cv::Point2f>> , std::vector<std::pair<cv::Point2f, cv::Point2f>>> 
    double render_error_affine(Section* neighbor, std::pair<cv::Point2f, cv::Point2f> bbox,
                      std::string filename, cv::Mat A,
                      std::vector<Tile*>& tiles_loaded, std::mutex& tiles_loaded_mutex);

    renderTriangle getRenderTriangle(tfkTriangle tri);
    std::tuple<bool, float, float, float, int> get_triangle_for_point(cv::Point2f pt);
    std::tuple<bool, float, float, float, int> get_triangle_for_point(cv::Point2f pt, Triangle tri);

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
        std::pair<cv::Point2f, cv::Point2f> bbox, bool use_elastic);
    
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
    bool use_bbox_prefilter;
    std::pair<cv::Point2f, cv::Point2f> _bounding_box;
    // list of sections
    std::vector<Section*> sections;

    Graph* merged_graph;

    MLBase* ml_models[NUM_MRTASKS];
    ParamDB* paramdbs[NUM_MRTASKS];


    // Init functions
    Stack(int base_section, int n_sections, std::string input_filepath,
               std::string output_dirpath);
    void init();

    void train_fsj(int trials);

    // Test functions
    void test_io();
    void compute_on_tile_neighborhood(Section* section, Tile* tile);

    // Rendering functions
    void render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix, Resolution res);
    void render_error(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix);

    // Alignment algorithms
    void align_2d();
    void align_3d();

    //void render_error(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix);
    void parameter_optimization(int trials, double threshold, std::vector<params> ps, std::vector<std::tuple<int, double, int>>& results);
    void test_learning(int trials, int vector_grid_size, int vector_mode);

};







} // end namespace tfk.
#endif // ALIGNSTACK

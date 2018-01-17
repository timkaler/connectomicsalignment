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

enum Resolution {THUMBNAIL, FULL, PERCENT30};

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

   bool image_data_replaced;

   std::vector<edata> edges;
   std::vector<edata> add_edges; //temporary.


   Tile(int section_id, int tile_id, int index, std::string filepath,
            int x_start, int x_finish, int y_start, int y_finish);

   Tile(TileData& tile_data);

   void compute_sift_keypoints2d();
   void compute_sift_keypoints3d(bool recomputation = false);

   cv::Point2f rigid_transform(cv::Point2f pt);

   void release_2d_keypoints();

   void get_3d_keypoints(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Mat>& desc);
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

    std::set<int> replaced_tile_ids;


    std::vector<tfkMatch> section_mesh_matches;
    cv::Mat coarse_transform;



    Section(int section_id);
    Section(SectionData& section_data);
    std::vector<int> get_all_close_tiles(int atile_id);
    std::vector<Tile*> get_all_close_tiles(Tile* atile_id);
    void compute_keypoints_and_matches();
    void compute_tile_matches(Tile* a_tile, Graph* graph);

    void recompute_keypoints();


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

    cv::Mat render(std::pair<cv::Point2f, cv::Point2f> bbox, Resolution resolution);
    void render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename, Resolution res);
    cv::Point2f get_render_scale(Resolution resolution);

     std::pair<std::vector<std::pair<cv::Point2f, cv::Point2f>> , std::vector<std::pair<cv::Point2f, cv::Point2f>>> render_error(Section* neighbor, Section* other_neighbor, Section* other2_neighbor, std::pair<cv::Point2f, cv::Point2f> bbox,
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

    void replace_bad_region(std::pair<cv::Point2f, cv::Point2f> bad_bbox,
                            Section* other_neighbor);

    // tile belongs to *this* section.
    void replace_bad_tile(Tile* tile, Section* other_neighbor);

    bool transformed_tile_overlaps_with(Tile* tile,
        std::pair<cv::Point2f, cv::Point2f> bbox);
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

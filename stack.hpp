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
static bool STORE_ALIGN_RESULTS = false;

static double totalTime = 0;
static float CONTRAST_THRESH = 0.04;
static float CONTRAST_THRESH_3D = 0.04;
static float EDGE_THRESH_3D = 5.0;
static float EDGE_THRESH_2D = 5.0;

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


namespace tfk {


class Tile {
  public:
   int section_id;
   int tile_id;
   int mfov_id;
   int index;
   std::string filepath;
   cv::Mat * p_image;
   int x_start;
   int x_finish;
   int y_start;
   int y_finish;

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


   Tile(int section_id, int tile_id, int index, std::string filepath,
            int x_start, int x_finish, int y_start, int y_finish);

   Tile(TileData& tile_data);

   void compute_sift_keypoints2d();
   void compute_sift_keypoints3d();

   bool overlaps_with(Tile* other);

};


class Section {
  public:
    int section_id;
    int n_tiles;
    int out_d1;
    int out_d2;
    cv::Mat* p_out;
    std::vector<cv::KeyPoint>* p_kps;
    std::string cached_2d_matches;
    std::string cached_3d_keypoints;

    std::vector<Tile*> tiles;
    Graph<vdata, edata>* graph;

    Section(int section_id);
    Section(SectionData& section_data);
    std::vector<int> get_all_close_tiles(int atile_id);
    void compute_keypoints_and_matches();
    void compute_tile_matches(int tile_id, Graph<vdata, edata>* graph);
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

    Graph<vdata, edata>* merged_graph;

    Stack(int base_section, int n_sections, std::string input_filepath,
               std::string output_dirpath);

    void init();
    void align_2d();
    void pack_graph();
    void unpack_graph();
};

} // end namespace tfk.
#endif // ALIGNSTACK

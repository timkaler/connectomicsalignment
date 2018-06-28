

#ifndef MATCHTILEPAIRTASK
#define MATCHTILEPAIRTASK
#include "mrtask.hpp"
#include "stack.hpp"
#include "tilesifttask.hpp"

namespace tfk {

class MatchTilePairTask : public MRTask {
  public:
    std::vector<int> param_adjustments;
    std::vector<int> param_train_deltas;

    std::map<int, TileSiftTask*> dependencies;

    void set_random_train();
    //void update_result(float last_correct, float next_correct);

    Tile* a_tile;
    Tile* b_tile;
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f> > matched_points;
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f> > alt_matched_points;
    bool success;

    void compute_tile_matches_pair(Tile* a_tile, Tile* b_tile,
      std::vector< cv::KeyPoint >& a_tile_keypoints, std::vector< cv::KeyPoint >& b_tile_keypoints,
      cv::Mat& a_tile_desc, cv::Mat& b_tile_desc,
      std::vector< cv::Point2f > &filtered_match_points_a,
      std::vector< cv::Point2f > &filtered_match_points_b, float ransac_thresh);

    void alternative_compute_tile_matches_pair(Tile* a_tile, Tile* b_tile,
      std::vector< cv::KeyPoint >& a_tile_keypoints, std::vector< cv::KeyPoint >& b_tile_keypoints,
      cv::Mat& a_tile_desc, cv::Mat& b_tile_desc,
      std::vector< cv::Point2f > &filtered_match_points_a,
      std::vector< cv::Point2f > &filtered_match_points_b, float ransac_thresh);

    MatchTilePairTask(Tile* a_tile, Tile* b_tile);
    virtual ~MatchTilePairTask() final;

    void compute_with_params(tfk::MRParams* mr_params_local);
    void commit();
    bool error_check(float false_negative_rate);
    void get_parameter_options(std::vector<tfk::MRParams*>* vec);
    bool bbox_contains(float pt_x, float pt_y,
                              int x_start, int x_finish,
                              int y_start, int y_finish);
};

} // end namespace tfk
#endif


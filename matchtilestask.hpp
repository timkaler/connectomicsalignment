

#ifndef MATCHTILESTASK
#define MATCHTILESTASK
#include "mrtask.hpp"
#include "stack.hpp"

namespace tfk {

class MatchTilesTask : public MRTask {
  public:
    std::vector<int> param_adjustments;
    std::vector<int> param_train_deltas;

    void set_random_train();
    //void update_result(float last_correct, float next_correct);

    Tile* tile;
    std::vector<Tile*> neighbors;
    std::map<Tile*, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f> > > neighbor_to_matched_points;
    std::map<Tile*, bool> neighbor_to_success;
     void compute_tile_matches_pair(Tile* a_tile, Tile* b_tile,
      std::vector< cv::KeyPoint >& a_tile_keypoints, std::vector< cv::KeyPoint >& b_tile_keypoints,
      cv::Mat& a_tile_desc, cv::Mat& b_tile_desc,
      std::vector< cv::Point2f > &filtered_match_points_a,
      std::vector< cv::Point2f > &filtered_match_points_b, float ransac_thresh);

    MatchTilesTask (ParamDB* paramDB, Tile* tile, std::vector<Tile*> neighbors);

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


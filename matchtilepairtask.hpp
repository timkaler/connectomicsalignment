

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

    cv::Point2d current_offset;

    void set_random_train();
    //void update_result(float last_correct, float next_correct);

    cv::Point2f compute_quick(Tile* a_tile, Tile* b_tile);

    bool second_pass;
    Tile* a_tile;
    Tile* b_tile;
    cv::Point2d predicted_offset;
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f> > matched_points;
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f> > alt_matched_points;
    cv::Point2f best_offset;
    bool success;
    float successful_rod; 
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
    MatchTilePairTask(Tile* a_tile, Tile* b_tile, bool train);
    virtual ~MatchTilePairTask() final;

    void compute_with_params(tfk::MRParams* mr_params_local);
    void commit();
    bool error_check(float false_negative_rate);
    void get_parameter_options(std::vector<tfk::MRParams*>* vec);
    bool bbox_contains(float pt_x, float pt_y,
                              int x_start, int x_finish,
                              int y_start, int y_finish);
    bool compare_results_and_update_model(MRTask* known_good, float accuracy);
    std::vector<float> feature_vector;
    std::vector<float> get_feature_vector();
};

} // end namespace tfk
#endif


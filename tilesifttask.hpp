
#include "mrtask.hpp"
#include "stack.hpp"

#ifndef TILESIFTTASK
#define TILESIFTTASK

namespace tfk {

class TileSiftTask : public MRTask {
  public:
    Tile* tile;
    std::vector<cv::KeyPoint> tile_keypoints;
    cv::Mat tile_desc;
    // std::vector<cv::KeyPoint> alt_tile_keypoints;
    // cv::Mat alt_tile_desc;

    TileSiftTask (ParamDB* paramDB, Tile* tile);
    virtual ~TileSiftTask () final ;
    void compute_with_params(tfk::MRParams* mr_params_local);
    void commit();
    bool error_check(float false_negative_rate);
    void get_parameter_options(std::vector<tfk::MRParams*>* vec);
};

} // end namespace tfk
#endif


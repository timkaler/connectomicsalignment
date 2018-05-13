#include "tilesifttask.hpp"
namespace tfk {

    TileSiftTask::~TileSiftTask () {
      this->tile_keypoints.clear();
      std::vector<cv::KeyPoint>().swap(this->tile_keypoints);
      this->tile_desc.release();
    }

    TileSiftTask::TileSiftTask (ParamDB* paramDB, Tile* tile) {
      this->paramDB = paramDB;
      this->tile = tile;
      this->task_type_id = 1;
    }

    void TileSiftTask::compute_with_params(MRParams* mr_params_local) {

      MRParams* mr_params = mr_params_local;

      tfk::params new_params;
      new_params.scale_x = mr_params->get_float_param("scale");
      new_params.scale_y = mr_params->get_float_param("scale");

      //printf("scale x %f scale y %f\n", new_params.scale_x, new_params.scale_y);
      new_params.num_features = mr_params->get_int_param("num_features");
      new_params.num_octaves = mr_params->get_int_param("num_octaves");
      new_params.contrast_threshold = 0.015;//mr_params->get_float_param("contrast_threshold");
      new_params.edge_threshold = 6;// mr_params->get_float_param("edge_threshold");
      new_params.sigma = 1.2;//mr_params->get_float_param("sigma");

      tile->compute_sift_keypoints2d_params(new_params, tile_keypoints,
                                              tile_desc, tile);
    }

    void TileSiftTask::commit() {
      // do nothing
    }

    bool TileSiftTask::error_check(float prob) {
      return true;
    }

    void TileSiftTask::get_parameter_options(std::vector<tfk::MRParams*>* vec) {
      //TODO(wheatman) figure out what the parameters of this function are
    }
}


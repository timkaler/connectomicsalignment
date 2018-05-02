#include "matchtilestask.hpp"
#include "matchtilepairtask.hpp"
namespace tfk {

    MatchTilesTask::MatchTilesTask (Tile* tile, std::vector<Tile*> neighbors) {
        //TODO(wheatman) should be a different paramdb, but we aren't using it yet
      this->tile = tile;
      this->neighbors = neighbors;
      this->task_type_id = MATCH_TILES_TASK_ID;
      this->paramDB = tile->paramdbs[this->task_type_id];
      this->model = tile->ml_models[this->task_type_id];
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        child_tasks[b_tile] = new MatchTilePairTask(tile, b_tile);
      }
    }


    void MatchTilesTask::compute_with_params(MRParams* mr_params_local) {
      this->mr_params = mr_params_local;

      //int neighbor_success_count = 0;
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        //TODO(wheatman) something smarter here dealing with the parameters
        child_tasks[b_tile]->dependencies = dependencies;
        child_tasks[b_tile]->compute(.9);
      }
    }
//TODO(wheatman) mark to neighbors as bad

    bool MatchTilesTask::error_check(float false_negative_rate) {
      int neighbor_success_count = 0;

      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        //TODO(wheatman) something smarter to pick this number
        bool guess = child_tasks[b_tile]->error_check(.9);
        if (guess) {
          neighbor_to_success[b_tile] = true;
          neighbor_success_count++;
        } else {
          cv::Point2f a_point = cv::Point2f(tmp_a_tile.x_start+tmp_a_tile.offset_x,
                                            tmp_a_tile.y_start+tmp_a_tile.offset_y);
          cv::Point2f b_point = cv::Point2f(b_tile->x_start+b_tile->offset_x,
                                            b_tile->y_start+b_tile->offset_y);
          cv::Point2f delta = a_point - b_point;

          a_tile->ideal_offsets[b_tile->tile_id] = delta;
          neighbor_to_success[b_tile] = false;
        }
      }
      //TODO(wheatman) some thing smarter here
      // its own mlbase model to learn how to predict
      if (neighbor_success_count > neighbors.size()*2.0/4.0 && neighbor_success_count >= 3.0) {
        return true;
      } else {
        return false;
      }
    }

    void MatchTilesTask::commit() {
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        if (neighbor_to_success[b_tile]) {
          child_tasks[b_tile]->commit();
        }
      }
    }

    void MatchTilesTask::get_parameter_options(std::vector<tfk::MRParams*>* vec) {
      //TODO(wheatman) figure out what the parameters of this function are
    }

}


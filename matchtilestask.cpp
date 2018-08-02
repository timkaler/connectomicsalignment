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

    MatchTilesTask::~MatchTilesTask () {
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        delete child_tasks[b_tile];
      }
    }

    void MatchTilesTask::compute_with_params(MRParams* mr_params_local) {
      printf("UH OH?!?!\n");
      exit(1);
      this->mr_params = mr_params_local;

      //int neighbor_success_count = 0;
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        //TODO(wheatman) something smarter here dealing with the parameters
        dynamic_cast<MatchTilePairTask*>(child_tasks[b_tile])->dependencies = dependencies;
        child_tasks[b_tile]->compute(.9);
      }
    }

    void MatchTilesTask::compute(float accuracy) {
      //int neighbor_success_count = 0;
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        //TODO(wheatman) something smarter here dealing with the parameters
        dynamic_cast<MatchTilePairTask*>(child_tasks[b_tile])->dependencies = dependencies;
        if (!neighbor_to_success[b_tile] && !tile->ml_preds[b_tile]) {
          child_tasks[b_tile]->compute(accuracy);
        }
      }
    }

//TODO(wheatman) mark to neighbors as bad

    bool MatchTilesTask::error_check(float false_negative_rate) {
      int neighbor_success_count = 0;

      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        //TODO(wheatman) something smarter to pick this number
        bool guess = child_tasks[b_tile]->error_check(false_negative_rate);
        if (guess) {
          neighbor_to_success[b_tile] = true;
          neighbor_success_count++;
        } else {
          //cv::Point2f a_point = cv::Point2f(tmp_a_tile.x_start+tmp_a_tile.offset_x,
          //                                  tmp_a_tile.y_start+tmp_a_tile.offset_y);
          //cv::Point2f b_point = cv::Point2f(b_tile->x_start+b_tile->offset_x,
          //                                  b_tile->y_start+b_tile->offset_y);
          //cv::Point2f delta = a_point - b_point;

          //a_tile->ideal_offsets[b_tile->tile_id] = delta;
          neighbor_to_success[b_tile] = false;
        }
      }
      //TODO(wheatman) some thing smarter here
      // its own mlbase model to learn how to predict
      //if (neighbor_success_count > neighbors.size()*2.0/4.0 && neighbor_success_count >= 3.0) {


      // require in addition that there be a neighbor up,down,left,right.

      bool required[4];
      bool has[4];
      for (int i = 0; i < 4; i++) {
        required[i] = false;
        has[i] = false;
      }

      int mx = tile->x_start+tile->offset_x;
      int my = tile->y_start+tile->offset_y;
      for (int j = 0; j < neighbors.size(); j++) {
        int nx = neighbors[j]->x_start + neighbors[j]->offset_x;
        int ny = neighbors[j]->y_start + neighbors[j]->offset_y;
        for (int i = 0; i < 4; i++) {
          if (nx > mx + 1000) {
            required[0] = true;
            if (neighbor_to_success[neighbors[j]]) {
              has[0] = true;
            }
          }
          if (ny > my + 1000) {
            required[1] = true;
            if (neighbor_to_success[neighbors[j]]) {
              has[1] = true;
            }
          }
          if (nx < mx - 1000) {
            required[2] = true;
            if (neighbor_to_success[neighbors[j]]) {
              has[2] = true;
            }
          }
          if (ny < my - 1000) {
            required[3] = true;
            if (neighbor_to_success[neighbors[j]]) {
              has[3] = true;
            }
          }
        }
      }

      //for (int i = 0; i < 4; i++) {
      //  if (required[i] && !has[i]) return false;
      //}

      if (neighbor_success_count >= neighbors.size() * 3.0/4.0 && neighbor_success_count >= 2.0) {
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


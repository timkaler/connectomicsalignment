#include "matchtilestask.hpp"
#include "matchtilepairtask.hpp"
namespace tfk {

    MatchTilesTask::MatchTilesTask (ParamDB* paramDB, ParamDB* paramDB_for_children, Tile* tile, std::vector<Tile*> neighbors) {
        //TODO(wheatman) should be a different paramdb, but we aren't using it yet
      this->paramDB = paramDB_for_children;
      this->tile = tile;
      this->neighbors = neighbors;
      this->task_type_id = MATCH_TILES_TASK_ID;
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        child_tasks[b_tile] = new MatchTilePairTask(paramDB_for_children, tile, b_tile);
      }
    }



    void MatchTilesTask::compute_with_params(MRParams* mr_params_local) {
      this->mr_params = mr_params_local;


      //int neighbor_success_count = 0;
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        //TODO(wheatman) something smarter here dealing with the parameters
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


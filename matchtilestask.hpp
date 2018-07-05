

#ifndef MATCHTILESTASK
#define MATCHTILESTASK

#include "mrtask.hpp"
#include "stack.hpp"
#include "tilesifttask.hpp"

namespace tfk {

class MatchTilesTask : public MRTask {
  public:
    std::vector<int> param_adjustments;
    std::vector<int> param_train_deltas;
    std::map<int, TileSiftTask*> dependencies;

    void set_random_train();
    //void update_result(float last_correct, float next_correct);

    Tile* tile;
    std::vector<Tile*> neighbors;
    std::map<Tile*, bool> neighbor_to_success;
    std::map<Tile*, MRTask*> child_tasks;

    virtual ~MatchTilesTask () final;
    MatchTilesTask (Tile* tile, std::vector<Tile*> neighbors);
    MatchTilesTask (Tile* tile, std::vector<Tile*> neighbors,
                    std::map<int, TileSiftTask*> dependencies);
    //~MatchTilesTask ();

    void compute_with_params(tfk::MRParams* mr_params_local);
    void compute(float probability_correct);
    void commit();
    bool error_check(float false_negative_rate);
    void get_parameter_options(std::vector<tfk::MRParams*>* vec);
    bool compare_results_and_update_model(MRTask* known_good, float accuracy) {return false;}
};

} // end namespace tfk
#endif


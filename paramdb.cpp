#include "paramdb.hpp"

namespace tfk {

  ParamDB::ParamDB (MRParams* default_params, MRParams* min_params, MRParams* max_params) {
    this->default_params = default_params;
    this->min_params = min_params;
    this->max_params = max_params;

    possible_params.push_back(default_params);
    possible_params.push_back(min_params);
    possible_params.push_back(max_params);
  }

  MRParams* ParamDB::get_params_for_accuracy(float accuracy) {
    MRParams* ret = this->possible_params[0];
    // Find the lowest cost MRParams with greater than given accuracy.
    for (int i = 0; i < possible_params.size(); i++) {
      MRParams* params = possible_params[i];
      if (params->get_accuracy() >= accuracy) {
        if (params->get_cost() < ret->get_cost() || ret->get_accuracy() < accuracy) {
          ret = params;  
        }
      }
    }
  }

  void ParamDB::import_params(MRParams* params) {
    possible_params.push_back(params);
  }

  MRParams* ParamDB::get_max_params() {return max_params;}
  MRParams* ParamDB::get_min_params() {return min_params;}
  MRParams* ParamDB::get_default_params() {return default_params;}

// end namespace tfk
}

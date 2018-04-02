
#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <cmath>
#include "mrparams.hpp"

#ifndef PARAMDB
#define PARAMDB

namespace tfk {
  class ParamDB {
    public:
      MRParams* default_params;
      MRParams* min_params;
      MRParams* max_params;

      std::mutex* mutex;

      std::vector<MRParams*> possible_params;

      ParamDB (MRParams* default_params, MRParams* min_params, MRParams* max_params);

      MRParams* get_params_for_accuracy(float accuracy);

      MRParams* get_max_params();
      MRParams* get_min_params();
      MRParams* get_default_params();

      void record_success(MRParams* params);
      void record_failure(MRParams* params);

      void import_params(MRParams* params);
      void print_possible_params();
  };
}


#endif

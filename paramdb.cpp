#include "paramdb.hpp"

namespace tfk {

  ParamDB::ParamDB (MRParams* default_params, MRParams* min_params, MRParams* max_params) {
    this->default_params = default_params;
    this->min_params = min_params;
    this->max_params = max_params;
    this->mutex = new std::mutex();
    possible_params.push_back(default_params);
    possible_params.push_back(min_params);
    possible_params.push_back(max_params);
  }

  MRParams* ParamDB::get_params_for_accuracy(float accuracy) {
    mutex->lock();
    MRParams* ret = this->possible_params[0];

    bool removed = false;
    for (int i = 0; i < possible_params.size(); i++) {
      if (possible_params[i]->success_count+possible_params[i]->failure_count >= 200) {
        float acc = possible_params[i]->get_accuracy();
        float cst = possible_params[i]->get_cost();
        for (int j = 0; j < possible_params.size(); j++) {
          if (j==i) continue;
          if (possible_params[j]->success_count + possible_params[j]->failure_count < 200) continue;
          if (acc <= possible_params[j]->get_accuracy() && cst >= possible_params[j]->get_cost()) {
            possible_params[i] = possible_params[possible_params.size()-1];
            possible_params.pop_back();
            removed = true;
            break;
          }
        }
        if (removed) break;
      }
    }



    // Find the lowest cost MRParams with greater than given accuracy.
    for (int i = 0; i < possible_params.size(); i++) {
      MRParams* params = possible_params[i];
      if (params->get_accuracy() >= accuracy) {
        if (params->get_cost() < ret->get_cost() || ret->get_accuracy() < accuracy) {
          ret = params;  
        }
      }
    }

    if (ret->get_accuracy() > accuracy+0.001) {
      MRParams* direction = NULL;
      // find the parameters with largest accuracy less than accuracy+0.01
      for (int i = 0; i < possible_params.size(); i++) {
        if (possible_params[i]->get_accuracy() <= accuracy+0.01) {
          if (direction == NULL || possible_params[i]->get_accuracy() > direction->get_accuracy()) {
            direction = possible_params[i]; 
          }
        }
      }
    

    if (direction != NULL) {
      float accuracy_difference = ret->get_accuracy() - direction->get_accuracy();
      //printf("adjusting params accuracy diff %f\n", accuracy_difference);
      std::vector<std::string> fparams = ret->float_params();
      std::vector<std::string> iparams = ret->int_params();
      int param_length = fparams.size()+iparams.size();
      int num_to_change = rand()%param_length;
      MRParams* new_params = new MRParams();

      for (int i = 0; i < fparams.size(); i++) {
        bool change = (rand()%param_length) < num_to_change;
        float p1 = ret->get_float_param(fparams[i]);
        if (change) {
          float p2 = direction->get_float_param(fparams[i]);
          float maxp = std::max(p1,p2);
          float minp = std::min(p1,p2);
          float newp;
          if (std::abs(maxp-minp) < 0.0001 || true) {
            float delta_step =
                (max_params->get_float_param(fparams[i]) - min_params->get_float_param(fparams[i])) /
                20;
            if (rand()%2 == 0) {
              newp = p1 + delta_step;
            } else {
              newp = p1 - delta_step;
            }
          } else {
            newp = minp+(maxp-minp)*0.5;
          }
          if (newp < min_params->get_float_param(fparams[i])) {
            newp = min_params->get_float_param(fparams[i]);
          }
          if (newp > max_params->get_float_param(fparams[i])) {
            newp = max_params->get_float_param(fparams[i]);
          }
          new_params->put_float_param(fparams[i], newp);
        } else {
          new_params->put_float_param(fparams[i], p1);
        }
      }

      for (int i = 0; i < iparams.size(); i++) {
        bool change = (rand()%param_length) < num_to_change;
        int p1 = ret->get_int_param(iparams[i]);
        if (change) {
          int p2 = direction->get_int_param(iparams[i]);
          int maxp = std::max(p1,p2);
          int minp = std::min(p1,p2);
          int newp;
          if (maxp == minp || true) {
            if (rand()%2==0) {
              newp = p1 + 1;
            } else {
              newp = p1 - 1;
            }
          } else {
            newp = minp+(maxp-minp)*0.5;
          }
          if (newp < min_params->get_int_param(iparams[i])) {
            newp = min_params->get_int_param(iparams[i]);
          }
          if (newp > max_params->get_int_param(iparams[i])) {
            newp = max_params->get_int_param(iparams[i]);
          }
          new_params->put_int_param(iparams[i], newp);
        } else {
          new_params->put_int_param(iparams[i], p1);
        }
      }


      

      new_params->set_accuracy(accuracy+0.0001);
      float new_cost =
          std::pow(new_params->get_float_param("scale_x")*new_params->get_float_param("scale_y"), 0.5);
      new_params->set_cost(new_cost);
      if (new_cost < ret->get_cost()+0.0001 || ret->success_count+ret->failure_count < 200) { 
        possible_params.push_back(new_params);
        ret = new_params;
      }
    }
  }
    mutex->unlock();
    return ret;
  }


  void ParamDB::print_possible_params() {
    for (int i = 0; i < possible_params.size(); i++) {
      possible_params[i]->print();
    }
  }

  void ParamDB::import_params(MRParams* params) {
    possible_params.push_back(params);
  }

  void ParamDB::record_success(MRParams* params) {
    mutex->lock();
    params->increment_success(); 
    mutex->unlock();
  }

  void ParamDB::record_failure(MRParams* params) {
    mutex->lock();
    params->increment_failure();
    mutex->unlock();
  }



  MRParams* ParamDB::get_max_params() {return max_params;}
  MRParams* ParamDB::get_min_params() {return min_params;}
  MRParams* ParamDB::get_default_params() {return default_params;}

// end namespace tfk
}

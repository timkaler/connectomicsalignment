#include "mrparams.hpp"

namespace tfk {

  MRParams::MRParams() { }

  float MRParams::get_float_param(std::string name) {
    return name_to_float_param[name];
  }

  int MRParams::get_int_param(std::string name) {
    return name_to_int_param[name];
  }

  void MRParams::put_float_param(std::string name, float val) {
    name_to_float_param[name] = val;
  }

  void MRParams::put_int_param(std::string name, int val) {
    name_to_int_param[name] = val;
  }

  std::vector<std::string> MRParams::float_params() {
    std::vector<std::string> ret;
    for (auto iter = name_to_float_param.begin(); iter != name_to_float_param.end();
        iter++) {
      ret.push_back(iter->first); 
    }
    return ret;
  }

  std::vector<std::string> MRParams::int_params() {
    std::vector<std::string> ret;
    for (auto iter = name_to_int_param.begin(); iter != name_to_int_param.end();
        iter++) {
      ret.push_back(iter->first); 
    }
    return ret;
  }

  float MRParams::get_accuracy() {return accuracy;}
  float MRParams::get_cost() {return cost;}

  void MRParams::set_accuracy(float val) {
    accuracy = val;
  }

  void MRParams::set_cost(float val) {
    cost = val;
  }

// end namespace tfk
}

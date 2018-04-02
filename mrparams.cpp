#include "mrparams.hpp"

namespace tfk {

  MRParams::MRParams() {
    success_count = 0;
    failure_count = 0;
  }

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

  void MRParams::increment_failure() {
    failure_count++;
    if (success_count+failure_count >= 200 && (success_count+failure_count)%200 == 0) {
      set_accuracy((1.0*success_count)/(success_count+failure_count));
    }
  }

  void MRParams::increment_success() {
    success_count++;
    if (success_count+failure_count >= 200 && (success_count+failure_count)%200 == 0) {
      set_accuracy((1.0*success_count)/(success_count+failure_count));
    }
  }

  void MRParams::set_accuracy(float val) {
    accuracy = val;
  }

  void MRParams::print() {
    if (this->failure_count+this->success_count < 200) return;
    printf("Accuracy: %f, Cost: %f, Trials: %d\n", this->get_accuracy(), this->get_cost(),
        this->failure_count+this->success_count);
    std::vector<std::string> fparams = float_params();
    std::vector<std::string> iparams = int_params();
    for (int i = 0; i < fparams.size(); i++) {
      printf("%s:%f, ", fparams[i].c_str(), get_float_param(fparams[i]));
    }
    for (int i = 0; i < iparams.size(); i++) {
      printf("%s:%d, ", iparams[i].c_str(), get_int_param(iparams[i]));
    }
    printf("\n");
  }

  void MRParams::set_cost(float val) {
    cost = val;
  }

// end namespace tfk
}

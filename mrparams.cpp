#include "mrparams.hpp"
#include "ParamsDatabase.pb.cc"
#include <math.h>

namespace tfk {

  void update_stats(stats *stat, double value) {
    stat->count++;
    double delta = value - stat->mean;
    stat->mean = stat->mean + delta / stat->count;
    double delta2 = value - stat->mean;
    stat->m2 = stat->m2 + delta * delta2;
  }

  MRParams::MRParams() {
    new_success_count = 0;
    new_failure_count = 0;
    cost = {0};
    accuracy = {0};
    this->mutex = new std::mutex();
  }

  MRParams::MRParams(Params p) {
    this->mutex = new std::mutex();
    new_success_count = p.new_success_count();
    new_failure_count = p.new_failure_count();
    cost = {p.cost_count(), p.cost_mean(), p.cost_m2()};
    accuracy = {p.accuracy_count(), p.accuracy_mean(), p.accuracy_m2()};
    for (int i = 0; i < p.float_params_size(); i++) {
      put_float_param(p.float_params(i).name(), p.float_params(i).value());
    }
    for (int i = 0; i < p.int_params_size(); i++) {
      put_int_param(p.int_params(i).name(), p.int_params(i).value());
    }
  }

  void MRParams::to_proto( Params *p) {
    p->set_accuracy_count(accuracy.count);
    p->set_accuracy_mean(accuracy.mean);
    p->set_accuracy_m2(accuracy.m2);
    p->set_cost_count(cost.count);
    p->set_cost_mean(cost.mean);
    p->set_cost_m2(cost.m2);
    p->set_new_success_count(new_success_count);
    p->set_new_failure_count(new_failure_count);
    std::vector<std::string> f_params = float_params();
    std::vector<std::string> i_params = int_params();
    for (int i = 0; i < f_params.size(); i++) {
      FloatParam *fp = p->add_float_params();
      fp->set_name(f_params[i]);
      fp->set_value(get_float_param(f_params[i]));
    }
    for (int i = 0; i < i_params.size(); i++) {
      IntParam *ip = p->add_int_params();
      ip->set_name(i_params[i]);
      ip->set_value(get_int_param(i_params[i]));
    }
  }

  float MRParams::get_float_param(std::string name) {
    return name_to_float_param[name];
  }

  int MRParams::get_int_param(std::string name) {
    return name_to_int_param[name];
  }

  void MRParams::put_float_param(std::string name, float val) {
    mutex->lock();
    name_to_float_param[name] = val;
    mutex->unlock();
  }

  void MRParams::put_int_param(std::string name, int val) {
    mutex->lock();
    name_to_int_param[name] = val;
    mutex->unlock();
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

  double MRParams::get_accuracy() {return accuracy.mean;}
  double MRParams::get_cost() {return cost.mean;}
  double MRParams::get_accuracy_std() {
    return sqrt(accuracy.m2/(accuracy.count-1));
  }
  double MRParams::get_cost_std() {
    return sqrt(cost.m2/(cost.count-1));
  }

  int MRParams::get_count() {
    return new_success_count + new_failure_count + cost.count;
  }

  void MRParams::increment_failure() {
    __sync_fetch_and_add(&new_failure_count, 1);
    if (get_count() >= update_frequency  && get_count()%update_frequency   == 0) {
      mutex->lock();
      update_stats(&accuracy, (1.0*new_success_count)/(new_success_count + new_failure_count) );
      new_failure_count = 0;
      new_success_count = 0;
      mutex->unlock();
    }
  }

  void MRParams::increment_success() {
     __sync_fetch_and_add(&new_success_count, 1);
    if (get_count() >= update_frequency && get_count()%update_frequency == 0) {
      mutex->lock();
      update_stats(&accuracy, (1.0*new_success_count)/(new_success_count + new_failure_count) );
      new_failure_count = 0;
      new_success_count = 0;
      mutex->unlock();
    }
  }

  void MRParams::set_accuracy(double val) {
    mutex->lock();
    update_stats(&accuracy, val);
    mutex->unlock();
  }

  void MRParams::print() {
    //if (this->failure_count+this->success_count < 200) return;
    printf("Accuracy: %f, std = %f, Cost: %f, std = %f, Trials: %d\n", this->get_accuracy(), this->get_accuracy_std(), this->get_cost(), this->get_cost_std(),
        this->get_count());
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

  void MRParams::set_cost(double val) {
    mutex->lock();
    update_stats(&cost, val );
    mutex->unlock();
  }

// end namespace tfk
}

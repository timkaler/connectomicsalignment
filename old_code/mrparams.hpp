
#include <string>
#include <map>
#include <vector>
#include <mutex>

#ifndef MRPARAMS
#define MRPARAMS
#include "ParamsDatabase.pb.h"

namespace tfk {

  typedef struct stats {
    int count;
    double mean;
    double m2;
  } stats;


  class MRParams {
    public:
      //enum TYPE {INT, FLOAT};
      std::map<std::string, float> name_to_float_param;
      std::map<std::string, int> name_to_int_param;
      stats cost;
      stats accuracy;

      int new_success_count;
      int new_failure_count;
      const int update_frequency = 1;

      std::mutex* mutex;

      MRParams ();
      MRParams (Params p);
      void to_proto(Params *p);
      float get_float_param(std::string name); 
      int get_int_param(std::string name);
      void put_float_param(std::string name, float val); 
      void put_int_param(std::string name, int val); 


      std::vector<std::string> float_params();
      std::vector<std::string> int_params();

      void increment_success();
      void increment_failure();

      double get_accuracy();
      double get_accuracy_std();
      double get_cost();
      double get_cost_std();
      int get_count();
      void set_accuracy(double val);
      void set_cost(double val);
      void print();
  };
}


#endif

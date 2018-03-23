
#include <string>
#include <map>
#include <vector>

#ifndef MRPARAMS
#define MRPARAMS

namespace tfk {
  class MRParams {
    public:
      //enum TYPE {INT, FLOAT};
      std::map<std::string, float> name_to_float_param;
      std::map<std::string, int> name_to_int_param;

      MRParams ();
      float get_float_param(std::string name); 
      int get_int_param(std::string name);
      void put_float_param(std::string name, float val); 
      void put_int_param(std::string name, int val); 


      std::vector<std::string> float_params();
      std::vector<std::string> int_params();
  };
}


#endif

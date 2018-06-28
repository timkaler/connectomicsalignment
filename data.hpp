#include "render.hpp"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cstdio>

#ifndef TFK_DATA
#define TFK_DATA

namespace tfk {

  class Data {
    public:
      Data();
      void sample_stack(Stack* stack, int num_samples, int box_size, std::string filename_prefix);
  };


}


#endif
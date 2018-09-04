

#ifndef TILECACHE
#define TILECACHE
#include "./stack.hpp"
#include "./cilk_tools/Graph.h"

namespace tfk {
  class TileCache {
    public:
      int max_size;
      std::vector<std::pair<std::string, cv::Mat> > tile_cache;
      TileCache (int max_size);

      bool contains(std::string);
      cv::Mat get(std::string);

            
      

  }
}


#endif

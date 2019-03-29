
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc/types_c.h>
#include <tuple>
#include <set>
#include "./stack.hpp"

#ifndef TFK_TILE_CACHE
#define TFK_TILE_CACHE

namespace tfk {
  class TileCache {
    public:

      cv::Mat** matrices;
      uint64_t* has_data;

      TileCache(int num_tiles);
      ~TileCache();
      void load_tile_image(Tile* tile);
      cv::Mat get_tile_image(Tile* tile);
      void release_tile_image(Tile* tile);
      bool has_tile_image(Tile* tile);


  };
}


#endif

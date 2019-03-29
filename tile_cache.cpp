#include "./tile_cache.hpp"

namespace tfk {

  TileCache::TileCache(int num_tiles) {
    matrices = (cv::Mat**) malloc(sizeof(cv::Mat*)*num_tiles);
    has_data = (uint64_t*) calloc(num_tiles, sizeof(uint64_t));
  }

  TileCache::~TileCache() {
    // release all tiles.
    free(matrices);
    free(has_data);
  }


  // not threadsafe, must be phased separately from reads to be correct.
  void TileCache::load_tile_image(Tile* tile) {
    if (has_data[tile->tile_id] == 1) {
      printf("Error loading a tile that's already been loaded!\n");
      return;
    }
    cv::Mat* mat = new cv::Mat();
    *mat = tile->read_tile_image();
    matrices[tile->tile_id] = mat;
    has_data[tile->tile_id] = 1;
  }

  cv::Mat TileCache::get_tile_image(Tile* tile) {
    if (!has_data[tile->tile_id]) {
      printf("Error trying to read image without first loading it into cache!\n");
      return cv::Mat();
    }
    return *(matrices[tile->tile_id]);
  }

  void TileCache::release_tile_image(Tile* tile) {
    if (!has_data[tile->tile_id]) {
      printf("Error trying to release image without first loading it into cache!\n");
      return;
    }
    matrices[tile->tile_id]->release();
    delete matrices[tile->tile_id];
    has_data[tile->tile_id] = 0;
  }

  bool TileCache::has_tile_image(Tile* tile) {
    return (has_data[tile->tile_id] == 1);
  }

} // end namespace tfk.

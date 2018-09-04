
#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <cmath>
#include "stack.hpp"

#ifndef TFK_RENDER
#define TFK_RENDER

namespace tfk {
  class Render {
    public:
      Render();
      std::pair<cv::Point2f, cv::Point2f> scale_bbox(std::pair<cv::Point2f, cv::Point2f> bbox,
                                                     cv::Point2f scale);
      cv::Point2f get_render_scale(Section* section, Resolution resolution);
      bool tile_in_render_box(Section* section, Tile* tile,
                              std::pair<cv::Point2f, cv::Point2f> bbox); 
      //cv::Mat render(Stack* stack, std::pair<cv::Point2f, cv::Point2f> bbox,
      //               Resolution resolution);
      cv::Mat render(Section* section, std::pair<cv::Point2f, cv::Point2f> bbox,
                     Resolution resolution, bool nomesh = false);
      void render(Section* section, std::pair<cv::Point2f, cv::Point2f> bbox,
                  std::string filename, Resolution res);

      void render(Section* section, std::pair<cv::Point2f, cv::Point2f> bbox,
          Resolution resolution, std::string filename);
      void render_stack(Stack* stack, std::pair<cv::Point2f, cv::Point2f> bbox,
          Resolution resolution, std::string filename_prefix); 

      void render_tile_helper(cv::Mat& tile_p_image,
          Section* section, Tile* tile,
          cv::Point2f render_scale, cv::Mat* section_p_out_sum,
          cv::Mat* section_p_out_ncount,
          int nrows, int ncols, int lower_x, int lower_y, bool nomesh,
          int start_x, int start_y, int end_x, int end_y);


      void render_stack_with_patch(Stack* stack,
          std::pair<cv::Point2f, cv::Point2f> bbox, tfk::Resolution resolution,
          std::string filename_prefix);
  };
}


#endif

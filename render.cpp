#include "render.hpp"

namespace tfk {


Render::Render() {}

void Render::render(Section* section, std::pair<cv::Point2f, cv::Point2f> bbox,
                    std::string filename, Resolution res) {
  cv::Mat img = this->render(section, bbox, res);
  cv::imwrite(filename, img);
}

bool Render::tile_in_render_box(Section* section, Tile* tile, std::pair<cv::Point2f, cv::Point2f> bbox) {
  //return tile->overlaps_with(bbox);
  std::pair<cv::Point2f, cv::Point2f> tile_bbox = tile->get_bbox();

  cv::Point2f corners[4];
  corners[0] = cv::Point2f(tile_bbox.first.x, tile_bbox.first.y);
  corners[1] = cv::Point2f(tile_bbox.second.x, tile_bbox.first.y);
  corners[2] = cv::Point2f(tile_bbox.first.x, tile_bbox.second.y);
  corners[3] = cv::Point2f(tile_bbox.second.x, tile_bbox.second.y);

  for (int i = 0; i < 4; i++) {
    corners[i] = section->affine_transform(corners[i]);
    corners[i] = section->elastic_transform(corners[i]);
  }

  float tile_min_x = corners[0].x;
  float tile_max_x = corners[0].x;
  float tile_min_y = corners[0].y;
  float tile_max_y = corners[0].y;
  for (int i = 1; i < 4; i++) {
    if (corners[i].x < tile_min_x) tile_min_x = corners[i].x;
    if (corners[i].y < tile_min_y) tile_min_y = corners[i].y;

    if (corners[i].x > tile_max_x) tile_max_x = corners[i].x;
    if (corners[i].y > tile_max_y) tile_max_y = corners[i].y;
  }

  if (tile_max_x < bbox.first.x) return false;
  if (tile_max_y < bbox.first.y) return false;
  if (tile_min_x > bbox.second.x) return false;
  if (tile_min_y > bbox.second.y) return false;

  return true;
}

cv::Point2f Render::get_render_scale(Section* section, Resolution resolution) {
  if (resolution == THUMBNAIL || resolution == THUMBNAIL2) {
    Tile* first_tile = section->tiles[0];

    //std::string thumbnailpath = std::string(first_tile->filepath);
    //thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
    //thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");

    cv::Mat thumbnail_img = first_tile->get_tile_data(THUMBNAIL);
    cv::Mat img = first_tile->get_tile_data(FULL);
    //cv::Mat thumbnail_img = cv::imread(thumbnailpath, CV_LOAD_IMAGE_UNCHANGED);
    //cv::Mat img = cv::imread(first_tile->filepath, CV_LOAD_IMAGE_UNCHANGED);

    float scale_x = (float)(img.size().width)/thumbnail_img.size().width;
    float scale_y = (float)(img.size().height)/thumbnail_img.size().height;
    thumbnail_img.release();
    img.release();
    return cv::Point2f(scale_x, scale_y);
  }

  if (resolution == FULL) {
    return cv::Point2f(1.0,1.0);
  }

  if (resolution == PERCENT30) {
    return cv::Point2f(10.0/3, 10.0/3);
  }

  return cv::Point2f(1.0,1.0);
}

std::pair<cv::Point2f, cv::Point2f> Render::scale_bbox(
    std::pair<cv::Point2f, cv::Point2f> bbox, cv::Point2f scale) {
  int lower_x = (int) (bbox.first.x/scale.x + 0.5);
  int lower_y = (int) (bbox.first.y/scale.y + 0.5);
  int upper_x = (int) (bbox.second.x/scale.x + 0.5);
  int upper_y = (int) (bbox.second.y/scale.y + 0.5);
  return std::make_pair(cv::Point2f(1.0*lower_x, 1.0*lower_y),
                        cv::Point2f(1.0*upper_x, 1.0*upper_y));
}


#define TFKMAT(matrix_ptr, rows, cols, row, col, type) ((type *) matrix_ptr->ptr(row))[col]
void tfk::Render::render_tile_helper(cv::Mat& tile_p_image,
    Section* section, Tile* tile,
    cv::Point2f render_scale, cv::Mat* section_p_out_sum,
    cv::Mat* section_p_out_ncount,
    int nrows, int ncols, int lower_x, int lower_y, bool nomesh,
    int start_x, int start_y, int end_x, int end_y) {


  bool recurse = false;
  bool fastpath = true;
  Triangle tri;
  if (end_x - start_x < 128 && end_y - start_y < 128) {
    cv::Point2f corners[4];
    corners[0] = cv::Point2f(start_x*render_scale.x, start_y*render_scale.y);
    corners[1] = cv::Point2f(end_x*render_scale.x, start_y*render_scale.y);
    corners[2] = cv::Point2f(start_x*render_scale.x, end_y*render_scale.y);
    corners[3] = cv::Point2f(end_x*render_scale.x, end_y*render_scale.y);

    tri = section->triangle_mesh->find_triangle(tile->rigid_transform(corners[0]));

    for (int i = 1; i < 4; i++) {
      bool in_tri = section->triangle_mesh->index->point_in_triangle(tile->rigid_transform(corners[i]), tri);
      if (!in_tri) {
        fastpath = false;
        recurse = true;
        break;
      }
    }
  } else {
    recurse = true;
  }

  if (recurse) {
    if (end_x - start_x > end_y - start_y) {
      if (end_x - start_x > 8) {
        int offset_x = (end_x - start_x)/2;
        render_tile_helper(tile_p_image, section, tile, render_scale, section_p_out_sum,
                           section_p_out_ncount, nrows, ncols, lower_x, lower_y, nomesh,
                           start_x, start_y, end_x-offset_x, end_y);
        render_tile_helper(tile_p_image, section, tile, render_scale, section_p_out_sum,
                           section_p_out_ncount, nrows, ncols, lower_x, lower_y, nomesh,
                           start_x+offset_x, start_y, end_x, end_y);
        return;
      }
    } else {
      if (end_y - start_y > 8) {
        int offset_y = (end_y - start_y)/2;
        render_tile_helper(tile_p_image, section, tile, render_scale, section_p_out_sum,
                           section_p_out_ncount, nrows, ncols, lower_x, lower_y, nomesh,
                           start_x, start_y, end_x, end_y-offset_y);
        render_tile_helper(tile_p_image, section, tile, render_scale, section_p_out_sum,
                           section_p_out_ncount, nrows, ncols, lower_x, lower_y, nomesh,
                           start_x, start_y+offset_y, end_x, end_y);
        return;
      }
    }
  }
    //Triangle tri_test =
    //    section->triangle_mesh->find_triangle(tile->rigid_transform(corners[i]));

    //if (tri_test.index != tri.index) {
    //  fastpath = false;
    //  break;
    //}
  //}

  for (int _y = start_y; _y < end_y; _y++) {
    for (int _x = start_x; _x < end_x; _x++) {
      cv::Point2f p = cv::Point2f(_x*render_scale.x, _y*render_scale.y);

      cv::Point2f post_rigid_p = tile->rigid_transform(p);

      cv::Point2f post_affine_p = section->affine_transform(post_rigid_p);
      cv::Point2f transformed_p = post_affine_p;
      if (!nomesh) {
        if (fastpath) {
          transformed_p = section->elastic_transform(post_affine_p, tri);
        } else {
          transformed_p = section->elastic_transform(post_affine_p);
        }
      }

      int x_c = (int)(transformed_p.x/render_scale.x + 0.5);
      int y_c = (int)(transformed_p.y/render_scale.y + 0.5);
      for (int k = -1; k < 2; k++) {
        for (int m = -1; m < 2; m++) {
          //if (k != 0 || m!=0) continue;
          unsigned char val = tile_p_image.at<unsigned char>(_y, _x);
          int x = x_c+k;
          int y = y_c+m;
          if (y-lower_y >= 0 && y-lower_y < nrows && x-lower_x >= 0 && x-lower_x < ncols) {
            //section_p_out_sum->at<unsigned short>(y-lower_y, x-lower_x) = val;
            //section_p_out_ncount->at<unsigned short>(y-lower_y, x-lower_x) = 1;
            __sync_fetch_and_add(&section_p_out_sum->at<unsigned short>(y-lower_y, x-lower_x), val);
            __sync_fetch_and_add(&section_p_out_ncount->at<unsigned short>(y-lower_y, x-lower_x),1);
          }
        }
      }
    }
  }


}

cv::Mat tfk::Render::render(Section* section, std::pair<cv::Point2f, cv::Point2f> bbox,
    tfk::Resolution resolution, bool nomesh) {

  //int bad_2d_alignment = 0;
  //for (int i = 0; i < section->tiles.size(); i++) {
  //  if (section->tiles[i]->bad_2d_alignment) bad_2d_alignment++;
  //}
  //printf("total tiles %d, bad 2d count %d\n", section->tiles.size(), bad_2d_alignment);

  cv::Point2f render_scale = this->get_render_scale(section, resolution);

  // scaled_bbox is in transformed coordinate system
  std::pair<cv::Point2f, cv::Point2f> scaled_bbox = this->scale_bbox(bbox, render_scale);

  int input_lower_x = bbox.first.x;
  int input_lower_y = bbox.first.y;
  int input_upper_x = bbox.second.x;
  int input_upper_y = bbox.second.y;

  int lower_x = scaled_bbox.first.x;
  int lower_y = scaled_bbox.first.y;
  //int upper_x = scaled_bbox.second.x;
  //int upper_y = scaled_bbox.second.y;

  int nrows = (input_upper_y-input_lower_y)/render_scale.y;
  int ncols = (input_upper_x-input_lower_x)/render_scale.x;

  // temporary matrix for the section.
  cv::Mat section_p_out;// = new cv::Mat();
  //section->p_out = new cv::Mat();
  (section_p_out).create(nrows, ncols, CV_8UC1);

  // temporary matrix for the section.
  cv::Mat* section_p_out_sum = new cv::Mat();
  //section->p_out = new cv::Mat();
  (*section_p_out_sum).create(nrows, ncols, CV_16UC1);

  // temporary matrix for the section.
  cv::Mat* section_p_out_ncount = new cv::Mat();
  //section->p_out = new cv::Mat();
  (*section_p_out_ncount).create(nrows, ncols, CV_16UC1);

  for (int y = 0; y < nrows; y++) {
    for (int x = 0; x < ncols; x++) {
      section_p_out.at<unsigned char>(y,x) = 0;
      section_p_out_sum->at<unsigned short>(y,x) = 0;
      section_p_out_ncount->at<unsigned short>(y,x) = 0;
    }
  }



      cilk_for (int i = 0; i < section->tiles.size(); i++) {
        Tile* tile = section->tiles[i];
        if (tile->bad_2d_alignment) continue;
        if (!this->tile_in_render_box(section,tile, bbox)) continue;


        //cv::Mat* tile_p_image = this->read_tile(tile->filepath, resolution);
        cv::Mat tile_p_image = tile->get_tile_data(resolution);


        render_tile_helper(tile_p_image, section, tile, render_scale, section_p_out_sum,
                           section_p_out_ncount, nrows, ncols, lower_x, lower_y, nomesh,
                           0,0,tile_p_image.size().width, tile_p_image.size().height);

        //for (int _y = 0; _y < (tile_p_image).size().height; _y++) {
        //  for (int _x = 0; _x < (tile_p_image).size().width; _x++) {
        //    if (_x > 10 && _x < tile_p_image.size().width-10 &&
        //        _y > 10 && _y < tile_p_image.size().height-10) continue;
        //    cv::Point2f p = cv::Point2f(_x*render_scale.x, _y*render_scale.y);
        //    cv::Point2f post_rigid_p = tile->rigid_transform(p);
        //    cv::Point2f transformed_p = post_rigid_p;
        //    if (!nomesh) {
        //      transformed_p = section->elastic_transform(post_rigid_p);
        //    }
        //    int x_c = (int)(transformed_p.x/render_scale.x + 0.5);
        //    int y_c = (int)(transformed_p.y/render_scale.y + 0.5);
        //    for (int k = -1; k < 2; k++) {
        //      for (int m = -1; m < 2; m++) {
        //        if (k != 0 || m!=0) continue;
        //        unsigned char val = 0;//tile_p_image.at<unsigned char>(_y, _x);
        //        int x = x_c+k;
        //        int y = y_c+m;
        //        if (y-lower_y >= 0 && y-lower_y < nrows && x-lower_x >= 0 && x-lower_x < ncols) {
        //          section_p_out_sum->at<unsigned short>(y-lower_y, x-lower_x) = val;
        //          section_p_out_ncount->at<unsigned short>(y-lower_y, x-lower_x) = 1;
        //        }
        //      }
        //    }
        //  }
        //}
        //for (int _y = 0; _y < (tile_p_image).size().height; _y++) {
        //  for (int _x = 0; _x < (tile_p_image).size().width; _x++) {
        //    cv::Point2f p = cv::Point2f(_x*render_scale.x, _y*render_scale.y);

        //    cv::Point2f post_rigid_p = tile->rigid_transform(p);

        //    cv::Point2f post_affine_p = section->affine_transform(post_rigid_p);
        //    cv::Point2f transformed_p = post_affine_p;
        //    if (!nomesh) {
        //      transformed_p = section->elastic_transform(post_affine_p);
        //    }

        //    //cv::Point2f transformed_p = affine_transform(&tile, p);
        //    //transformed_p = elastic_transform(&tile, &triangles, transformed_p);

        //    int x_c = (int)(transformed_p.x/render_scale.x + 0.5);
        //    int y_c = (int)(transformed_p.y/render_scale.y + 0.5);
        //    for (int k = -1; k < 2; k++) {
        //      for (int m = -1; m < 2; m++) {
        //        //if (k != 0 || m!=0) continue;
        //        unsigned char val = tile_p_image.at<unsigned char>(_y, _x);
        //        int x = x_c+k;
        //        int y = y_c+m;
        //        if (y-lower_y >= 0 && y-lower_y < nrows && x-lower_x >= 0 && x-lower_x < ncols) {
        //          __sync_fetch_and_add(&section_p_out_sum->at<unsigned short>(y-lower_y, x-lower_x), val);
        //          __sync_fetch_and_add(&section_p_out_ncount->at<unsigned short>(y-lower_y, x-lower_x),1);
        //          //section_p_out_sum->at<unsigned short>(y-lower_y, x-lower_x) += val;
        //          //section_p_out_ncount->at<unsigned short>(y-lower_y, x-lower_x) += 1;
        //        }
        //      }
        //    }
        //  }
        //}
        tile_p_image.release();
        tile->release_full_image();
      }
  //}

  //cilk_sync;
  cilk_for (int y = 0; y < section_p_out.size().height; y++) {
    for (int x = 0; x < section_p_out.size().width; x++) {
      if (section_p_out_ncount->at<unsigned short>(y,x) == 0) {
        continue;
      }
      section_p_out.at<unsigned char>(y, x) =
          section_p_out_sum->at<unsigned short>(y, x) / section_p_out_ncount->at<unsigned short>(y,x);
      // force the min value to be at least 1 so that we can check for out-of-range pixels.
      if (section_p_out.at<unsigned char>(y, x) == 0) {
        section_p_out.at<unsigned char>(y, x) = 1;
      }
    }
  }

  //if (write) {
  //  cv::imwrite(filename, (*section_p_out));
  //}
  section_p_out_sum->release();
  section_p_out_ncount->release();
  delete section_p_out_sum;
  delete section_p_out_ncount;
  //delete section_p_out;
  //delete section_p_out_ncount;


  return (section_p_out);
}

void Render::render(Section* section, std::pair<cv::Point2f, cv::Point2f> bbox,
    tfk::Resolution resolution, std::string filename) {
  cv::Mat img = this->render(section, bbox, resolution);
  cv::imwrite(filename, img); 
}

void Render::render_stack(Stack* stack,
    std::pair<cv::Point2f, cv::Point2f> bbox, tfk::Resolution resolution,
    std::string filename_prefix) {

  cilk_for (int i = 0; i < stack->sections.size(); i++) {
    Section* section = stack->sections[i];
    render(section, bbox, resolution,
           filename_prefix+"_"+std::to_string(section->real_section_id)+".tif"); 
  }
}

void Render::render_stack_with_patch(Stack* stack,
    std::pair<cv::Point2f, cv::Point2f> bbox, tfk::Resolution resolution,
    std::string filename_prefix) {

  if (stack->sections.size() < 2) {
    printf("Less than two images, so we're going to just render the stack without patching.\n");
    return render_stack(stack, bbox, resolution, filename_prefix);
  }

  // do the first section.
  Section* next_section = stack->sections[1];
  cv::Mat next_section_img = render(next_section, bbox, resolution);

  cv::Mat img = this->render(stack->sections[0], bbox, resolution);
  cv::Mat last_img;
  // fill the buffer.
  for (int i = 0; i < stack->sections.size(); i++) {
    Section* section = stack->sections[i];
    cv::Mat section_p_out_sum, section_p_out_count;
    section_p_out_sum.create(img.rows, img.cols, CV_16UC1);
    section_p_out_count.create(img.rows, img.cols, CV_16UC1);
    cilk_for (int c = 0; c < img.cols; c++) {
      for (int r = 0; r < img.rows; r++) {
        section_p_out_sum.at<unsigned short>(r,c) = 0;
        section_p_out_count.at<unsigned short>(r,c) = 0;
      }
    }
    // now scan the img and get data from section above.
    cilk_for (int c = 3; c < img.cols-3; c++) {
      for (int r = 3; r < img.rows-3; r++) {
        if (img.at<unsigned char>(r,c) == 0) {
          for (int dc = -2; dc < 3; dc++) {
            for (int dr = -2; dr < 3; dr++) {
              if (next_section_img.at<unsigned char>(r+dr,c+dc) != 0) {
                section_p_out_sum.at<unsigned short>(r,c) += next_section_img.at<unsigned char>(r+dr,c+dc);
                section_p_out_count.at<unsigned short>(r,c) += 1;
              }
              if (i > 0 && last_img.at<unsigned char>(r+dr,c+dc) != 0) {
                section_p_out_sum.at<unsigned short>(r,c) += last_img.at<unsigned char>(r+dr, c+dc);
                section_p_out_count.at<unsigned short>(r,c) += 1;
              }
              if (img.at<unsigned char>(r,c) != 0) {
                section_p_out_sum.at<unsigned short>(r,c) += img.at<unsigned char>(r+dr, c+dc);
                section_p_out_count.at<unsigned short>(r,c) += 1;
              }
            }
          }
        }
        if (i > 0 && img.at<unsigned char>(r,c) == 0) {
          img.at<unsigned char>(r,c) = last_img.at<unsigned char>(r,c);
        }
      }
    }
    cilk_for (int c = 3; c < img.cols-3; c++) {
      for (int r = 3; r < img.rows-3; r++) {
        if (section_p_out_count.at<unsigned short>(r,c) > 0) {
          img.at<unsigned char>(r,c) = section_p_out_sum.at<unsigned short>(r,c) / section_p_out_count.at<unsigned short>(r,c);
        }
      }
    }
    cv::imwrite(filename_prefix+"_"+std::to_string(section->real_section_id)+".tif", img);
    last_img = img;
    img = next_section_img;

    if (i+1 < stack->sections.size()) {
      next_section_img = render(stack->sections[i+1], bbox, resolution);
    }

    //render(section, bbox, resolution,
    //       filename_prefix+"_"+std::to_string(section->real_section_id)+".tif"); 
  }


  for (int i = 0; i < stack->sections.size(); i++) {
    Section* section = stack->sections[i];
    render(section, bbox, resolution,
           filename_prefix+"_"+std::to_string(section->real_section_id)+".tif"); 
  }
}
// end namespace tfk
}

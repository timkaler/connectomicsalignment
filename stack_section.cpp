
#include "stack_helpers.cpp"

tfk::Section::Section(int section_id) {
  this->section_id = section_id;
}


cv::Point2f tfk::Section::affine_transform(cv::Point2f pt) {
  float new_x = pt.x*this->a00 + pt.y * this->a01 + this->offset_x;
  float new_y = pt.x*this->a10 + pt.y * this->a11 + this->offset_y;
  return cv::Point2f(new_x, new_y);
}

//void render_section(double min_x, double min_y, double max_x, double max_y) {
//}

// need to implement tile intersects with bounding box.
// need to implement render_replacement_tile

bool tfk::Section::transformed_tile_overlaps_with(Tile* tile,
    std::pair<cv::Point2f, cv::Point2f> bbox) {
  auto tile_bbox = tile->get_bbox();
  tile_bbox = this->affine_transform_bbox(tile_bbox);
  tile_bbox = this->elastic_transform_bbox(tile_bbox);

  int x1_start = tile_bbox.first.x;
  int x1_finish = tile_bbox.second.x;
  int y1_start = tile_bbox.first.y;
  int y1_finish = tile_bbox.second.y;

  int x2_start = bbox.first.x;
  int x2_finish = bbox.second.x;
  int y2_start = bbox.first.y;
  int y2_finish = bbox.second.y;

  bool res = false;
  if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
      (y1_start < y2_finish) && (y1_finish > y2_start)) {
      res = true;
  }
  return res;

}


void tfk::Section::replace_bad_region(std::pair<cv::Point2f, cv::Point2f> bad_bbox,
                                     Section* other_neighbor) {
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    //if (tile->overlaps_with(bad_bbox)) {
    if (this->transformed_tile_overlaps_with(tile, bad_bbox)) {
      // check to make sure this tile hasn't already been replaced.
      if (this->replaced_tile_ids.find(i) == this->replaced_tile_ids.end()) {
        this->replaced_tile_ids.insert(i);
        printf("Replacing tile in section %d with tile_id %d\n", this->real_section_id, i);
        this->replace_bad_tile(tile, other_neighbor);
      }
    }
  }
}

void tfk::Section::replace_bad_tile(Tile* tile, Section* other_neighbor) {

  auto bbox = tile->get_bbox();
  bbox = this->affine_transform_bbox(bbox);
  bbox = this->elastic_transform_bbox(bbox);

  float slack = 4000.0;
  bbox.first.x -= slack;
  bbox.first.y -= slack;
  bbox.second.x += slack;
  bbox.second.y += slack;

{
  // full resolution
  cv::Mat halo = other_neighbor->render(bbox, FULL);

  cv::Mat tile_img = cv::imread(tile->filepath, CV_LOAD_IMAGE_UNCHANGED);

  imwrite("orig_tiles/sec_"+std::to_string(this->real_section_id) +
          "_tileid_" + std::to_string(tile->tile_id) +".bmp", tile_img);

  for (int y = 0; y < tile_img.rows; y++) {
    for (int x = 0; x < tile_img.cols; x++) {
      cv::Point2f pt = cv::Point2f(x,y);
      pt = tile->rigid_transform(pt);
      pt = this->affine_transform(pt);
      pt = this->elastic_transform(pt);
      uint8_t halo_val = halo.at<uint8_t>((int)(pt.y - bbox.first.y), (int)(pt.x - bbox.first.x));
      if (halo_val == 0) {
        printf("Halo value 0 detected, skipping this one.\n");
        return;
      }
      tile_img.at<uint8_t>(y,x) = halo_val;
    }
  }

  imwrite("new_tiles/sec_"+std::to_string(this->real_section_id) +
          "_tileid_" + std::to_string(tile->tile_id) +".bmp", tile_img);
}

{
  cv::Point2f render_scale = this->get_render_scale(THUMBNAIL);

  std::string thumbnailpath = std::string(tile->filepath);
  thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
  thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");

  // thumbnail resolution
  cv::Mat halo = other_neighbor->render(bbox, THUMBNAIL);

  cv::Mat tile_img = tile->get_tile_data(THUMBNAIL);
  //cv::Mat tile_img = cv::imread(thumbnailpath, CV_LOAD_IMAGE_GRAYSCALE);

  imwrite("orig_tiles/thumbnail_sec_"+std::to_string(this->real_section_id) +
          "_tileid_" + std::to_string(tile->tile_id) +".jpg", tile_img);

  for (int y = 0; y < tile_img.rows; y++) {
    for (int x = 0; x < tile_img.cols; x++) {
      cv::Point2f pt = cv::Point2f(x*render_scale.x,y*render_scale.y);
      pt = tile->rigid_transform(pt);
      pt = this->affine_transform(pt);
      pt = this->elastic_transform(pt);
      uint8_t halo_val = halo.at<uint8_t>((int)((pt.y - bbox.first.y)/render_scale.y), (int)((pt.x - bbox.first.x)/render_scale.x));
      if (halo_val == 0) {
        printf("Halo value 0 detected, skipping this one.\n");
        return;
      }
      tile_img.at<uint8_t>(y,x) = halo_val;
    }
  }

  imwrite("new_tiles/thumbnail_sec_"+std::to_string(this->real_section_id) +
          "_tileid_" + std::to_string(tile->tile_id) +".jpg", tile_img);
}

  tile->filepath = "new_tiles/sec_"+std::to_string(this->real_section_id) +
          "_tileid_" + std::to_string(tile->tile_id) +".bmp";
  tile->image_data_replaced = true;


}


void tfk::Section::render_error(Section* neighbor, Section* other_neighbor, Section* other2_neighbor,
    std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix) {
  cv::Mat n_image = neighbor->render(bbox, THUMBNAIL);
  cv::Mat other_n_image = other_neighbor->render(bbox, THUMBNAIL);
  cv::Mat other2_n_image = other2_neighbor->render(bbox, THUMBNAIL);
  cv::Mat my_image = this->render(bbox, THUMBNAIL);

  int nrows = n_image.rows;
  int ncols = n_image.cols;

  cv::Mat heat_map;
  cv::Mat n_patch;
  cv::Mat other_n_patch;
  cv::Mat other2_n_patch;
  cv::Mat my_patch;


  int patch_3_size = 100;
  int patch_2_size = 20;


  n_patch.create(patch_2_size, patch_2_size, CV_8UC1);
  other_n_patch.create(patch_2_size, patch_2_size, CV_8UC1);
  other2_n_patch.create(patch_2_size, patch_2_size, CV_8UC1);
  my_patch.create(patch_2_size, patch_2_size, CV_8UC1);
  heat_map.create(nrows, ncols, CV_32F);

  std::vector<std::pair<cv::Point2f, float> > patch_results;

  for (int y = 0; y < nrows; y++) {
    for (int x = 0; x < ncols; x++) {
      heat_map.at<float>(y,x) = 0.0;
    }
  }


  cv::Point2f render_scale = this->get_render_scale(THUMBNAIL);




  for (int by = 0; by + patch_3_size < nrows; by += patch_3_size/2) {
    for (int bx = 0; bx + patch_3_size < ncols; bx += patch_3_size/2) {
      int bad = 0;
      int total = 0;
      bool skip = false;
      int bad_above = 0;
      for (int y = 0; y + patch_2_size < patch_3_size; y += patch_2_size/2) {
        for (int x = 0; x + patch_2_size < patch_3_size; x += patch_2_size/2) {
          for (int _y = 0; _y < patch_2_size; _y++) {
            for (int _x = 0; _x < patch_2_size; _x++) {
              other_n_patch.at<uint8_t>(_y, _x) = other_n_image.at<uint8_t>(by+y+_y, bx+x+_x);
              other2_n_patch.at<uint8_t>(_y, _x) = other2_n_image.at<uint8_t>(by+y+_y, bx+x+_x);
              n_patch.at<uint8_t>(_y, _x) = n_image.at<uint8_t>(by+y+_y, bx+x+_x);
              my_patch.at<uint8_t>(_y, _x) = my_image.at<uint8_t>(by+y+_y, bx+x+_x);
              if (n_patch.at<uint8_t>(_y,_x) == 0 ||
                  my_patch.at<uint8_t>(_y,_x) == 0 ||
                  other_n_patch.at<uint8_t>(_y,_x) == 0 ||
                  other2_n_patch.at<uint8_t>(_y,_x) == 0) {
                skip = true;
              }
            }
          }

          cv::Mat result;
          cv::matchTemplate(n_patch, my_patch, result, CV_TM_CCOEFF_NORMED);
          float corr = result.at<float>(0,0);

          cv::Mat other_result;
          cv::matchTemplate(other_n_patch, my_patch, other_result, CV_TM_CCOEFF_NORMED);
          float other_corr = other_result.at<float>(0,0);

          cv::Mat other2_result;
          cv::matchTemplate(other2_n_patch, other_n_patch, other2_result, CV_TM_CCOEFF_NORMED);
          float other2_corr = other2_result.at<float>(0,0);


          if (corr < 0.1 && other_corr < 0.1 && other2_corr > 0.1) {
            bad++;
          }

          if (other2_corr < 0.1) {
            bad_above++;
          }

          total++;

          //for (int _y = 0; _y < 10; _y++) {
          //  for (int _x = 0; _x < 10; _x++) {
          //    if (corr < 0.1) {
          //      heat_map.at<float>(y+_y, x+_x) = 1.0;
          //    } else {
          //      //heat_map.at<float>(y+_y, x+_x) = 0.0;
          //    }
          //  }
          //}
        }
      }

      if (bad > total/10 && !skip && bad_above < total/10) {
        for (int y = 0; y < patch_3_size; y++) {
          for (int x = 0; x < patch_3_size; x++) {
            heat_map.at<float>(by+y, bx+x) = 1.0;
          }
        }

        int bad_min_x = bbox.first.x + (bx)*render_scale.x;
        int bad_max_x = bbox.first.x + (bx+patch_3_size)*render_scale.x;

        int bad_min_y = bbox.first.y + (by)*render_scale.y;
        int bad_max_y = bbox.first.y + (by+patch_3_size)*render_scale.y;

        auto bad_bbox = std::make_pair(cv::Point2f(bad_min_x, bad_min_y),
                                   cv::Point2f(bad_max_x, bad_max_y));

        this->replace_bad_region(bad_bbox, other_neighbor);
      }



    }
  }

  cv::Mat heatmap = apply_heatmap_to_grayscale(&my_image, &heat_map, nrows, ncols);
  imwrite(filename_prefix, heatmap);
}

bool tfk::Section::section_data_exists() {
  std::string filename =
      std::string("newcached_data/prefix_"+std::to_string(this->real_section_id));


  cv::FileStorage fs(filename+std::string("_3d_keypoints.yml.gz"), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  } else {
    return true;
  }
}

cv::Point2f tfk::Section::get_render_scale(Resolution resolution) {
  if (resolution == THUMBNAIL) {
    Tile* first_tile = this->tiles[0];

    std::string thumbnailpath = std::string(first_tile->filepath);
    thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
    thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");

    cv::Mat thumbnail_img = first_tile->get_tile_data(THUMBNAIL);
    cv::Mat img = first_tile->get_tile_data(FULL);
    //cv::Mat thumbnail_img = cv::imread(thumbnailpath, CV_LOAD_IMAGE_UNCHANGED);
    //cv::Mat img = cv::imread(first_tile->filepath, CV_LOAD_IMAGE_UNCHANGED);

    float scale_x = (float)(img.size().width)/thumbnail_img.size().width;
    float scale_y = (float)(img.size().height)/thumbnail_img.size().height;

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

std::pair<cv::Point2f, cv::Point2f> tfk::Section::scale_bbox(
    std::pair<cv::Point2f, cv::Point2f> bbox, cv::Point2f scale) {
  int lower_x = (int) (bbox.first.x/scale.x + 0.5);
  int lower_y = (int) (bbox.first.y/scale.y + 0.5);
  int upper_x = (int) (bbox.second.x/scale.x + 0.5);
  int upper_y = (int) (bbox.second.y/scale.y + 0.5);
  return std::make_pair(cv::Point2f(1.0*lower_x, 1.0*lower_y),
                        cv::Point2f(1.0*upper_x, 1.0*upper_y));
}

bool tfk::Section::tile_in_render_box(Tile* tile, std::pair<cv::Point2f, cv::Point2f> bbox) {

  //return tile->overlaps_with(bbox);

  std::pair<cv::Point2f, cv::Point2f> tile_bbox = tile->get_bbox();

  cv::Point2f corners[4];
  corners[0] = cv::Point2f(tile_bbox.first.x, tile_bbox.first.y);
  corners[1] = cv::Point2f(tile_bbox.second.x, tile_bbox.first.y);
  corners[2] = cv::Point2f(tile_bbox.first.x, tile_bbox.second.y);
  corners[3] = cv::Point2f(tile_bbox.second.x, tile_bbox.second.y);

  for (int i = 0; i < 4; i++) {
    corners[i] = this->affine_transform(corners[i]);
    corners[i] = this->elastic_transform(corners[i]);
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


//cv::Mat* tfk::Section::read_tile(std::string filepath, Resolution res) {
//  cv::Mat* tile_p_image = new cv::Mat();
//  if(res == THUMBNAIL) {
//    std::string path = std::string(filepath);
//    path = path.replace(path.find(".bmp"), 4,".jpg");
//    path = path.insert(path.find_last_of("/") + 1, "thumbnail_");
//    (*tile_p_image) = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
//  }
//  if(res == FULL) {
//    (*tile_p_image) = cv::imread(filepath, CV_LOAD_IMAGE_UNCHANGED);
//  }
//  return tile_p_image;
//}

renderTriangle tfk::Section::getRenderTriangle(tfkTriangle tri) {
  renderTriangle rTri;
  rTri.p[0] = (*(this->mesh_orig))[tri.index1];
  rTri.p[1] = (*(this->mesh_orig))[tri.index2];
  rTri.p[2] = (*(this->mesh_orig))[tri.index3];

  rTri.q[0] = (*(this->mesh))[tri.index1];
  rTri.q[1] = (*(this->mesh))[tri.index2];
  rTri.q[2] = (*(this->mesh))[tri.index3];
  return rTri;
}

std::tuple<bool, float, float, float> tfk::Section::get_triangle_for_point(cv::Point2f pt) {
  for (int i = 0; i < this->triangles->size(); i++) {
    renderTriangle rTri = this->getRenderTriangle((*this->triangles)[i]);
    float u,v,w;
    cv::Point2f a,b,c;
    a = rTri.p[0];
    b = rTri.p[1];
    c = rTri.p[2];

    Barycentric(pt, a,b,c,u,v,w);
    if (u >=0 && v>=0 && w >= 0) {
      int j = i;
      while (j > 0) {
        tfkTriangle tmp = (*this->triangles)[j-1];
        (*this->triangles)[j-1] = (*this->triangles)[j];
        (*this->triangles)[j] = tmp;
        j--;
      }
      //printf("found the triangle\n");
      return std::make_tuple(true, u,v,w);
    }
  }
  //printf("didn't find the triangle\n");
  return std::make_tuple(false, -1, -1, -1);
}

// assumes point p is post section-global affine.
cv::Point2f tfk::Section::elastic_transform(cv::Point2f p) {
  std::tuple<bool, float, float, float> info = this->get_triangle_for_point(p);
  if (!std::get<0>(info)) return p;

  renderTriangle tri = this->getRenderTriangle((*this->triangles)[0]);
  float u = std::get<1>(info);
  float v = std::get<2>(info);
  float w = std::get<3>(info);
  float new_x = u*tri.q[0].x + v*tri.q[1].x + w*tri.q[2].x;
  float new_y = u*tri.q[0].y + v*tri.q[1].y + w*tri.q[2].y;
  return cv::Point2f(new_x, new_y);
}

// bbox is in unscaled (i.e. full resolution) transformed coordinate system.
cv::Mat tfk::Section::render(std::pair<cv::Point2f, cv::Point2f> bbox,
    tfk::Resolution resolution) {
  cv::Point2f render_scale = this->get_render_scale(resolution);

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
  cv::Mat* section_p_out = new cv::Mat();
  //section->p_out = new cv::Mat();
  (*section_p_out).create(nrows, ncols, CV_8UC1);

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
      section_p_out->at<unsigned char>(y,x) = 0;
      section_p_out_sum->at<unsigned short>(y,x) = 0;
      section_p_out_ncount->at<unsigned short>(y,x) = 0;
    }
  }

  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    if (!this->tile_in_render_box(tile, bbox)) continue;

    //cv::Mat* tile_p_image = this->read_tile(tile->filepath, resolution);
    cv::Mat tile_p_image = tile->get_tile_data(resolution);

    for (int _y = 0; _y < (tile_p_image).size().height; _y++) {
      for (int _x = 0; _x < (tile_p_image).size().width; _x++) {
        cv::Point2f p = cv::Point2f(_x*render_scale.x, _y*render_scale.y);

        cv::Point2f post_rigid_p = tile->rigid_transform(p);

        cv::Point2f post_affine_p = this->affine_transform(post_rigid_p);

        cv::Point2f transformed_p = this->elastic_transform(post_affine_p);

        //cv::Point2f transformed_p = affine_transform(&tile, p);
        //transformed_p = elastic_transform(&tile, &triangles, transformed_p);

        int x_c = (int)(transformed_p.x/render_scale.x + 0.5);
        int y_c = (int)(transformed_p.y/render_scale.y + 0.5);
        for (int k = -1; k < 2; k++) {
          for (int m = -1; m < 2; m++) {
            unsigned char val = tile_p_image.at<unsigned char>(_y, _x);
            int x = x_c+k;
            int y = y_c+m;
            if (y-lower_y >= 0 && y-lower_y < nrows && x-lower_x >= 0 && x-lower_x < ncols) {
              section_p_out_sum->at<unsigned short>(y-lower_y, x-lower_x) += val;
              section_p_out_ncount->at<unsigned short>(y-lower_y, x-lower_x) += 1;
            }
          }
        }
      }
    }
    tile_p_image.release();
  }

  for (int y = 0; y < section_p_out->size().height; y++) {
    for (int x = 0; x < section_p_out->size().width; x++) {
      if (section_p_out_ncount->at<unsigned short>(y,x) == 0) {
        continue;
      }
      section_p_out->at<unsigned char>(y, x) =
          section_p_out_sum->at<unsigned short>(y, x) / section_p_out_ncount->at<unsigned short>(y,x);
      // force the min value to be at least 1 so that we can check for out-of-range pixels.
      if (section_p_out->at<unsigned char>(y, x) == 0) {
        section_p_out->at<unsigned char>(y, x) = 1;
      }
    }
  }

  //if (write) {
  //  cv::imwrite(filename, (*section_p_out));
  //}
  return (*section_p_out);
}




void tfk::Section::render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename,
    Resolution res) {
  cv::Mat img = this->render(bbox, res);
  cv::imwrite(filename, img);
}

std::pair<cv::Point2f, cv::Point2f> tfk::Section::elastic_transform_bbox(
    std::pair<cv::Point2f, cv::Point2f> bbox) {
  cv::Point2f corners[4];
  corners[0] = cv::Point2f(bbox.first.x, bbox.first.y);
  corners[1] = cv::Point2f(bbox.second.x, bbox.first.y);
  corners[2] = cv::Point2f(bbox.first.x, bbox.second.y);
  corners[3] = cv::Point2f(bbox.second.x, bbox.second.y);

  for (int i = 0; i < 4; i++) {
    corners[i] = this->elastic_transform(corners[i]);
  }

  float min_x = corners[0].x;
  float max_x = corners[0].x;
  float min_y = corners[0].y;
  float max_y = corners[0].y;
  for (int i = 1; i < 4; i++) {
    if (corners[i].x < min_x) min_x = corners[i].x;
    if (corners[i].x > max_x) max_x = corners[i].x;

    if (corners[i].y < min_y) min_y = corners[i].y;
    if (corners[i].y > max_y) max_y = corners[i].y;
  }
  return std::make_pair(cv::Point2f(min_x,min_y), cv::Point2f(max_x, max_y));
}


std::pair<cv::Point2f, cv::Point2f> tfk::Section::affine_transform_bbox(
    std::pair<cv::Point2f, cv::Point2f> bbox) {
  cv::Point2f corners[4];
  corners[0] = cv::Point2f(bbox.first.x, bbox.first.y);
  corners[1] = cv::Point2f(bbox.second.x, bbox.first.y);
  corners[2] = cv::Point2f(bbox.first.x, bbox.second.y);
  corners[3] = cv::Point2f(bbox.second.x, bbox.second.y);

  for (int i = 0; i < 4; i++) {
    corners[i] = this->affine_transform(corners[i]);
  }

  float min_x = corners[0].x;
  float max_x = corners[0].x;
  float min_y = corners[0].y;
  float max_y = corners[0].y;
  for (int i = 1; i < 4; i++) {
    if (corners[i].x < min_x) min_x = corners[i].x;
    if (corners[i].x > max_x) max_x = corners[i].x;

    if (corners[i].y < min_y) min_y = corners[i].y;
    if (corners[i].y > max_y) max_y = corners[i].y;
  }
  return std::make_pair(cv::Point2f(min_x,min_y), cv::Point2f(max_x, max_y));
}


void tfk::Section::affine_transform_keypoints(std::vector<cv::KeyPoint>& keypoints) {
  for (int i = 0; i < keypoints.size(); i++) {
    keypoints[i].pt = this->affine_transform(keypoints[i].pt);
  }
}

void tfk::Section::get_elastic_matches_one(Section* neighbor) {

  double ransac_thresh = 128.0;


  this->section_mesh_matches.clear();

  // Determine a good bounding box.
  std::pair<cv::Point2f, cv::Point2f> bbox = this->get_bbox();

  // bbox is before the affine transform so I need to recompute it.
  bbox = this->affine_transform_bbox(bbox);


  double min_x = bbox.first.x;
  double max_x = bbox.second.x;
  double min_y = bbox.first.y;
  double max_y = bbox.second.y;


  std::vector <cv::KeyPoint > atile_all_kps;
  std::vector <cv::Mat > atile_all_kps_desc;

  std::vector <cv::KeyPoint > btile_all_kps;
  std::vector <cv::Mat > btile_all_kps_desc;

  for (int i = 0; i < this->tiles.size(); i++) {
    this->tiles[i]->get_3d_keypoints(atile_all_kps, atile_all_kps_desc);
  }

  for (int i = 0; i < neighbor->tiles.size(); i++) {
    neighbor->tiles[i]->get_3d_keypoints(btile_all_kps, btile_all_kps_desc);
  }

  this->affine_transform_keypoints(atile_all_kps);
  neighbor->affine_transform_keypoints(btile_all_kps);

  std::vector< cv::Point2f > filtered_match_points_a(0);
  std::vector< cv::Point2f > filtered_match_points_b(0);

  for (double box_iter_x = min_x; box_iter_x < max_x + 48000; box_iter_x += 24000) {
    for (double box_iter_y = min_y; box_iter_y < max_y + 48000; box_iter_y += 24000) {
      // Filter the matches with RANSAC
      int num_filtered = 0;
      std::vector<cv::Point2f> match_points_a, match_points_b;
      double box_min_x = box_iter_x;
      double box_max_x = box_iter_x+48000;
      double box_min_y = box_iter_y;
      double box_max_y = box_iter_y+48000;

      std::vector <cv::KeyPoint > atile_kps_in_overlap;
      std::vector <cv::Mat > atile_kps_desc_in_overlap_list;
      std::vector<int> atile_kps_tile_list;
      std::vector <cv::KeyPoint > btile_kps_in_overlap;
      std::vector <cv::Mat > btile_kps_desc_in_overlap_list;
      std::vector<int> btile_kps_tile_list;

      std::vector<double> atile_weights;
      std::vector<double> btile_weights;

      // filter out any keypoints that are not inside this iterations box.
      for (int i = 0; i < atile_all_kps.size(); i++) {
        if (atile_all_kps[i].pt.x < box_min_x) continue;
        if (atile_all_kps[i].pt.x > box_max_x) continue;
        if (atile_all_kps[i].pt.y < box_min_y) continue;
        if (atile_all_kps[i].pt.y > box_max_y) continue;
        atile_kps_in_overlap.push_back(atile_all_kps[i]);
        atile_kps_desc_in_overlap_list.push_back(atile_all_kps_desc[i]);
      }
      for (int i = 0; i < btile_all_kps.size(); i++) {
        if (btile_all_kps[i].pt.x < box_min_x) continue;
        if (btile_all_kps[i].pt.x > box_max_x) continue;
        if (btile_all_kps[i].pt.y < box_min_y) continue;
        if (btile_all_kps[i].pt.y > box_max_y) continue;
        btile_kps_in_overlap.push_back(btile_all_kps[i]);
        btile_kps_desc_in_overlap_list.push_back(btile_all_kps_desc[i]);
      }

      if (atile_kps_in_overlap.size() < 4 || btile_kps_in_overlap.size() < 4) continue;

      // Now do the matching.
      cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
      cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
      cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));

      std::vector< cv::DMatch > matches;
      match_features(matches,
                     atile_kps_desc_in_overlap,
                     btile_kps_desc_in_overlap,
                     0.92);
      //printf("Done with the matching. Num matches is %lu\n", matches.size());
      if (matches.size() == 0) continue;

      for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
        match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
        match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
      }

      bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
      tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, ransac_thresh, mask);
      for (int c = 0; c < match_points_a.size(); c++) {
        if (mask[c]) {
          num_filtered++;
        }
      }

      if (num_filtered < match_points_a.size()*0.2 || num_filtered < 12) {
        free(mask);
        continue;
      }

      for (int c = 0; c < match_points_a.size(); c++) {
        if (mask[c]) {
          filtered_match_points_a.push_back(
              match_points_a[c]);
          filtered_match_points_b.push_back(
              match_points_b[c]);
        }
      }
      free(mask);
    }
  }

  for (int m = 0; m < filtered_match_points_a.size(); m++) {
    cv::Point2f my_pt = filtered_match_points_a[m];
    cv::Point2f n_pt = filtered_match_points_b[m];

    tfkMatch match;
    // find the triangle...
    std::vector<tfkTriangle>* triangles = this->triangles;
    std::vector<cv::Point2f>* mesh = this->mesh;

    std::vector<tfkTriangle>* n_triangles = neighbor->triangles;
    std::vector<cv::Point2f>* n_mesh = neighbor->mesh;


    int my_triangle_index = -1;
    int n_triangle_index = -1;
    for (int s = 0; s < triangles->size(); s++) {
      float u,v,w;
      cv::Point2f pt1 = (*mesh)[(*triangles)[s].index1];
      cv::Point2f pt2 = (*mesh)[(*triangles)[s].index2];
      cv::Point2f pt3 = (*mesh)[(*triangles)[s].index3];
      Barycentric(my_pt, pt1,pt2,pt3,u,v,w);
      if (u <= 0 || v <= 0 || w <= 0) continue;
      my_triangle_index = s;
      match.my_tri = (*triangles)[my_triangle_index];
      match.my_barys[0] = (double)1.0*u;
      match.my_barys[1] = (double)1.0*v;
      match.my_barys[2] = (double)1.0*w;
      break;
    }

    for (int s = 0; s < n_triangles->size(); s++) {
      float u,v,w;
      cv::Point2f pt1 = (*n_mesh)[(*n_triangles)[s].index1];
      cv::Point2f pt2 = (*n_mesh)[(*n_triangles)[s].index2];
      cv::Point2f pt3 = (*n_mesh)[(*n_triangles)[s].index3];
      Barycentric(n_pt, pt1,pt2,pt3,u,v,w);
      if (u <= 0 || v <= 0 || w <= 0) continue;
      n_triangle_index = s;
      match.n_tri = (*n_triangles)[n_triangle_index];
      match.n_barys[0] = (double)1.0*u;
      match.n_barys[1] = (double)1.0*v;
      match.n_barys[2] = (double)1.0*w;
      break;
    }
    if (my_triangle_index == -1 || n_triangle_index == -1) continue;
    //match.my_section_data = *section_data_a;
    //match.n_section_data = *section_data_b;
    match.my_section = (void*) this;
    match.n_section = (void*) neighbor;

    this->section_mesh_matches.push_back(match);
  }


}

void tfk::Section::get_elastic_matches(std::vector<Section*> neighbors) {
  for (int i = 0; i < neighbors.size(); i++) {
    this->get_elastic_matches_one(neighbors[i]);
  }
}


std::vector<cv::Point2f>* tfk::Section::generate_hex_grid(double* bounding_box, double spacing) {
  double hexheight = spacing;
  double hexwidth = sqrt(3.0) * spacing / 2.0;
  double vertspacing = 0.75 * hexheight;
  double horizspacing = hexwidth;
  int sizex = (int)((bounding_box[1]-bounding_box[0])/horizspacing) + 2; 
  int sizey = (int)((bounding_box[3]-bounding_box[2])/vertspacing) + 2;

  if (sizey % 2 == 0) {
    sizey += 1;
  }

  std::vector<cv::Point2f>* hex_grid = new std::vector<cv::Point2f>();
  for (int i = -2; i < sizex; i++) {
    for (int j = -2; j < sizey; j++) {
      //double xpos = i * spacing;
      //double ypos = j * spacing;
      double xpos = i * horizspacing;
      double ypos = j * vertspacing;
      if (j % 2 == 1) {
        xpos += spacing * 0.5;
      }
      if (j % 2 == 1 && i == sizex-1) {
        continue;
      }
      hex_grid->push_back(cv::Point2f(xpos+bounding_box[0], ypos+bounding_box[2]));
    }
  }
  return hex_grid;
}

void tfk::Section::affine_transform_mesh() {
  for (int mesh_index = 0; mesh_index < this->mesh->size(); mesh_index++) {
        (*this->mesh)[mesh_index] = this->affine_transform((*this->mesh)[mesh_index]);
        (*this->mesh_orig)[mesh_index] = this->affine_transform((*this->mesh_orig)[mesh_index]);
  }
}


void tfk::Section::construct_triangles() {
  double hex_spacing = 3000.0;

  std::pair<cv::Point2f, cv::Point2f> bbox = this->get_bbox();

  double min_x = bbox.first.x;
  double min_y = bbox.first.y;
  double max_x = bbox.second.x;
  double max_y = bbox.second.y;

  double bounding_box[4] = {min_x,max_x,min_y,max_y};
  std::vector<cv::Point2f>* hex_grid = this->generate_hex_grid(bounding_box, hex_spacing);

  cv::Rect rect(min_x-hex_spacing*2,min_y-hex_spacing*2,max_x-min_x+hex_spacing*4, max_y-min_y + hex_spacing*4);
  cv::Subdiv2D subdiv(rect);
  subdiv.initDelaunay(rect);
  for (int i = 0; i < hex_grid->size(); i++) {
    cv::Point2f pt = (*hex_grid)[i];
    subdiv.insert(pt);
  }

  std::vector<cv::Vec6f> triangle_list;
  subdiv.getTriangleList(triangle_list);

  printf("The number of triangles is %lu\n", triangle_list.size());

  std::vector<tfkTriangle> triangle_list_index;
  for (int i = 0; i < triangle_list.size(); i++) {
    cv::Point2f tpt1 = cv::Point2f(triangle_list[i][0],triangle_list[i][1]);
    cv::Point2f tpt2 = cv::Point2f(triangle_list[i][2],triangle_list[i][3]);
    cv::Point2f tpt3 = cv::Point2f(triangle_list[i][4],triangle_list[i][5]);
    int index1=-1;
    int index2=-1;
    int index3=-1;

    for (int j = 0; j < hex_grid->size(); j++) {
      cv::Point2f pt = (*hex_grid)[j];
      if (std::abs(pt.x-tpt1.x) < 0.01 && std::abs(pt.y-tpt1.y) < 0.01) {
        index1 = j;    
      }

      if (std::abs(pt.x-tpt2.x) < 0.01 && std::abs(pt.y-tpt2.y) < 0.01) {
        index2 = j;    
      }
      
      if (std::abs(pt.x-tpt3.x) < 0.01 && std::abs(pt.y-tpt3.y) < 0.01) {
        index3 = j;    
      }
    }
    //printf("triangle is %f %f %f %f %f %f\n", triangle_list[i][0], triangle_list[i][1], triangle_list[i][2], triangle_list[i][3], triangle_list[i][4], triangle_list[i][5]);
    //printf("points are %f %f %f %f %f %f\n", tpt1.x, tpt1.y, tpt2.x, tpt2.y, tpt3.x, tpt3.y);
    if (!(index1 >= 0 && index2 >= 0 && index3 >=0)) continue;
   // printf("Success\n");
    tfkTriangle tri;
    tri.index1 = index3;
    tri.index2 = index2;
    tri.index3 = index1;
    triangle_list_index.push_back(tri);
  }

  printf("Triangle_list_index length is %lu\n", triangle_list_index.size());

  std::vector<std::pair<int,int> > triangle_edges;
  for (int i = 0; i < triangle_list_index.size(); i++) {
    tfkTriangle tri = triangle_list_index[i];

    if (tri.index1 < tri.index2) {
      triangle_edges.push_back(std::make_pair(tri.index1,tri.index2));
    } else {
      triangle_edges.push_back(std::make_pair(tri.index2,tri.index1));
    }

    if (tri.index2 < tri.index3) {
      triangle_edges.push_back(std::make_pair(tri.index2,tri.index3));
    } else {
      triangle_edges.push_back(std::make_pair(tri.index3,tri.index2));
    }

    if (tri.index1 < tri.index3) {
      triangle_edges.push_back(std::make_pair(tri.index1,tri.index3));
    } else {
      triangle_edges.push_back(std::make_pair(tri.index3,tri.index1));
    }
  } 

  std::vector<std::pair<int,int> > triangle_edges_dedupe;
  std::set<std::pair<int, int> > triangle_edges_set;
  for (int i = 0; i < triangle_edges.size(); i++) {
    if (triangle_edges_set.find(triangle_edges[i]) == triangle_edges_set.end()) {
      triangle_edges_set.insert(triangle_edges[i]);
      triangle_edges_dedupe.push_back(triangle_edges[i]);
    }
  }


  std::vector<std::pair<int, int> >* _triangle_edges = new std::vector<std::pair<int, int> >();
  for (int i = 0; i < triangle_edges_dedupe.size(); i++) {
    _triangle_edges->push_back(triangle_edges_dedupe[i]);
  }
  std::vector<tfkTriangle>* _triangle_list = new std::vector<tfkTriangle>();
  for (int i = 0; i < triangle_list_index.size(); i++) {
    _triangle_list->push_back(triangle_list_index[i]);
  }  

  std::vector<cv::Point2f>* orig_hex_grid = new std::vector<cv::Point2f>();
  for (int i = 0; i < hex_grid->size(); i++) {
    orig_hex_grid->push_back((*hex_grid)[i]);
  }

  this->triangle_edges = _triangle_edges;
  this->mesh_orig = orig_hex_grid;
  this->mesh = hex_grid;
  this->triangles = _triangle_list;

}

void tfk::Section::write_wafer(FILE* wafer_file, int base_section) {
  fprintf(wafer_file, "[\n");
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    // Begin tile.
    fprintf(wafer_file, "\t{\n");
    tile->write_wafer(wafer_file, this->section_id, base_section);
    // End tile.
    if (i != graph->num_vertices()-1) {
      fprintf(wafer_file,"\t},\n");
    } else {
      fprintf(wafer_file,"\t}\n]");
    }
  }
}

std::pair<cv::Point2f, cv::Point2f> tfk::Section::get_bbox() {

  float min_x = 0;
  float max_x = 0;
  float min_y = 0;
  float max_y = 0;

  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    std::pair<cv::Point2f, cv::Point2f> bbox = tile->get_bbox();
    if (i == 0) {
      min_x = bbox.first.x;
      min_y = bbox.first.y;
      max_x = bbox.second.x;
      max_y = bbox.second.y;
    } else {
      if (bbox.first.x < min_x) min_x = bbox.first.x;
      if (bbox.first.y < min_y) min_y = bbox.first.y;
      if (bbox.second.x > max_x) max_x = bbox.second.x;
      if (bbox.second.y > max_y) max_y = bbox.second.y;
    }
  }
  return std::make_pair(cv::Point2f(min_x, min_y), cv::Point2f(max_x, max_y));
}


void tfk::Section::apply_affine_transforms() {
  // init identity matrix.
  cv::Mat A(3, 3, cv::DataType<double>::type);
  A.at<double>(0,0) = 1.0;
  A.at<double>(0,1) = 0.0;
  A.at<double>(0,2) = 0.0;
  A.at<double>(1,0) = 0.0;
  A.at<double>(1,1) = 1.0;
  A.at<double>(1,2) = 0.0;
  A.at<double>(2,0) = 0.0;
  A.at<double>(2,1) = 0.0;
  A.at<double>(2,2) = 1.0;

  for (int i = 0; i < this->affine_transforms.size(); i++) {
    A = A*this->affine_transforms[i];
  }

  this->a00 = A.at<double>(0,0);
  this->a01 = A.at<double>(0,1);
  this->offset_x = A.at<double>(0,2);
  this->a10 = A.at<double>(1,0);
  this->a11 = A.at<double>(1,1);
  this->offset_y = A.at<double>(1,2);

}

// Find affine transform for this section that aligns it to neighbor.
void tfk::Section::coarse_affine_align(Section* neighbor) {

  // a = neighbor.
  std::vector <cv::KeyPoint > atile_kps_in_overlap;
  std::vector <cv::Mat > atile_kps_desc_in_overlap_list;

  // b = this
  std::vector <cv::KeyPoint > btile_kps_in_overlap;
  std::vector <cv::Mat > btile_kps_desc_in_overlap_list;

  for (int i = 0; i < this->tiles.size(); i++) {
    this->tiles[i]->get_3d_keypoints(btile_kps_in_overlap, btile_kps_desc_in_overlap_list);
  }

  for (int i = 0; i < neighbor->tiles.size(); i++) {
    neighbor->tiles[i]->get_3d_keypoints(atile_kps_in_overlap, atile_kps_desc_in_overlap_list);
  }


  printf("Total size of a tile (%d) kps is %lu\n", neighbor->section_id, atile_kps_in_overlap.size());
  printf("Total size of b tile (%d) kps is %lu\n", this->section_id, btile_kps_in_overlap.size());

  cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
  cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
  cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));

  std::vector< cv::DMatch > matches;
  match_features(matches,
                 atile_kps_desc_in_overlap,
                 btile_kps_desc_in_overlap,
                 0.92);

  printf("Done with the matching. Num matches is %lu\n", matches.size());

  // Filter the matches with RANSAC
  std::vector<cv::Point2f> match_points_a, match_points_b;

  // Grab the matches.
  for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
    match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
    match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
    //p_align_data->sec_data[section_a].p_kps->push_back(atile_kps_in_overlap[matches[tmpi].queryIdx]);
    //p_align_data->sec_data[section_b].p_kps->push_back(btile_kps_in_overlap[matches[tmpi].trainIdx]);
  }

  bool* mask = (bool*)calloc(matches.size()+1, 1);
  //std::pair<double,double> offset_pair;

  // pre-filter matches with very forgiving ransac threshold.
  tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 1024, mask);

  std::vector< cv::Point2f > filtered_match_points_a_pre(0);
  std::vector< cv::Point2f > filtered_match_points_b_pre(0);
  int num_filtered = 0;
  for (int c = 0; c < matches.size(); c++) {
    if (mask[c]) {
      num_filtered++;
      filtered_match_points_a_pre.push_back(
          match_points_a[c]);
      filtered_match_points_b_pre.push_back(
          match_points_b[c]);
    }
  }
  free(mask);

  mask = (bool*)calloc(matches.size()+1, 1);
  printf("First pass filter got %d matches\n", num_filtered);

  if (num_filtered < 32) {
    printf("Not enough matches, skipping section\n");
    return;
  }

  tfk_simple_ransac_strict_ret_affine(filtered_match_points_a_pre, filtered_match_points_b_pre, 64.0, mask);

  std::vector< cv::Point2f > filtered_match_points_a(0);
  std::vector< cv::Point2f > filtered_match_points_b(0);

  num_filtered = 0;
  for (int c = 0; c < filtered_match_points_a_pre.size(); c++) {
    if (mask[c]) {
      num_filtered++;
      filtered_match_points_a.push_back(
          filtered_match_points_a_pre[c]);
      filtered_match_points_b.push_back(
          filtered_match_points_b_pre[c]);
    }
  }
  printf("Second pass filter got %d matches\n", num_filtered);

  if (num_filtered < 12) {
    printf("Not enough matches %d for section %d with thresh\n", num_filtered, this->section_id);
    return;
  } else {
    printf("Got enough matches %d for section %d with thresh\n", num_filtered, this->section_id);
  }

  cv::Mat section_transform;

  cv::computeAffineTFK(filtered_match_points_a, filtered_match_points_b, section_transform);

  //std::cout << section_transform << std::endl;

  // push this affine transform onto the neighbor.

  cv::Mat A(3, 3, cv::DataType<double>::type);

  //cv::Mat& B = section_transform;

  //neighbor->a00 = B.at<double>(0,0);
  //neighbor->a01 = B.at<double>(0,1);
  //neighbor->a10 = B.at<double>(1,0);
  //neighbor->a11 = B.at<double>(1,1);
  //neighbor->offset_x = B.at<double>(0,2);
  //neighbor->offset_y = B.at<double>(1,2);

  A.at<double>(0,0) = section_transform.at<double>(0,0);
  A.at<double>(0,1) = section_transform.at<double>(0,1);
  A.at<double>(0,2) = section_transform.at<double>(0,2);
  A.at<double>(1,0) = section_transform.at<double>(1,0);
  A.at<double>(1,1) = section_transform.at<double>(1,1);
  A.at<double>(1,2) = section_transform.at<double>(1,2);
  A.at<double>(2,0) = 0.0;
  A.at<double>(2,1) = 0.0;
  A.at<double>(2,2) = 1.0;

  printf("Printing out A\n");
  std::cout << A << std::endl;

  //neighbor->affine_transforms.push_back(A);
  //this->coarse_transform = new cv::Mat();
  this->coarse_transform = A.clone();

  //this->a00 = 1.0;
  //this->a01 = 0.0;
  //this->a10 = 0.0;
  //this->a11 = 1.0;
  //this->offset_x = 0.0;
  //this->offset_y = 0.0;

}


void tfk::Section::compute_tile_matches(Tile* a_tile, Graph* graph) {

  std::vector<int> neighbors = get_all_close_tiles(a_tile->tile_id);

  //Tile* a_tile = this->tiles[tile_id];

  for (int i = 0; i < neighbors.size(); i++) {
    //int atile_id = tile_id;
    int btile_id = neighbors[i];
    Tile* b_tile = this->tiles[btile_id];

    if (a_tile->p_kps->size() < MIN_FEATURES_NUM) continue;
    if (b_tile->p_kps->size() < MIN_FEATURES_NUM) continue;

    // Filter the features, so that only features that are in the
    //   overlapping tile will be matches.
    std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;
    atile_kps_in_overlap.reserve(a_tile->p_kps->size());
    btile_kps_in_overlap.reserve(b_tile->p_kps->size());
    // atile_kps_in_overlap.clear(); btile_kps_in_overlap.clear();
    cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;

    { // Begin scoped block A.
      // Compute bounding box of overlap
      int overlap_x_start = a_tile->x_start > b_tile->x_start ?
                                a_tile->x_start : b_tile->x_start;
      int overlap_x_finish = a_tile->x_finish < b_tile->x_finish ?
                                a_tile->x_finish : b_tile->x_finish;
      int overlap_y_start = a_tile->y_start > b_tile->y_start ?
                                a_tile->y_start : b_tile->y_start;
      int overlap_y_finish = a_tile->y_finish < b_tile->y_finish ?
                                a_tile->y_finish : b_tile->y_finish;
      // Add 50-pixel offset
      const int OFFSET = 50;
      overlap_x_start -= OFFSET;
      overlap_x_finish += OFFSET;
      overlap_y_start -= OFFSET;
      overlap_y_finish += OFFSET;

      std::vector< cv::Mat > atile_kps_desc_in_overlap_list;
      atile_kps_desc_in_overlap_list.reserve(a_tile->p_kps->size());
      std::vector< cv::Mat > btile_kps_desc_in_overlap_list;
      btile_kps_desc_in_overlap_list.reserve(b_tile->p_kps->size());

      // Filter the points in a_tile.
      for (size_t pt_idx = 0; pt_idx < a_tile->p_kps->size(); ++pt_idx) {
        cv::Point2f pt = (*a_tile->p_kps)[pt_idx].pt;
        if (bbox_contains(pt.x + a_tile->x_start,
                          pt.y + a_tile->y_start,  // transformed_pt[0],
                          overlap_x_start, overlap_x_finish,
                          overlap_y_start, overlap_y_finish)) {
          atile_kps_in_overlap.push_back((*a_tile->p_kps)[pt_idx]);
          atile_kps_desc_in_overlap_list.push_back(
              a_tile->p_kps_desc->row(pt_idx).clone());
        }
      }
      cv::vconcat(atile_kps_desc_in_overlap_list,
          (atile_kps_desc_in_overlap));

      // Filter the points in b_tile.
      for (size_t pt_idx = 0; pt_idx < b_tile->p_kps->size(); ++pt_idx) {
        cv::Point2f pt = (*b_tile->p_kps)[pt_idx].pt;
        if (bbox_contains(pt.x + b_tile->x_start,
                          pt.y + b_tile->y_start,  // transformed_pt[0],
                          overlap_x_start, overlap_x_finish,
                          overlap_y_start, overlap_y_finish)) {
          btile_kps_in_overlap.push_back((*b_tile->p_kps)[pt_idx]);
          btile_kps_desc_in_overlap_list.push_back(b_tile->p_kps_desc->row(pt_idx).clone());
        }
      }
      cv::vconcat(btile_kps_desc_in_overlap_list,
          (btile_kps_desc_in_overlap));
    } // End scoped block A

    if (atile_kps_in_overlap.size() < MIN_FEATURES_NUM) continue;
    if (btile_kps_in_overlap.size() < MIN_FEATURES_NUM) continue;

    float trial_rod;
    for (int trial = 0; trial < 4; trial++) {
      if (trial == 0) trial_rod = 0.7;
      if (trial == 1) trial_rod = 0.8;
      if (trial == 2) trial_rod = 0.92;
      if (trial == 3) trial_rod = 0.96;
      // Match the features
      std::vector< cv::DMatch > matches;
      match_features(matches,
                     atile_kps_desc_in_overlap,
                     btile_kps_desc_in_overlap,
                     trial_rod);

      // Filter the matches with RANSAC
      std::vector<cv::Point2f> match_points_a, match_points_b;
      for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
        match_points_a.push_back(
            atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
        match_points_b.push_back(
            btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
      }

      if (matches.size() < MIN_FEATURES_NUM) {
        continue;
      }

      bool* mask = (bool*) calloc(match_points_a.size(), 1);
      double thresh = 5.0;
      tfk_simple_ransac(match_points_a, match_points_b, thresh, mask);


      std::vector< cv::Point2f > filtered_match_points_a(0);
      std::vector< cv::Point2f > filtered_match_points_b(0);

      int num_matches_filtered = 0;
      // Use the output mask to filter the matches
      for (size_t i = 0; i < matches.size(); ++i) {
        if (mask[i]) {
          num_matches_filtered++;
          filtered_match_points_a.push_back(
              atile_kps_in_overlap[matches[i].queryIdx].pt);
          filtered_match_points_b.push_back(
              btile_kps_in_overlap[matches[i].trainIdx].pt);
        }
      }
      free(mask);
      if (num_matches_filtered > 12) {
        a_tile->insert_matches(b_tile, filtered_match_points_a, filtered_match_points_b);
        //graph->insert_matches(atile_id, btile_id,
        //    filtered_match_points_a, filtered_match_points_b, 1.0);
        break;
      }
    }
  }
}



void tfk::Section::read_3d_keypoints(std::string filename) {
  cv::FileStorage fs(filename+std::string("_3d_keypoints.yml.gz"), cv::FileStorage::READ);
  int count = 0;
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    tile->p_kps_3d = new std::vector<cv::KeyPoint>();
    tile->p_kps_desc_3d = new cv::Mat();
    fs["keypoints_"+std::to_string(i)] >> *(tile->p_kps_3d);
    count += tile->p_kps_3d->size();
    fs["descriptors_"+std::to_string(i)] >> *(tile->p_kps_desc_3d);
  }
  fs.release();
  printf("Read %d 3d matches for section %d\n", count, this->real_section_id);
}

void tfk::Section::save_3d_keypoints(std::string filename) {
  cv::FileStorage fs(filename+std::string("_3d_keypoints.yml.gz"),
                     cv::FileStorage::WRITE);
  // store the 3d keypoints
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    cv::write(fs, "keypoints_"+std::to_string(i),
              (*(tile->p_kps_3d)));
    cv::write(fs, "descriptors_"+std::to_string(i),
              (*(tile->p_kps_desc_3d)));
  }
  fs.release();
}

void tfk::Section::save_2d_graph(std::string filename) {
  cv::FileStorage fs(filename+std::string("_2d_matches.yml.gz"), cv::FileStorage::WRITE);
  int count = 0;
  for (int i = 0; i < this->tiles.size(); i++) {
      Tile* tile = this->tiles[i];
      cv::write(fs, "num_edges_"+std::to_string(i), (int) tile->edges.size());
    for (int j = 0; j < tile->edges.size(); j++) {
      cv::write(fs, "neighbor_id_"+std::to_string(i) + "_" + std::to_string(j),
          tile->edges[j].neighbor_id);
      cv::write(fs, "weight_"+std::to_string(i) + "_" + std::to_string(j),
          1.0);
      cv::write(fs, "v_points_"+std::to_string(i)+"_"+std::to_string(j),
          *(tile->edges[j].v_points));
      cv::write(fs, "n_points_"+std::to_string(i)+"_"+std::to_string(j),
          *(tile->edges[j].n_points));
      count++;
    }
  }
  printf("wrote %d edges\n", count);
  fs.release();
}
void tfk::Section::read_2d_graph(std::string filename) {
  cv::FileStorage fs(filename+std::string("_2d_matches.yml.gz"), cv::FileStorage::READ);
  int count = 0;
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    int edge_size;
    fs["num_edges_"+std::to_string(i)] >> edge_size;
    std::vector<edata> edge_data;
    std::set<int> n_ids_seen;
    tile->edges.clear();
    n_ids_seen.clear();
    for (int j = 0; j < edge_size; j++) {
      edata edge;
      fs["neighbor_id_"+std::to_string(i) + "_" + std::to_string(j)] >> edge.neighbor_id;
      std::vector<cv::Point2f>* v_points = new std::vector<cv::Point2f>();
      std::vector<cv::Point2f>* n_points = new std::vector<cv::Point2f>();
      fs["weight_"+std::to_string(i) + "_" + std::to_string(j)] >> edge.weight;
      fs["v_points_"+std::to_string(i) + "_" + std::to_string(j)] >> *v_points;
      fs["n_points_"+std::to_string(i) + "_" + std::to_string(j)] >> *n_points;
      edge.v_points = v_points;
      edge.n_points = n_points;
      edge.neighbor_tile = this->tiles[edge.neighbor_id];
      tile->edges.push_back(edge);
      count++;
    }
    //tile->edges = edge_data;
  }

  printf("read %d edges\n", count);
  fs.release();
}



void tfk::Section::read_tile_matches() {

  std::string filename =
      std::string("newcached_data/prefix_"+std::to_string(this->real_section_id));

  this->read_3d_keypoints(filename);
  this->read_2d_graph(filename);
}

void tfk::Section::save_tile_matches() {

  std::string filename =
      std::string("newcached_data/prefix_"+std::to_string(this->real_section_id));

  this->save_3d_keypoints(filename);
  this->save_2d_graph(filename);
}

void tfk::Section::recompute_keypoints() {
  cilk_for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    if (tile->image_data_replaced) {
      //tile->compute_sift_keypoints2d();
      tile->compute_sift_keypoints3d();
      tile->image_data_replaced = false;
    }
  }
}

void tfk::Section::compute_keypoints_and_matches() {
  // assume that section data doesn't exist.

  if (!this->section_data_exists()) {

    std::vector<std::pair<float, Tile*> > sorted_tiles;
    for (int i = 0; i < this->tiles.size(); i++) {
      Tile* t = this->tiles[i];
      sorted_tiles.push_back(std::make_pair(t->x_start, t));
    }
    std::sort(sorted_tiles.begin(), sorted_tiles.end());
    // sorted tiles is now sorted in increasing order based on x_start;

    std::set<Tile*> active_set;
    std::set<Tile*> neighbor_set;

    std::set<Tile*> opened_set;
    std::set<Tile*> closed_set;

    Tile* pivot = sorted_tiles[0].second;

    bool pivot_good = false;
    int pivot_search_start = 0;
    for (int i = pivot_search_start; i < sorted_tiles.size(); i++) {
      if (sorted_tiles[i].second->x_start > pivot->x_finish) {
        pivot = sorted_tiles[i].second;
        pivot_search_start = i;
        pivot_good = true;
        break;
      } else {
        active_set.insert(sorted_tiles[i].second);
      }
    }
    printf("Num tiles in sweep 0 is %lu\n", active_set.size()); 

    while (active_set.size() > 0) {
      printf("Current active set size is %lu\n", active_set.size());
      // find all the neighbors.
      for (auto it = active_set.begin(); it != active_set.end(); ++it) {
        Tile* tile = *it;
        std::vector<Tile*> overlapping = this->get_all_close_tiles(tile);
        for (int j = 0; j < overlapping.size(); j++) {
          neighbor_set.insert(overlapping[j]);
        }
      }

      // close open tiles that aren't in active or neighbor set.
      for (auto it = opened_set.begin(); it != opened_set.end(); ++it) {
        Tile* tile = *it;
        if (active_set.find(tile) == active_set.end() &&
            neighbor_set.find(tile) == neighbor_set.end()) {
          closed_set.insert(tile);
          tile->release_2d_keypoints();
        }
      }

      std::vector<Tile*> tiles_to_process_keypoints, tiles_to_process_matches;

      for (auto it = active_set.begin(); it != active_set.end(); ++it) {
        Tile* tile = *it;
        tiles_to_process_matches.push_back(tile);
        if (opened_set.find(tile) == opened_set.end()) {
          opened_set.insert(tile);
          tiles_to_process_keypoints.push_back(tile);
        }
      }

      for (auto it = neighbor_set.begin(); it != neighbor_set.end(); ++it) {
        Tile* tile = *it;
        if (opened_set.find(tile) == opened_set.end()) {
          opened_set.insert(tile);
          tiles_to_process_keypoints.push_back(tile);
        }
      }

      cilk_for (int i = 0; i < tiles_to_process_keypoints.size(); i++) {
        Tile* tile = tiles_to_process_keypoints[i];
        tile->compute_sift_keypoints2d();
        //tile->compute_sift_keypoints3d();
      }

      cilk_for (int i = 0; i < tiles_to_process_matches.size(); i++) {
        this->compute_tile_matches(tiles_to_process_matches[i], graph);
      }

      opened_set.clear();
      for (auto it = active_set.begin(); it != active_set.end(); ++it) {
        opened_set.insert(*it);
      }
      for (auto it = neighbor_set.begin(); it != neighbor_set.end(); ++it) {
        opened_set.insert(*it);
      }
      // clear the active and neighbor set.
      active_set.clear();
      neighbor_set.clear();

      pivot_good = false;
      for (int i = pivot_search_start; i < sorted_tiles.size(); i++) {
        if (sorted_tiles[i].second->x_start > pivot->x_finish) {
          pivot = sorted_tiles[i].second;
          pivot_search_start = i;
          pivot_good = true;
          break;
        } else {
          active_set.insert(sorted_tiles[i].second);
        }
      }
      if (!pivot_good) break;
    }

    //cilk_for (int i = 0; i < this->tiles.size(); i++) {
    //  Tile* tile = this->tiles[i];
    //  tile->compute_sift_keypoints2d();
    //  tile->compute_sift_keypoints3d();
    //}


    //cilk_for (int i = 0; i < this->tiles.size(); i++) {
    //  this->compute_tile_matches(i, graph);
    //}

    //this->save_tile_matches();
  } else {
    this->read_tile_matches();
  }

    this->graph = new Graph();
    graph->resize(this->tiles.size());
  // phase 0 of make_symmetric --- find edges to add.
  cilk_for (int i = 0; i < this->tiles.size(); i++) {
    this->tiles[i]->make_symmetric(0, this->tiles);
  }

  // phase 1 of make_symmetric --- insert found edges.
  cilk_for (int i = 0; i < this->tiles.size(); i++) {
    this->tiles[i]->make_symmetric(1, this->tiles);
  }


  for (int i = 0; i < graph->num_vertices(); i++) {
    vdata* d = graph->getVertexData(i);


    for (int j = 0; j < this->tiles[i]->edges.size(); j++) {
      graph->edgeData[i].push_back(this->tiles[i]->edges[j]);
    }

    //_tile_data tdata = p_sec_data->tiles[i];
    Tile* tile = this->tiles[i];
    d->tile = tile;
    d->vertex_id = i;
    d->mfov_id = tile->mfov_id;
    d->tile_index = tile->index;
    d->tile_id = i;
    d->start_x = tile->x_start;
    d->end_x = tile->x_finish;
    d->start_y = tile->y_start;
    d->end_y = tile->y_finish;
    d->offset_x = 0.0;
    d->offset_y = 0.0;
    d->iteration_count = 0;
    //d->last_radius_value = 9.0;
    d->z = /*p_align_data->base_section + */this->section_id;
    d->a00 = 1.0;
    d->a01 = 0.0;
    d->a10 = 0.0;
    d->a11 = 1.0;
    //d->neighbor_grad_x = 0.0;
    //d->neighbor_grad_y = 0.0;
    //d->converged = 0;
    d->original_center_point =
      cv::Point2f((tile->x_finish-tile->x_start)/2,
                  (tile->y_finish-tile->y_start)/2);
  }

  printf("Num vertices is %d\n", graph->num_vertices());
  for (int i = 0; i < graph->num_vertices(); i++) {
    printf("The graph vertex id is %d\n",graph->getVertexData(i)->vertex_id);
  }
  printf("Num vertices is %d\n", graph->num_vertices());
  graph->section_id = this->section_id;

  // now compute keypoint matches
  //cilk_for (int i = 0; i < this->tiles.size(); i++) {
  //    compute_tile_matches_active_set(p_align_data, sec_id, active_set, graph);
  //}

}


std::vector<tfk::Tile*> tfk::Section::get_all_close_tiles(Tile* a_tile) {
  std::vector<Tile*> neighbor_tiles(0);
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* b_tile = this->tiles[i];
    if (a_tile->overlaps_with(b_tile)) {
      neighbor_tiles.push_back(b_tile);
    }
  }
  return neighbor_tiles;
}


std::vector<int> tfk::Section::get_all_close_tiles(int atile_id) {
  std::vector<int> neighbor_index_list(0);

  //int indices_to_check_len = 0;

  Tile* a_tile = this->tiles[atile_id];
  for (int i = atile_id+1; i < this->tiles.size(); i++) {
    Tile* b_tile = this->tiles[i];
    if (a_tile->overlaps_with(b_tile)) {
      neighbor_index_list.push_back(i);
    }
  }

  return neighbor_index_list;
}

// Section from protobuf
tfk::Section::Section(SectionData& section_data) {
  //section_data_t *p_sec_data = &(p_tile_data->sec_data[i - p_tile_data->base_section]);

  this->section_id = section_data.section_id();
  this->real_section_id = section_data.section_id();
  this->n_tiles = 0;
  this->a00 = 1.0;
  this->a11 = 1.0;
  this->a01 = 0.0;
  this->a10 = 0.0;
  this->offset_x = 0.0;
  this->offset_y = 0.0;
  if (section_data.has_out_d1()) {
    this->out_d1 = section_data.out_d1();
  }
  if (section_data.has_out_d2()) {
    this->out_d2 = section_data.out_d2();
  }


  for (int j = 0; j < section_data.tiles_size(); j++) {
    TileData tile_data = section_data.tiles(j);

    Tile* tile = new Tile(tile_data);
    tile->tile_id = j;

    std::string new_filepath = "new_tiles/sec_"+std::to_string(this->real_section_id) +
        "_tileid_"+std::to_string(tile->tile_id) + ".bmp";

    std::string test_filepath = "new_tiles/thumbnail_sec_"+std::to_string(this->real_section_id) +
        "_tileid_"+std::to_string(tile->tile_id) + ".jpg";
    //if (FILE *file = fopen(test_filepath.c_str(), "r")) {
    //    fclose(file);
    //  tile->filepath = std::string(new_filepath);
    //  printf("Found replacement tile\n");
    //} else {
    //}

    this->tiles.push_back(tile);
  }

}



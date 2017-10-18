#include <iostream>
#include "./simple_mutex.h"

#include <fstream>
//#include "mesh.h"

enum Resolution {THUMBNAIL, FULL};

static float Dot(cv::Point2f a, cv::Point2f b) {
  return a.x*b.x + a.y*b.y;
}

static cv::Mat colorize(cv::Mat c1, cv::Mat c2, cv::Mat c3, unsigned int channel = 0) {
    std::vector<cv::Mat> channels;
    channels.push_back(c1);
    channels.push_back(c2);
    channels.push_back(c3);
    cv::Mat color;
    cv::merge(channels, color);
    return color;
}

static cv::Mat apply_heatmap_to_grayscale(cv::Mat* gray, cv::Mat* heat_floats, int nrows, int ncols) {
  cv::Mat c1,c2,c3;
  c1.create(nrows, ncols, CV_8UC1);
  c2.create(nrows, ncols, CV_8UC1);
  c3.create(nrows, ncols, CV_8UC1);

  for (int x = 0; x < gray->size().width; x++) {
    for (int y = 0; y < gray->size().height; y++) {
       float g = (1.0*gray->at<unsigned char>(y,x))/255;
       float h = 0.5 + 0.5*(heat_floats->at<float>(y,x)*heat_floats->at<float>(y,x));

       c1.at<unsigned char>(y,x) = (unsigned char) (g*255*(1-h));
       c2.at<unsigned char>(y,x) = (unsigned char) (g*255*(1-h));
       c3.at<unsigned char>(y,x) = (unsigned char) (g*255*((h)));
    }
  }
  return colorize(c1,c2,c3);
}


/*MOVE TO COMMON.H OR SOME OTHER FILE*/
static void Barycentric(cv::Point2f p, cv::Point2f a, cv::Point2f b, cv::Point2f c,
   float &u, float &v, float &w)
{
    cv::Point2f v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = Dot(v0, v0);
    float d01 = Dot(v0, v1);
    float d11 = Dot(v1, v1);
    float d20 = Dot(v2, v0);
    float d21 = Dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}

/* resizing using remap */
cv::Mat resize(float scale, cv::Mat img) {
	int new_width = scale*img.size().width;
	int new_height = scale*img.size().height;
	cv::Mat new_img;
	cv::Mat map_x, map_y;
	new_img.create(new_height, new_width, CV_8UC1);
	map_x.create(new_img.size(),CV_32FC1);
	map_y.create(new_img.size(), CV_32FC1);
	
	for(int i = 0; i < new_img.rows; i ++) {
		for(int j = 0; j < new_img.cols; j ++) {
			map_x.at<float>(i,j) = j/scale;
			map_y.at<float>(i,j) = i/scale;
		}
	}
	cv::remap(img, new_img, map_x, map_y, CV_INTER_AREA, cv::BORDER_TRANSPARENT, cv::Scalar(0, 0, 0));
	return new_img;
}

bool inBox(int x, int y, int x_lower, int x_upper, int y_lower, int y_upper) {
	if(x >= x_lower && x <= x_upper && y >= y_lower && y <= y_lower) {
		return true;
	}
	return false;
}


std::tuple<bool, float, float, float> findTriangle(std::vector<renderTriangle>*mesh_triangles, cv::Point2f point, bool useQValues = false) {
	for(int i = 0; i < mesh_triangles->size(); i ++) {
      float u, v, w;	
      cv::Point2f a,b,c; 
      if (!useQValues) {
      a = (*mesh_triangles)[i].p[0];
      b = (*mesh_triangles)[i].p[1];
      c = (*mesh_triangles)[i].p[2];
      } else {
      a = (*mesh_triangles)[i].q[0];
      b = (*mesh_triangles)[i].q[1];
      c = (*mesh_triangles)[i].q[2];
      }
      Barycentric(point, a,b,c,u,v,w);
      if (u >= 0 && v >= 0 && w >= 0) {
			int j = i;
			while(j > 0) {
				renderTriangle temp = (*mesh_triangles)[j - 1];
				(*mesh_triangles)[j - 1] = (*mesh_triangles)[j];
				(*mesh_triangles)[j] = temp;
				j --;
			} 
			return std::make_tuple(true, u, v, w);
		}
	}	
	return std::make_tuple(false, -1, -1, -1);
}



cv::Point2f elastic_transform(tile_data_t *t, std::vector<renderTriangle>* mesh_triangles, cv::Point2f point) {
	//std::vector<renderTriangle>*mesh_triangles = t->mesh_triangles;
	auto  foundTriangle = findTriangle(mesh_triangles, point);
	if(!std::get<0>(foundTriangle)) {
		//std::cout << " elastic TRIANGLE NOT FOUND " << std::endl;
		return point;
	}
	renderTriangle tri = (*mesh_triangles)[0];	
	float u = std::get<1>(foundTriangle);
	float v = std::get<2>(foundTriangle);
	float w = std::get<3>(foundTriangle);
	float new_x = u*tri.q[0].x + v*tri.q[1].x + w*tri.q[2].x;
	float new_y = u*tri.q[0].y + v*tri.q[1].y + w*tri.q[2].y;

	return cv::Point2f(new_x, new_y);
}

cv::Point2f affine_transform(tile_data_t * t, cv::Point2f point) {
	float new_x = point.x*t->a00 + point.y*t->a01 + t->offset_x;
	float new_y = point.x*t->a10 + point.y*t->a11 + t->offset_y;
	return cv::Point2f(new_x, new_y);
}

float max(float a, float b) {
	if(a > b) return a;
	return b;
} 
float min(float a, float b) {
	if(a < b) return a;
	return b;
}

float max(float a, float b, float c, float d){
	if(a >= b && a >= c && a >= d) {
		return a;
	} else if(b >= a && b >= c && b >= d) {
		return b;
	} else if(c >= d) {
		return c;
	} else {
		return d;
	}
}
float min(float a, float b, float c, float d){
	if(a <= b && a <= c && a <= d) {
		return a;
	} else if(b <= a && b <= c && b <= d) {
		return b;
	} else if(c <= d) {
		return c;
	} else {
		return d;
	}
}


bool area_overlap(cv::Point2f c1, cv::Point2f c2, cv::Point2f c3, cv::Point2f c4, cv::Point2f d1, cv::Point2f d2, cv::Point2f d3, cv::Point2f d4) {
		float lower_x = min(d1.x, d2.x, d3.x, d4.x);
		float upper_x = max(d1.x, d2.x, d3.x, d4.x);
		float lower_y = min(d1.y, d2.y, d3.y, d4.y);
		float upper_y = max(d1.y, d2.y, d3.y, d4.y);

		if((c1.y < upper_y && c1.y >= lower_y && c1.x < upper_x && c1.x >= lower_x) ||
			(c2.y < upper_y && c2.y >= lower_y && c2.x < upper_x && c2.x >= lower_x) ||
			(c3.y < upper_y && c3.y >= lower_y && c3.x < upper_x && c3.x >= lower_x) ||
			(c4.y < upper_y && c4.y >= lower_y && c4.x < upper_x && c4.x >= lower_x)) {	
			return true;
		} 
		//add case for tile completely contained
		if((c1.y > upper_y && c1.y <= lower_y && c1.x > upper_x && c1.x <= lower_x) &&
			(c2.y > upper_y && c2.y <= lower_y && c2.x > upper_x && c2.x <= lower_x) &&
			(c3.y > upper_y && c3.y <= lower_y && c3.x > upper_x && c3.x <= lower_x) &&
			(c4.y > upper_y && c4.y <= lower_y && c4.x > upper_x && c4.x <= lower_x)) {
			return true;
		}
		bool x_overlap = (lower_x >= min(min(c1.x, c2.x), min(c3.x, c4.x)) && lower_x < max(max(c1.x, c2.x), max(c3.x, c4.x))) ||
						(upper_x >= min(min(c1.x, c2.x), min(c3.x, c4.x)) && upper_x < max(max(c1.x, c2.x), max(c3.x, c4.x)));
		bool y_overlap = (lower_y >= min(min(c1.y, c2.y), min(c3.y, c4.y)) && lower_y < max(max(c1.y, c2.y), max(c3.y, c4.y))) ||
						(upper_y >= min(min(c1.y, c2.y), min(c3.y, c4.y)) && upper_y < max(max(c1.y, c2.y), max(c3.y, c4.y)));
		if(x_overlap && y_overlap) {			
			return true;
		}
		return false; 
}
float max_x(tile_data_t *tile) {
    int width = SIFT_D2_SHIFT_3D;
    int height = SIFT_D1_SHIFT_3D; //MIGHT BE OTHER WAY AROUND
    cv::Point2f c1 = cv::Point2f(0.0, 0.0);
    cv::Point2f c2 = cv::Point2f(width, 0.0);
    cv::Point2f c3 = cv::Point2f(0.0, height);
    cv::Point2f c4 = cv::Point2f(width, height);
    c1 = affine_transform(tile, c1);
    c2 = affine_transform(tile, c2);
    c3 = affine_transform(tile, c3);
    c4 = affine_transform(tile, c4);
    return max(c1.x, c2.x, c3.x, c4.x);
}
float max_y(tile_data_t *tile) {
    int width = SIFT_D2_SHIFT_3D;
    int height = SIFT_D1_SHIFT_3D; //MIGHT BE OTHER WAY AROUND
    cv::Point2f c1 = cv::Point2f(0.0, 0.0);
    cv::Point2f c2 = cv::Point2f(width, 0.0);
    cv::Point2f c3 = cv::Point2f(0.0, height);
    cv::Point2f c4 = cv::Point2f(width, height);
    c1 = affine_transform(tile, c1);
    c2 = affine_transform(tile, c2);
    c3 = affine_transform(tile, c3);
    c4 = affine_transform(tile, c4);
    return max(c1.y, c2.y, c3.y, c4.y);
}
float min_x(tile_data_t *tile) {
    int width = SIFT_D2_SHIFT_3D;
    int height = SIFT_D1_SHIFT_3D; //MIGHT BE OTHER WAY AROUND
    cv::Point2f c1 = cv::Point2f(0.0, 0.0);
    cv::Point2f c2 = cv::Point2f(width, 0.0);
    cv::Point2f c3 = cv::Point2f(0.0, height);
    cv::Point2f c4 = cv::Point2f(width, height);
    c1 = affine_transform(tile, c1);
    c2 = affine_transform(tile, c2);
    c3 = affine_transform(tile, c3);
    c4 = affine_transform(tile, c4);
    return min(c1.x, c2.x, c3.x, c4.x);
}
float min_y(tile_data_t *tile) {
    int width = SIFT_D2_SHIFT_3D;
    int height = SIFT_D1_SHIFT_3D; //MIGHT BE OTHER WAY AROUND
    cv::Point2f c1 = cv::Point2f(0.0, 0.0);
    cv::Point2f c2 = cv::Point2f(width, 0.0);
    cv::Point2f c3 = cv::Point2f(0.0, height);
    cv::Point2f c4 = cv::Point2f(width, height);
    c1 = affine_transform(tile, c1);
    c2 = affine_transform(tile, c2);
    c3 = affine_transform(tile, c3);
    c4 = affine_transform(tile, c4);
    return min(c1.y, c2.y, c3.y, c4.y);
}



bool tile_in_bounds(tile_data_t tile, int lower_x, int upper_x, int lower_y, int upper_y) {
		int width = SIFT_D2_SHIFT_3D;
		int height = SIFT_D1_SHIFT_3D; //MIGHT BE OTHER WAY AROUND
		cv::Point2f c1 = cv::Point2f(0.0, 0.0);
		cv::Point2f c2 = cv::Point2f(width, 0.0);
		cv::Point2f c3 = cv::Point2f(0.0, height);
		cv::Point2f c4 = cv::Point2f(width, height);
		c1 = affine_transform(&tile, c1);		
		c2 = affine_transform(&tile, c2);
		c3 = affine_transform(&tile, c3);
		c4 = affine_transform(&tile, c4);
		cv::Point2f d1 = cv::Point2f(lower_x, lower_y);
		cv::Point2f d2 = cv::Point2f(lower_x, upper_y);
		cv::Point2f d3 = cv::Point2f(upper_x, lower_y);
		cv::Point2f d4 = cv::Point2f(upper_x, upper_y);
		return area_overlap(c1, c2, c3, c4, d1, d2, d3, d4);	
}

bool tile_completely_in_bounds(tile_data_t tile, int lower_x, int upper_x, int lower_y, int upper_y) {
		int width = SIFT_D2_SHIFT_3D;
		int height = SIFT_D1_SHIFT_3D; //MIGHT BE OTHER WAY AROUND
		cv::Point2f c1 = cv::Point2f(0.0, 0.0);
		cv::Point2f c2 = cv::Point2f(1.0f*width, 0.0);
		cv::Point2f c3 = cv::Point2f(0.0, 1.0f*height);
		cv::Point2f c4 = cv::Point2f(1.0f*width, 1.0f*height);
		c1 = affine_transform(&tile, c1);		
		c2 = affine_transform(&tile, c2);
		c3 = affine_transform(&tile, c3);
		c4 = affine_transform(&tile, c4);
		std::cout << "TILE " << c1.x << " " << c1.y << " "
					<< c2.x << " " << c2.y << " "
					<< c3.x << " " << c3.y << " "
					<< c4.x << " " << c4.y << " " << std::endl;
		return c1.x >= lower_x && c1.x <= upper_x && c1.y >= lower_y && c1.y <= upper_y &&
				c2.x >= lower_x && c2.x <= upper_x && c2.y >= lower_y && c2.y <= upper_y &&
				c3.x >= lower_x && c3.x <= upper_x && c3.y >= lower_y && c3.y <= upper_y &&
				c4.x >= lower_x && c4.x <= upper_x && c4.y >= lower_y && c4.y <= upper_y;
}

typedef struct {
  cv::Mat img;
  std::string image_path;
} worker_image_cache_entry;

thread_local std::vector<worker_image_cache_entry>* thread_worker_cache;   
static int MAX_WORKER_IMAGE_CACHE_SIZE = 1000;

// NOTE WARNING: This code caches but the cache doesn't consider load_type.
cv::Mat imread_with_cache(std::string filepath, int load_type) {
  if (thread_worker_cache == NULL) {
    // init the cache.
    thread_worker_cache = new std::vector<worker_image_cache_entry>();
  }

  for (int i = 0; i < thread_worker_cache->size(); i++) {
    std::string cache_filename = (*thread_worker_cache)[i].image_path;
    if (cache_filename.compare(filepath) == 0) {
      return (*thread_worker_cache)[i].img;
    }
  }

  cv::Mat img = cv::imread(filepath, load_type);
  worker_image_cache_entry entry;
  entry.img = img;
  entry.image_path = filepath;
  if (thread_worker_cache->size() > MAX_WORKER_IMAGE_CACHE_SIZE) {
    (*thread_worker_cache)[thread_worker_cache->size()-1].img.release();
    thread_worker_cache->pop_back();
    thread_worker_cache->push_back(entry);
    for (int i = thread_worker_cache->size(); --i > 0; ) {
      worker_image_cache_entry tmp = (*thread_worker_cache)[i];
      (*thread_worker_cache)[i] = (*thread_worker_cache)[i-1];
      (*thread_worker_cache)[i-1] = tmp;
    } 
  } else {
    thread_worker_cache->push_back(entry);
    for (int i = thread_worker_cache->size(); --i > 0; ) {
      worker_image_cache_entry tmp = (*thread_worker_cache)[i];
      (*thread_worker_cache)[i] = (*thread_worker_cache)[i-1];
      (*thread_worker_cache)[i-1] = tmp;
    } 
  }
  return entry.img;
}
void set_render_parameters(int &lower_y, int &lower_x, int &upper_y, int &upper_x, int &ncols, int &nrows,double &scale_x, double &scale_y, section_data_t* section, std::string filename, int input_lower_x, int input_upper_x, int input_lower_y, int input_upper_y, Resolution res) {
  if (res == THUMBNAIL) {
    std::string thumbnailpath = std::string(section->tiles[0].filepath);
    thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
    thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");

    
    cv::Mat thumbnail_img = imread_with_cache(thumbnailpath,CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat img = imread_with_cache(section->tiles[0].filepath,CV_LOAD_IMAGE_UNCHANGED);

    scale_x = (double)(img.size().width)/thumbnail_img.size().width;
    scale_y = (double)(img.size().height)/thumbnail_img.size().height;

    //create new matrix
    lower_y = (int)(input_lower_y/scale_y + 0.5);
    lower_x = (int)(input_lower_x/scale_x + 0.5);
    upper_y = (int)(input_upper_y/scale_y + 0.5);
    upper_x = (int)(input_upper_x/scale_x + 0.5);

    nrows = (input_upper_y-input_lower_y)/scale_y;
    ncols = (input_upper_x-input_lower_x)/scale_x; 
  }

  if (res == FULL) {
    lower_y = input_lower_y;
    lower_x = input_lower_x;
    upper_y = input_upper_y;
    upper_x = input_upper_x;
    nrows = upper_y - lower_y;
    ncols = upper_x - lower_x;
    scale_x = 1;
    scale_y = 1;
  }

}

cv::Mat* read_tile(std::string filepath, Resolution res) {
    cv::Mat* tile_p_image = new cv::Mat();
    if(res == THUMBNAIL) {
      std::string path = std::string(filepath);
      path = path.replace(path.find(".bmp"), 4,".jpg");
      path = path.insert(path.find_last_of("/") + 1, "thumbnail_");
      (*tile_p_image) = imread_with_cache(path,CV_LOAD_IMAGE_GRAYSCALE);
	}
    if(res == FULL) {
      (*tile_p_image) = imread_with_cache(filepath, CV_LOAD_IMAGE_UNCHANGED);
    }
	return tile_p_image;
} 

cv::Mat render(section_data_t* section, std::string filename, int input_lower_x, int input_upper_x, int input_lower_y, int input_upper_y, Resolution res, bool write) {
  int lower_y, lower_x, upper_y, upper_x;
  int nrows, ncols;
  double scale_x, scale_y;

  //set parameteres
  set_render_parameters(lower_y, lower_x, upper_y, upper_x, ncols, nrows, scale_x, scale_y, section, filename, input_lower_x, input_upper_x, input_lower_y, input_upper_y, res);
  

  std::vector<renderTriangle> triangles;
  std::set<std::pair<int,int> > added_triangles;
  bool empty_image = true;
  for (int i = 0; i < section->n_tiles; i++) {
    tile_data_t tile = section->tiles[i];
    if (!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
      continue;
    }
    empty_image = false;
    for (int j = 0; j < tile.mesh_triangles->size(); j ++) {
      if (added_triangles.find((*tile.mesh_triangles)[j].key) == added_triangles.end()) {
        triangles.push_back((*tile.mesh_triangles)[j]);
        added_triangles.insert((*tile.mesh_triangles)[j].key);
      }
    }
  }

  if (empty_image) {
    return cv::Mat();
  }



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
		
  for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
    tile_data_t tile = section->tiles[i];

    if(!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
      continue;
    } 

	//read in tile file
    cv::Mat* tile_p_image = read_tile(tile.filepath, res);
    
	for (int _x = 0; _x < (*tile_p_image).size().width; _x++) {
      for (int _y = 0; _y < (*tile_p_image).size().height; _y++) {
        cv::Point2f p = cv::Point2f(_x*scale_x, _y*scale_y);
        cv::Point2f transformed_p = affine_transform(&tile, p);
        transformed_p = elastic_transform(&tile, &triangles, transformed_p);
        int x_c = (int)(transformed_p.x/scale_x + 0.5);
        int y_c = (int)(transformed_p.y/scale_y + 0.5);
        for (int k = -1; k < 2; k++) {
          for (int m = -1; m < 2; m++) {
            unsigned char val = tile_p_image->at<unsigned char>(_y, _x);
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
    tile_p_image->release();
  }
    for (int x = 0; x < section_p_out->size().width; x++) {
      for (int y = 0; y < section_p_out->size().height; y++) {
        if (section_p_out_ncount->at<unsigned short>(y,x) == 0) continue;
        section_p_out->at<unsigned char>(y, x) =
            section_p_out_sum->at<unsigned short>(y, x) / section_p_out_ncount->at<unsigned short>(y,x);
      }
    }

  //for (int y = 0; y < nrows; y++) {
  //  for (int x = 0; x < ncols; x++) {
  //    if (section->p_out->at<unsigned char>(y,x) < 0) {
  //      std::cout << y << " " << x << " unassigned" << std::endl;
  //    }
  //  }
  //}
  if (write) {
    cv::imwrite(filename, (*section_p_out));
  }
  return (*section_p_out);
}

void  update_tiles(section_data_t* section, cv::Mat * halo, std::string filename, int input_lower_x, int input_upper_x, int input_lower_y, int input_upper_y, Resolution res, bool write) {
  int lower_y, lower_x, upper_y, upper_x;
  int nrows, ncols;
  double scale_x, scale_y;

  printf("halo info is %d,%d,%d,%d\n", input_lower_x, input_lower_y, input_upper_x, input_upper_y);

  //set parameteres
  set_render_parameters(lower_y, lower_x, upper_y, upper_x, ncols, nrows, scale_x, scale_y, section, filename, input_lower_x, input_upper_x, input_lower_y, input_upper_y, res);
  

  std::vector<renderTriangle> triangles;
  std::set<std::pair<int,int> > added_triangles;
  bool empty_image = true;
	std::cout << "BOUNDS OF UPDATE " << lower_x << " " << lower_y << " " << upper_x << " " << upper_y << " " << std::endl;
  for (int i = 0; i < section->n_tiles; i++) {
    tile_data_t tile = section->tiles[i];
    if (!tile_completely_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
      continue; //FIX THIS LATER (should be copletely) 
    }
	std::cout << "TILE IN BOUNDS SKW" << std::endl;
    empty_image = false;
    for (int j = 0; j < tile.mesh_triangles->size(); j ++) {
      if (added_triangles.find((*tile.mesh_triangles)[j].key) == added_triangles.end()) {
        triangles.push_back((*tile.mesh_triangles)[j]);
        added_triangles.insert((*tile.mesh_triangles)[j].key);
      }
    }
  }

  for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
    tile_data_t tile = section->tiles[i];

    if(!tile_completely_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
      continue;
    } 
	//read in tile file
    cv::Mat* tile_p_image = read_tile(tile.filepath, res);

   	cv::Mat* new_tile = new cv::Mat();
  	(*new_tile).create((*tile_p_image).size().width, (*tile_p_image).size().height, CV_8UC1);

	for (int _x = 0; _x < (*tile_p_image).size().width; _x++) {
      for (int _y = 0; _y < (*tile_p_image).size().height; _y++) {
        cv::Point2f p = cv::Point2f(_x*scale_x, _y*scale_y);
        cv::Point2f transformed_p = affine_transform(&tile, p);
        transformed_p = elastic_transform(&tile, &triangles, transformed_p);
        int x_c = (int)(transformed_p.x/scale_x + 0.5);
        int y_c = (int)(transformed_p.y/scale_y + 0.5);

		new_tile->at<unsigned char>(_y, _x) = halo->at<unsigned short>(y_c - lower_y, x_c - lower_x);
		
      }	
    }
	if(write) {
		std::string s  = "tile" + std::to_string(i) + filename;
		cv::imwrite(s, (*new_tile));
	}
    tile_p_image->release();
  }
}




float matchTemplate(cv::Mat img1, cv::Mat img2) {
    cv::Mat result_SQDIFF, result_SQDIFF_NORMED, result_CCORR, result_CCORR_NORMED,
        result_CCOEFF, result_CCOEFF_NORMED;
    //cv::matchTemplate(img1, img2, result_SQDIFF, CV_TM_SQDIFF);
    //cv::matchTemplate(img1, img2, result_SQDIFF_NORMED, CV_TM_SQDIFF_NORMED);
    //cv::matchTemplate(img1, img2, result_CCORR, CV_TM_CCORR);
    //cv::matchTemplate(img1, img2, result_CCORR_NORMED, CV_TM_CCORR_NORMED);
    //cv::matchTemplate(img1, img2, result_CCOEFF, CV_TM_CCOEFF);
    cv::matchTemplate(img1, img2, result_CCOEFF_NORMED, CV_TM_CCOEFF_NORMED);
   
    //std::cout << "---------- MATCH TEMPLATE RESULTS (image size: " << img1.size().width << " " <<
    //    img1.size().height << " " << img2.size().width << " " << img2.size().height << ") -----------" << std::endl;
    //std::cout << "       SQDIFF: " <<        result_SQDIFF.at<float>(0,0) << std::endl;
    //std::cout << "SQDIFF_NORMED: " << result_SQDIFF_NORMED.at<float>(0,0) << std::endl;
    //std::cout << "        CCORR: " <<         result_CCORR.at<float>(0,0) << std::endl;
    //std::cout << " CCORR_NORMED: " <<  result_CCORR_NORMED.at<float>(0,0) << std::endl;
    //std::cout << "       CCOEFF: " <<        result_CCOEFF.at<float>(0,0) << std::endl;
    //std::cout << "CCOEFF_NORMED: " << result_CCOEFF_NORMED.at<float>(0,0) << std::endl;

    /*std::ofstream myfile;
    myfile.open ("output.csv", std::ios_base::app);
	myfile <<        result_SQDIFF.at<float>(0,0) << ","; 
	myfile << result_SQDIFF_NORMED.at<float>(0,0) << ","; 
	myfile <<         result_CCORR.at<float>(0,0) << ","; 
	myfile <<  result_CCORR_NORMED.at<float>(0,0) << ","; 
	myfile <<        result_CCOEFF.at<float>(0,0) << ","; 
	myfile << result_CCOEFF_NORMED.at<float>(0,0) << "\n"; 	
    myfile.close(); */
	return result_CCOEFF_NORMED.at<float>(0,0);
}


/*
call find bad triangles:
  1. render small boxes in a grid over the area of interest. 
  	  - for each small box render it, compute correlation, find center point of box P, query mesh triangles to find triangle T_P that P 
          is contained within, and then we increment triangle T_P.total_count and (if passes correlation check) increment T_P.valid_count.
  2. Render area of interest --- identical to previous code, but *after* we call elastic_tranform and set val, we call "is_bad_triangle" and if that returns true, then we overwrite val to be magic value (e.g. 0 or 255). 
*/
/* return unique key ID of bad renderTriangles (taken from tiles of the section) */


std::set<std::pair<int, int> > find_bad_triangles(std::vector<renderTriangle> * triangles, section_data_t* prev_section, section_data_t* section, int lower_x, int upper_x, int lower_y, int upper_y, int box_width, int box_height, Resolution res, std::map<std::pair<int, int>, float>& score_map) {
	std::set<std::pair<int, int> > bad_triangles;
	int count = 0;
	std::map<std::pair<int, int>, int>  num_valid;
	std::map<std::pair<int, int>, int>  num_invalid;
	for(int i = lower_y; i < upper_y - box_height; i += box_height) {
          for(int j = lower_x; j < upper_x - box_width; j += box_width) {
            std::string file1 = std::string("1box") + std::to_string(count) + std::string(".tif");
            std::string file2 = std::string("2box") + std::to_string(count) + std::string(".tif");

            cv::Mat im1 = render(section, file1, j, j + box_width, i, i + box_height, res, false);
            if (im1.empty()) continue;
	    cv::Mat im2 = render(prev_section, file2, j, j + box_width, i, i + box_height, res, false);
            if (im2.empty()) continue;
            count ++;
			float corr = matchTemplate(im1, im2);
			cv::Point2f middle(j-box_width/2, i-box_height/2);
			auto tri = findTriangle(triangles, middle, true); //this is super slow prob
			if(!std::get<0>(tri)) {
				//std::cout << "TRIANGLE NOT FOUND " << middle.x << " " << middle.y << std::endl;
			}
			std::pair<int, int> key = (*triangles)[0].key;
			if(num_valid.find(key) == num_valid.end()) {
				num_valid[key] = 0;
				num_invalid[key] = 0;
			}
			if(corr > 0.1) {
				num_valid[key] = num_valid[key] + 1; 
			} else {
				num_invalid[key] = num_invalid[key] + 1;
			}
		}
	}
	for(int i = 0; i < triangles->size(); i ++) {
		std::pair<int, int> key = (*triangles)[i].key;
		if(num_invalid[key] > 0 || num_valid[key] > 0) {
			//std::cout << "triangle valid " << num_valid[key] << " invalid " << num_invalid[key] << " key " << key.first << " " << key.second  << std::endl;
		}
		if(num_valid[key] < num_invalid[key]) {
			bad_triangles.insert(key);
		}
    	score_map[key] = num_invalid[key] / (num_valid[key]+num_invalid[key]+1.0);
	}
	return bad_triangles;
}

cv::Point2f triangle_midpoint(renderTriangle triangle) {
	// use q???
	float x = (triangle.q[0].x + triangle.q[1].x + triangle.q[2].x)/3;
	float y = (triangle.q[0].y + triangle.q[1].y + triangle.q[2].y)/3;
	return cv::Point2f(x, y);
}

std::vector<cv::Point2f> find_bad_triangles_midpoints(std::vector<renderTriangle> * triangles, section_data_t* prev_section, section_data_t* section, int lower_x, int upper_x, int lower_y, int upper_y, int box_width, int box_height, Resolution res, std::map<std::pair<int, int>, float>& score_map) {
	std::cout << "find bad midpoint method actual " << std::endl;
	std::set<std::pair<int, int> > bad_triangles;
	std::vector<cv::Point2f> bad_triangle_midpoints;
	int count = 0;
	std::map<std::pair<int, int>, int>  num_valid;
	std::map<std::pair<int, int>, int>  num_invalid;
	std::map<std::pair<int, int>, cv::Point2f> triangle_midpoints;
	for(int i = lower_y; i < upper_y - box_height; i += box_height) {
          for(int j = lower_x; j < upper_x - box_width; j += box_width) {
            std::string file1 = std::string("1box") + std::to_string(count) + std::string(".tif");
            std::string file2 = std::string("2box") + std::to_string(count) + std::string(".tif");

            cv::Mat im1 = render(section, file1, j, j + box_width, i, i + box_height, res, false);
            if (im1.empty()) continue;
	    cv::Mat im2 = render(prev_section, file2, j, j + box_width, i, i + box_height, res, false);
            if (im2.empty()) continue;
            count ++;
			float corr = matchTemplate(im1, im2);
			cv::Point2f middle(j-box_width/2, i-box_height/2);
			auto tri = findTriangle(triangles, middle, true); //this is super slow prob
			if(!std::get<0>(tri)) {
				//std::cout << "TRIANGLE NOT FOUND " << middle.x << " " << middle.y << std::endl;
			}
			std::pair<int, int> key = (*triangles)[0].key;
			if(num_valid.find(key) == num_valid.end()) {
				num_valid[key] = 0;
				num_invalid[key] = 0;
			 	triangle_midpoints[key] = triangle_midpoint((*triangles)[0]);   //rememebr the middle of the triangl
			}
			if(corr > 0.1) {
				num_valid[key] = num_valid[key] + 1; 
			} else {
				num_invalid[key] = num_invalid[key] + 1;
			}
		}
	}
	for(int i = 0; i < triangles->size(); i ++) {
		std::pair<int, int> key = (*triangles)[i].key;
		if(num_invalid[key] > 0 || num_valid[key] > 0) {
			//std::cout << "triangle valid " << num_valid[key] << " invalid " << num_invalid[key] << " key " << key.first << " " << key.second  << std::endl;
		}
		if(num_valid[key] < num_invalid[key]) {
			bad_triangles.insert(key);
		}
	}
	std::cout << "find midpoints actual inishe" << std::endl;
	std::set<std::pair<int,int>>::iterator it;
	for (it = bad_triangles.begin(); it != bad_triangles.end(); ++it) {
		bad_triangle_midpoints.push_back(triangle_midpoints[*it]);
	}
	return bad_triangle_midpoints;
}

std::set<std::pair<int, int> > find_bad_triangles_tile(std::vector<renderTriangle> * triangles, section_data_t* section, int lower_x, int upper_x, int lower_y, int upper_y, int box_width, int box_height, Resolution res, std::map<std::pair<int, int>, float>& score_map) {

	std::set<std::pair<int, int> > bad_triangles;
	std::map<std::pair<int, int>, int>  num_valid;
	std::map<std::pair<int, int>, int>  num_invalid;
	for(int i = lower_y; i < upper_y - box_height; i += box_height) {
      for(int j = lower_x; j < upper_x - box_width; j += box_width) {

						
	    for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
		  tile_data_t tile = section->tiles[i];
			
		  if(!tile_in_bounds(tile, lower_x, upper_x, lower_y, upper_y)) {
			  continue;
		  } 
		  printf("tile in bounds for box %d %d ", i, j);
		}

            //cv::Mat im1 = render(section, file1, j, j + box_width, i, i + box_height, res, false);
            //if (im1.empty()) continue;
	    	//cv::Mat im2 = render(prev_section, file2, j, j + box_width, i, i + box_height, res, false);
            //if (im2.empty()) continue;
            //count ++;

			//float corr = matchTemplate(im1, im2);
	
			//cv::Point2f middle(j-box_width/2, i-box_height/2);
			//auto tri = findTriangle(triangles, middle, true); //this is super slow prob
			//if(!std::get<0>(tri)) {
			//	std::cout << "TRIANGLE NOT FOUND " << middle.x << " " << middle.y << std::endl;
			//}
			//std::pair<int, int> key = (*triangles)[0].key;
			//if(num_valid.find(key) == num_valid.end()) {
			//	num_valid[key] = 0;
			//	num_invalid[key] = 0;
			//}
			//if(corr > 10000) {
			//	num_valid[key] = num_valid[key] + 1; 
			//} else {
			//	num_invalid[key] = num_invalid[key] + 1;
			//}
		}
	}
	//for(int i = 0; i < triangles->size(); i ++) {
	//	std::pair<int, int> key = (*triangles)[i].key;
	//	if(num_invalid[key] > 0 || num_valid[key] > 0) {
	//		std::cout << "triangle valid " << num_valid[key] << " invalid " << num_invalid[key] << " key " << key.first << " " << key.second  << std::endl;
	//	}
	//	if(num_valid[key] < num_invalid[key]) {
	//		bad_triangles.insert(key);
	//	}
    //    score_map[key] = num_invalid[key] / (num_valid[key]+num_invalid[key]+1.0);
	//}
	return bad_triangles;
}

/* rendering with error detection */
cv::Mat render_error(section_data_t* prev_section, section_data_t* section, std::string filename, int input_lower_x, int input_upper_x, int input_lower_y, int input_upper_y, int box_width, int box_height, Resolution res, bool write) {

	std::vector<renderTriangle> triangles;
	std::set<std::pair<int,int> > added_triangles;
	for (int i = 0; i < section->n_tiles; i++) {
	  	tile_data_t tile = section->tiles[i];
		if(!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
			continue;
	  	}
		for(int j = 0; j < tile.mesh_triangles->size(); j ++) {
			if(added_triangles.find((*tile.mesh_triangles)[j].key) == added_triangles.end()) {
				triangles.push_back((*tile.mesh_triangles)[j]);
				added_triangles.insert((*tile.mesh_triangles)[j].key);
			}
		}
	}
	int lower_y, lower_x, upper_y, upper_x;
	int nrows, ncols;
	double scale_x, scale_y;
	std::set<std::pair<int, int> > bad_triangles;
	//calculate scale
	if(res == THUMBNAIL) {
      std::string thumbnailpath = std::string(section->tiles[0].filepath);
    	  thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
    	  thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");
    	  cv::Mat thumbnail_img = imread_with_cache(thumbnailpath,CV_LOAD_IMAGE_UNCHANGED);
    	  cv::Mat img = imread_with_cache(section->tiles[0].filepath,CV_LOAD_IMAGE_UNCHANGED);
	  scale_x = (double)(img.size().width)/thumbnail_img.size().width;
    	  scale_y = (double)(img.size().height)/thumbnail_img.size().height;

      //create new matrix
	  lower_y = (int)(input_lower_y/scale_y + 0.5);
	  lower_x = (int)(input_lower_x/scale_x + 0.5);
	  upper_y = (int)(input_upper_y/scale_y + 0.5);
	  upper_x = (int)(input_upper_x/scale_x + 0.5);

	  nrows = (input_upper_y-input_lower_y)/scale_y;
    	  ncols = (input_upper_x-input_lower_x)/scale_x; 
	}
	if(res == FULL) {
	  lower_y = input_lower_y;
	  lower_x = input_lower_x;
	  upper_y = input_upper_y;
	  upper_x = input_upper_x;
	  nrows = upper_y - lower_y;
	  ncols = upper_x - lower_x;
	  scale_x = 1;
	  scale_y = 1;
	}
    std::map<std::pair<int, int>, float> score_map;	
    TFK_TIMER_VAR(timer_find_badt);
    TFK_START_TIMER(&timer_find_badt);
	bad_triangles = find_bad_triangles(&triangles, prev_section, section, input_lower_x, input_upper_x,
                                             input_lower_y, input_upper_y, box_height, box_width, res, score_map);	
    TFK_STOP_TIMER(&timer_find_badt, "Time to find bad triangles");

	cv::Mat* section_p_out = new cv::Mat();
	cv::Mat* section_p_out_mask = new cv::Mat();
	cv::Mat* tile_p_image = new cv::Mat();
	(*section_p_out).create(nrows, ncols, CV_8UC1);
	(*section_p_out_mask).create(nrows, ncols, CV_32F);
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
        section_p_out_mask->at<float>(y,x) = 0.0;
        section_p_out->at<unsigned char>(y,x) = 0;
        section_p_out_sum->at<unsigned short>(y,x) = 0;
        section_p_out_ncount->at<unsigned short>(y,x) = 0;
      }
    }

    for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
      tile_data_t tile = section->tiles[i];

      if (!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
		continue;
      } 

      if (res == THUMBNAIL) {	
        std::string path = std::string(tile.filepath);
    	path = path.replace(path.find(".bmp"), 4,".jpg");             		
        path = path.insert(path.find_last_of("/") + 1, "thumbnail_");
		(*tile_p_image) = imread_with_cache(path,CV_LOAD_IMAGE_GRAYSCALE);
      }
 
      if (res == FULL) {
        (*tile_p_image) = imread_with_cache(tile.filepath, CV_LOAD_IMAGE_UNCHANGED);
      }

      for (int _x = 0; _x < (*tile_p_image).size().width; _x++) {
        for (int _y = 0; _y < (*tile_p_image).size().height; _y++) {
          cv::Point2f p = cv::Point2f(_x*scale_x, _y*scale_y);
          cv::Point2f transformed_p = affine_transform(&tile, p);
          transformed_p = elastic_transform(&tile, &triangles, transformed_p);
          //renderTriangle tri = (*tile.mesh_triangles)[0]; //should have the triangle its in be correct
          renderTriangle tri = triangles[0]; //should have the triangle its in be correct

          int x_c = (int)(transformed_p.x/scale_x + 0.5);
          int y_c = (int)(transformed_p.y/scale_y + 0.5);
          unsigned char val = tile_p_image->at<unsigned char>(_y, _x);
          //if (bad_triangle) {
          //  val = 255;
          //}
				
          for (int k = -1; k < 2; k++) {
            for (int m = -1; m < 2; m++) {
              int x = x_c+k;
              int y = y_c+m;

              if (y-lower_y >= 0 && y-lower_y < nrows && x-lower_x >= 0 && x-lower_x < ncols) {
                //if (bad_triangle) {
                section_p_out_mask->at<float>(y-lower_y, x-lower_x) += score_map[tri.key];
                //}
                //section_p_out->at<unsigned char>(y-lower_y, x-lower_x) = val;
                section_p_out_sum->at<unsigned short>(y-lower_y, x-lower_x) += val;
                section_p_out_ncount->at<unsigned short>(y-lower_y, x-lower_x) += 1;
              }
            }
          }
		}
      }


      tile_p_image->release();
    }

    for (int x = 0; x < section_p_out->size().width; x++) {
      for (int y = 0; y < section_p_out->size().height; y++) {

        if (section_p_out_ncount->at<unsigned short>(y,x) == 0) continue;

        section_p_out_mask->at<float>(y, x) /= section_p_out_ncount->at<unsigned short>(y,x);
        section_p_out->at<unsigned char>(y, x) =
            section_p_out_sum->at<unsigned short>(y, x) / section_p_out_ncount->at<unsigned short>(y,x);
      }
    }

     cv::Mat heatmap_image = apply_heatmap_to_grayscale(section_p_out, section_p_out_mask, nrows, ncols); 

	bool ret = cv::imwrite(filename, (*section_p_out));
        std::string filename_mask = filename.replace(filename.find(".tif"),4,".png");
	if(write) {
		ret = cv::imwrite(filename_mask, heatmap_image);
	}
	return (*section_p_out);
}

bool tiles_overlap(tile_data_t *tile_1, tile_data_t *tile_2) {
  if ((max_x(tile_1) < min_x(tile_2)) || (max_x(tile_2) < min_x(tile_1))) {
    return false;
  }
  if ((max_y(tile_1) < min_y(tile_2)) || (max_y(tile_2) < min_y(tile_1))) {
    return false;
  }
  return true;
}

/* calculate the error between two tiles
1 is a perect match
-2 means they do not overlap
*/
float error_tile_pair(tile_data_t *tile_1, tile_data_t *tile_2) {
  if (!(tiles_overlap(tile_1, tile_2))) {
    return -2;
  }


  cv::Mat tile_p_image_1;
  cv::Mat tile_p_image_2;
  tile_p_image_1 = imread_with_cache(tile_1->filepath, CV_LOAD_IMAGE_UNCHANGED);
  tile_p_image_2 = imread_with_cache(tile_2->filepath, CV_LOAD_IMAGE_UNCHANGED);

  int nrows = min(max_y(tile_1), max_y(tile_2)) - max(min_y(tile_1), min_y(tile_2));
  int ncols = min(max_x(tile_1), max_x(tile_2)) - max(min_x(tile_1), min_x(tile_2));
  if ((nrows <= 0) || (ncols <= 0) ) {
    return -2;
  }
  int offset_x = max(min_x(tile_1), min_x(tile_2));
  int offset_y = max(min_y(tile_1), min_y(tile_2));
  cv::Mat transform_1 = cv::Mat::zeros(nrows, ncols, CV_8UC1);
  cv::Mat transform_2 = cv::Mat::zeros(nrows, ncols, CV_8UC1);

  // make the transformed images in the same size with the same cells in the same locations
  for (int _y = 0; _y < tile_p_image_1.rows; _y++) {
    for (int _x = 0; _x < tile_p_image_1.cols; _x++) {
      cv::Point2f p = cv::Point2f(_x, _y);
      cv::Point2f transformed_p = affine_transform(tile_1, p);

      int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
      int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
      if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
        transform_1.at<unsigned char>(y_c, x_c) +=
           tile_p_image_1.at<unsigned char>(_y, _x);
      }
    }
  }

  for (int _y = 0; _y < tile_p_image_2.rows; _y++) {
    for (int _x = 0; _x < tile_p_image_2.cols; _x++) {
      cv::Point2f p = cv::Point2f(_x, _y);
      cv::Point2f transformed_p = affine_transform(tile_2, p);
      
      int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
      int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
      if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
        transform_2.at<unsigned char>(y_c, x_c) +=
           tile_p_image_2.at<unsigned char>(_y, _x);
      }
    }
  }

  // clear any location which only has a value for one of them
  // note that the transforms are the same size
  for (int _y = 0; _y < transform_1.rows; _y++) {
    for (int _x = 0; _x < transform_1.cols; _x++) {
      if (transform_2.at<unsigned char>(_y, _x) == 0) {
       transform_1.at<unsigned char>(_y, _x) = 0;
      }
      else if (transform_1.at<unsigned char>(_y, _x) == 0) {
       transform_2.at<unsigned char>(_y, _x) = 0;
      }
    }
  }

  float result = matchTemplate(transform_1 , transform_2 );
  return result;
}

void get_all_error_pairs(section_data_t* section) {
  cilk_for (int i = 0; i < section->n_tiles; i++) {
    for (int j = i+1; j < section->n_tiles; j++) {
      if (!(tiles_overlap(&(section->tiles[i]), &(section->tiles[j])))) {
        continue;
      }
      double corr = error_tile_pair(&(section->tiles[i]), &(section->tiles[j]));
      if (corr >= -1) {
        __sync_fetch_and_add(&(section->tiles[i].number_overlaps),1);
        __sync_fetch_and_add(&(section->tiles[j].number_overlaps),1);
        int corr_slot = (int) (10*(corr+1));
         __sync_fetch_and_add(&(section->tiles[i].corralation_counts[corr_slot]),1);
         __sync_fetch_and_add(&(section->tiles[j].corralation_counts[corr_slot]),1);
      }
    }
  }
  for (int i = 0; i < section->n_tiles; i++) {
    printf("Tile %d overlaps %d times, corr counts are ",i, section->tiles[i].number_overlaps);
    for (int j = 0; j < 20; j++) {
      printf("%d",section->tiles[i].corralation_counts[j]);
    }
    printf("\n");
  }
}


/* rendering for a 2d section*/
cv::Mat render_2d(section_data_t* section, std::string filename, int input_lower_x, int input_upper_x, int input_lower_y, int input_upper_y, int box_width, int box_height, Resolution res, bool write) {


  int lower_y, lower_x, upper_y, upper_x;
  int nrows, ncols;
  double scale_x, scale_y;
  //calculate scale
  if(res == THUMBNAIL) {
      std::string thumbnailpath = std::string(section->tiles[0].filepath);
        thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
        thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");
        cv::Mat thumbnail_img = imread_with_cache(thumbnailpath,CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat img = imread_with_cache(section->tiles[0].filepath,CV_LOAD_IMAGE_UNCHANGED);
    scale_x = (double)(img.size().width)/thumbnail_img.size().width;
        scale_y = (double)(img.size().height)/thumbnail_img.size().height;

      //create new matrix
    lower_y = (int)(input_lower_y/scale_y + 0.5);
    lower_x = (int)(input_lower_x/scale_x + 0.5);
    upper_y = (int)(input_upper_y/scale_y + 0.5);
    upper_x = (int)(input_upper_x/scale_x + 0.5);

    nrows = (input_upper_y-input_lower_y)/scale_y;
        ncols = (input_upper_x-input_lower_x)/scale_x; 
  }
  if(res == FULL) {
    lower_y = input_lower_y;
    lower_x = input_lower_x;
    upper_y = input_upper_y;
    upper_x = input_upper_x;
    nrows = upper_y - lower_y;
    ncols = upper_x - lower_x;
    scale_x = 1;
    scale_y = 1;
  }

  cv::Mat* section_p_out = new cv::Mat();
  cv::Mat* section_p_out_mask = new cv::Mat();
  cv::Mat* tile_p_image = new cv::Mat();
  (*section_p_out).create(nrows, ncols, CV_8UC1);
  (*section_p_out_mask).create(nrows, ncols, CV_32F);
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
        section_p_out_mask->at<float>(y,x) = 0.0;
        section_p_out->at<unsigned char>(y,x) = 0;
        section_p_out_sum->at<unsigned short>(y,x) = 0;
        section_p_out_ncount->at<unsigned short>(y,x) = 0;
      }
    }

    for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
      tile_data_t tile = section->tiles[i];

      if (!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
    continue;
      } 

      if (res == THUMBNAIL) { 
        std::string path = std::string(tile.filepath);
      path = path.replace(path.find(".bmp"), 4,".jpg");                 
        path = path.insert(path.find_last_of("/") + 1, "thumbnail_");
    (*tile_p_image) = imread_with_cache(path,CV_LOAD_IMAGE_GRAYSCALE);
      }
 
      if (res == FULL) {
        (*tile_p_image) = imread_with_cache(tile.filepath, CV_LOAD_IMAGE_UNCHANGED);
      }

      for (int _x = 0; _x < (*tile_p_image).size().width; _x++) {
        for (int _y = 0; _y < (*tile_p_image).size().height; _y++) {
          cv::Point2f p = cv::Point2f(_x*scale_x, _y*scale_y);
          cv::Point2f transformed_p = affine_transform(&tile, p);

          int x_c = (int)(transformed_p.x/scale_x + 0.5);
          int y_c = (int)(transformed_p.y/scale_y + 0.5);
          unsigned char val = tile_p_image->at<unsigned char>(_y, _x);
        
          for (int k = -1; k < 2; k++) {
            for (int m = -1; m < 2; m++) {
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


      tile_p_image->release();
    }

    for (int x = 0; x < section_p_out->size().width; x++) {
      for (int y = 0; y < section_p_out->size().height; y++) {

        if (section_p_out_ncount->at<unsigned short>(y,x) == 0) continue;

        section_p_out_mask->at<float>(y, x) /= section_p_out_ncount->at<unsigned short>(y,x);
        section_p_out->at<unsigned char>(y, x) =
            section_p_out_sum->at<unsigned short>(y, x) / section_p_out_ncount->at<unsigned short>(y,x);
      }
    }
 

  bool ret = cv::imwrite(filename, (*section_p_out));
  std::string filename_mask = filename.replace(filename.find(".tif"),4,".png");
  if(write) {
    ret = cv::imwrite(filename_mask, (*section_p_out));
  }
  return (*section_p_out);
}

/* rendering with error detection for TILES*/
cv::Mat render_error_tiles(section_data_t* section, std::string filename, int input_lower_x, int input_upper_x, int input_lower_y, int input_upper_y, int box_width, int box_height, Resolution res) {

	std::vector<renderTriangle> triangles;
	std::set<std::pair<int,int> > added_triangles;
	for (int i = 0; i < section->n_tiles; i++) {
	  	tile_data_t tile = section->tiles[i];
		if(!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
			continue;
	  	}
		for(int j = 0; j < tile.mesh_triangles->size(); j ++) {
			if(added_triangles.find((*tile.mesh_triangles)[j].key) == added_triangles.end()) {
				triangles.push_back((*tile.mesh_triangles)[j]);
				added_triangles.insert((*tile.mesh_triangles)[j].key);
			}
		}
	}
	int lower_y, lower_x, upper_y, upper_x;
	int nrows, ncols;
	double scale_x, scale_y;
	std::set<std::pair<int, int> > bad_triangles;
	//calculate scale
	if(res == THUMBNAIL) {
      std::string thumbnailpath = std::string(section->tiles[0].filepath);
    	  thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
    	  thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");
    	  cv::Mat thumbnail_img = imread_with_cache(thumbnailpath,CV_LOAD_IMAGE_UNCHANGED);
    	  cv::Mat img = imread_with_cache(section->tiles[0].filepath,CV_LOAD_IMAGE_UNCHANGED);
	  scale_x = (double)(img.size().width)/thumbnail_img.size().width;
    	  scale_y = (double)(img.size().height)/thumbnail_img.size().height;

      //create new matrix
	  lower_y = (int)(input_lower_y/scale_y + 0.5);
	  lower_x = (int)(input_lower_x/scale_x + 0.5);
	  upper_y = (int)(input_upper_y/scale_y + 0.5);
	  upper_x = (int)(input_upper_x/scale_x + 0.5);

	  nrows = (input_upper_y-input_lower_y)/scale_y;
    	  ncols = (input_upper_x-input_lower_x)/scale_x; 
	}
	if(res == FULL) {
	  lower_y = input_lower_y;
	  lower_x = input_lower_x;
	  upper_y = input_upper_y;
	  upper_x = input_upper_x;
	  nrows = upper_y - lower_y;
	  ncols = upper_x - lower_x;
	  scale_x = 1;
	  scale_y = 1;
	}
    std::map<std::pair<int, int>, float> score_map;	
    TFK_TIMER_VAR(timer_find_badt);
    TFK_START_TIMER(&timer_find_badt);
	bad_triangles = find_bad_triangles_tile(&triangles, section, input_lower_x, input_upper_x,
                                             input_lower_y, input_upper_y, box_height, box_width, res, score_map);	
    TFK_STOP_TIMER(&timer_find_badt, "Time to find bad triangles");

	cv::Mat* section_p_out = new cv::Mat();
	cv::Mat* section_p_out_mask = new cv::Mat();
	cv::Mat* tile_p_image = new cv::Mat();
	(*section_p_out).create(nrows, ncols, CV_8UC1);
	(*section_p_out_mask).create(nrows, ncols, CV_32F);
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
        section_p_out_mask->at<float>(y,x) = 0.0;
        section_p_out->at<unsigned char>(y,x) = 0;
        section_p_out_sum->at<unsigned short>(y,x) = 0;
        section_p_out_ncount->at<unsigned short>(y,x) = 0;
      }
    }

    for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
      tile_data_t tile = section->tiles[i];

      if (!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
		continue;
      } 

      if (res == THUMBNAIL) {	
        std::string path = std::string(tile.filepath);
    	path = path.replace(path.find(".bmp"), 4,".jpg");             		
        path = path.insert(path.find_last_of("/") + 1, "thumbnail_");
		(*tile_p_image) = imread_with_cache(path,CV_LOAD_IMAGE_GRAYSCALE);
      }
 
      if (res == FULL) {
        (*tile_p_image) = imread_with_cache(tile.filepath, CV_LOAD_IMAGE_UNCHANGED);
      }

      for (int _x = 0; _x < (*tile_p_image).size().width; _x++) {
        for (int _y = 0; _y < (*tile_p_image).size().height; _y++) {
          cv::Point2f p = cv::Point2f(_x*scale_x, _y*scale_y);
          cv::Point2f transformed_p = affine_transform(&tile, p);
          transformed_p = elastic_transform(&tile, &triangles, transformed_p);
          renderTriangle tri = triangles[0]; //should have the triangle its in be correct

          int x_c = (int)(transformed_p.x/scale_x + 0.5);
          int y_c = (int)(transformed_p.y/scale_y + 0.5);
          unsigned char val = tile_p_image->at<unsigned char>(_y, _x);
				
          for (int k = -1; k < 2; k++) {
            for (int m = -1; m < 2; m++) {
              int x = x_c+k;
              int y = y_c+m;

              if (y-lower_y >= 0 && y-lower_y < nrows && x-lower_x >= 0 && x-lower_x < ncols) {
                //if (bad_triangle) {
                section_p_out_mask->at<float>(y-lower_y, x-lower_x) += score_map[tri.key];
                //}
                //section_p_out->at<unsigned char>(y-lower_y, x-lower_x) = val;
                section_p_out_sum->at<unsigned short>(y-lower_y, x-lower_x) += val;
                section_p_out_ncount->at<unsigned short>(y-lower_y, x-lower_x) += 1;
              }
            }
          }
		}
      }
      tile_p_image->release();
    }

    for (int x = 0; x < section_p_out->size().width; x++) {
      for (int y = 0; y < section_p_out->size().height; y++) {

        if (section_p_out_ncount->at<unsigned short>(y,x) == 0) continue;

          section_p_out_mask->at<float>(y, x) /= section_p_out_ncount->at<unsigned short>(y,x);
          section_p_out->at<unsigned char>(y, x) =
          section_p_out_sum->at<unsigned short>(y, x) / section_p_out_ncount->at<unsigned short>(y,x);
      }
    }

     cv::Mat heatmap_image = apply_heatmap_to_grayscale(section_p_out, section_p_out_mask, nrows, ncols); 

	bool ret = cv::imwrite(filename, (*section_p_out));
        printf("success of first write is %d\n", ret);
        std::string filename_mask = filename.replace(filename.find(".tif"),4,".png");
        printf("Writing filename %s\n", filename_mask.c_str());
	ret = cv::imwrite(filename_mask, heatmap_image);
        printf("success of second write is %d\n", ret);
        printf("after imwrite\n");
	return (*section_p_out);
}
std::vector<cv::Point2f> find_section_bad_triangles_midpoints(section_data_t* prev_section, section_data_t* section, std::string filename, int input_lower_x, int input_upper_x, int input_lower_y, int input_upper_y, int box_width, int box_height, Resolution res) {

	std::cout << "find bad midpoints " << std::endl;
	std::vector<renderTriangle> triangles;
	std::set<std::pair<int,int> > added_triangles;
	for (int i = 0; i < section->n_tiles; i++) {
	  	tile_data_t tile = section->tiles[i];
		if(!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
			continue;
	  	}
		for(int j = 0; j < tile.mesh_triangles->size(); j ++) {
			if(added_triangles.find((*tile.mesh_triangles)[j].key) == added_triangles.end()) {
				triangles.push_back((*tile.mesh_triangles)[j]);
				added_triangles.insert((*tile.mesh_triangles)[j].key);
			}
		}
	}
	std::vector<cv::Point2f> bad_triangles;
    std::map<std::pair<int, int>, float> score_map;	//not used, but maybe can return for recomputation later?
    TFK_TIMER_VAR(timer_find_badt);
    TFK_START_TIMER(&timer_find_badt);
	bad_triangles = find_bad_triangles_midpoints(&triangles, prev_section, section, input_lower_x, input_upper_x,
                                             input_lower_y, input_upper_y, box_height, box_width, res, score_map);	
    TFK_STOP_TIMER(&timer_find_badt, "Time to find bad triangles");
    printf("in methods %d\n", bad_triangles.size());
	return bad_triangles;
}



/* finds and returns bad triangles of the section */
std::set<std::pair<int, int> > find_section_bad_triangles(section_data_t* prev_section, section_data_t* section, std::string filename, int input_lower_x, int input_upper_x, int input_lower_y, int input_upper_y, int box_width, int box_height, Resolution res) {

	std::vector<renderTriangle> triangles;
	std::set<std::pair<int,int> > added_triangles;
	for (int i = 0; i < section->n_tiles; i++) {
	  	tile_data_t tile = section->tiles[i];
		if(!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
			continue;
	  	}
		for(int j = 0; j < tile.mesh_triangles->size(); j ++) {
			if(added_triangles.find((*tile.mesh_triangles)[j].key) == added_triangles.end()) {
				triangles.push_back((*tile.mesh_triangles)[j]);
				added_triangles.insert((*tile.mesh_triangles)[j].key);
			}
		}
	}
	std::set<std::pair<int, int> > bad_triangles;
    std::map<std::pair<int, int>, float> score_map;	//not used, but maybe can return for recomputation later?
    TFK_TIMER_VAR(timer_find_badt);
    TFK_START_TIMER(&timer_find_badt);
	bad_triangles = find_bad_triangles(&triangles, prev_section, section, input_lower_x, input_upper_x,
                                             input_lower_y, input_upper_y, box_height, box_width, res, score_map);	
    TFK_STOP_TIMER(&timer_find_badt, "Time to find bad triangles");
	return bad_triangles;
}



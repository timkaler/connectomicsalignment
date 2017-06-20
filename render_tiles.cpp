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

float sign(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool inTriangle(cv::Point2f p, cv::Point2f triangle[3]) {
	bool b1 = sign(p, triangle[0], triangle[1]) < 0;
	bool b2 = sign(p, triangle[0], triangle[2]) < 0;
	bool b3 = sign(p, triangle[1], triangle[2]) < 0;
	return ((b1 == b2) && (b2 == b3));
}

cv::Point2f elastic_transform(tile_data_t *t, cv::Point2f point) {
	std::vector<renderTriangle>*mesh_triangles = t->mesh_triangles;
	int index = -1;
	for(int i = 0; i < mesh_triangles->size(); i ++) {
		if(inTriangle(point, (*mesh_triangles)[i].p)) {
			index = i;
			int j = i;
			while(j > 0) {
				renderTriangle * temp = &(*mesh_triangles)[j - 1];
				(*mesh_triangles)[j - 1] = (*mesh_triangles)[j];
				(*mesh_triangles)[j] = *temp;
				j --;
			} 
			//swap first 
			//bubble sort up 
//divide and conquer 
//100x100 boxes
			break;
		}
	}

	if(index < 0) {
		std::cout << " TRIANGLE NOT FOUND " << std::endl;
	}
	float p_y0 = (*mesh_triangles)[index].p[0].y;
    float p_y1 = (*mesh_triangles)[index].p[1].y;
	float p_y2 = (*mesh_triangles)[index].p[2].y;
	float p_x0 = (*mesh_triangles)[index].p[0].x;
	float p_x1 = (*mesh_triangles)[index].p[1].x;
	float p_x2 = (*mesh_triangles)[index].p[2].x;
	float q_y0 = (*mesh_triangles)[index].q[0].y;
    float q_y1 = (*mesh_triangles)[index].q[1].y;
	float q_y2 = (*mesh_triangles)[index].q[2].y;
	float q_x0 = (*mesh_triangles)[index].q[0].x;
	float q_x1 = (*mesh_triangles)[index].q[1].x;
	float q_x2 = (*mesh_triangles)[index].q[2].x;
	
	cv::Mat P = cv::Mat(3, 3, cv::DataType<float>::type);
	cv::Mat Q = cv::Mat(3,3, cv::DataType<float>::type);
	P.at<float>(0,0) = p_x0;
	P.at<float>(1,0) = p_y0;
	P.at<float>(2,0) = 1;
	P.at<float>(0,1) = p_x1;
	P.at<float>(1,1) = p_y1;
	P.at<float>(2,1) = 1;
	P.at<float>(0,2) = p_x2;
	P.at<float>(1,2) = p_y2;
	P.at<float>(2,2) = 1; 
	Q.at<float>(0,0) = q_x0;
	Q.at<float>(1,0) = q_y0;
	Q.at<float>(2,0) = 1;
	Q.at<float>(0,1) = q_x1;
	Q.at<float>(1,1) = q_y1;
	Q.at<float>(2,1) = 1;
	Q.at<float>(0,2) = q_x2;
	Q.at<float>(1,2) = q_y2;
	Q.at<float>(2,2) = 1; 

	//Barycentric in mesh.h

	cv::Mat T = P*Q.inv();

	float new_x = point.x*T.at<float>(0,0) + point.y*T.at<float>(0,1) + T.at<float>(0,2);
	float new_y = point.x*T.at<float>(1,0) + point.y*T.at<float>(1,1) + T.at<float>(1,2);

	return cv::Point2f(new_x, new_y);
}

cv::Point2f affine_transform(tile_data_t * t, cv::Point2f point) {
	float new_x = point.x*t->a00 + point.y*t->a01 + t->offset_x;
	float new_y = point.x*t->a10 + point.y*t->a11 + t->offset_y;
	return cv::Point2f(new_x, new_y);
}
/*
cv::Point2f tile_transform_point_scale(tile_data_t *t, cv::Point2f point, float scale) {
	float new_x = scale*(point.x*t->a00 + point.y*t->a01 + t->offset_x);
	float new_y = scale*(point.x*t->a10 + point.y*t->a11 + t->offset_y);
	return cv::Point2f(new_x, new_y);
}*/

void output_section_image_thumbnail(section_data_t* section, std::string filename, int lower_x, int upper_x, int lower_y, int upper_y) {
	//calculate scale
    std::string thumbnailpath = std::string(section->tiles[0].filepath);
    thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
    thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");
    cv::Mat thumbnail_img = cv::imread(thumbnailpath,CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat img = cv::imread(section->tiles[0].filepath,CV_LOAD_IMAGE_UNCHANGED);
	double scale_x = (double)(img.size().width)/thumbnail_img.size().width;
    double scale_y = (double)(img.size().height)/thumbnail_img.size().height;

//HAVE DIFFERENT SCALE FOR X AND Y
scale_x = scale_y;
    std::cout << "SCALE OF THUMBNAIL: " << scale_x << " " << scale_y  << std::endl;
   	int width = thumbnail_img.size().width;
	int height = thumbnail_img.size().height; 
	std::cout << "thumbnail width: " << width << " height: " << height << " | original width: " << img.size().width << " height: " << img.size().height << std::endl;

	//create new matrix
	int nrows = upper_y-lower_y;
    int ncols = upper_x-lower_x; 
	section->p_out = new cv::Mat();
	(*section->p_out).create(nrows, ncols, CV_8UC1);
    for (int y = 0; y < nrows; y++) {
      	for (int x = 0; x < ncols; x++) {
        	section->p_out->at<unsigned char>(y,x) = 0;
      	}
    }

	for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
	  	tile_data_t tile = section->tiles[i];
    	std::string path = std::string(tile.filepath);
    	path = path.replace(path.find(".bmp"), 4,".jpg");             		
		path = path.insert(path.find_last_of("/") + 1, "thumbnail_");
		(*tile.p_image) = cv::imread(path,CV_LOAD_IMAGE_GRAYSCALE);
		int max_x = 0;		
		for (int _x = 0; _x < width*scale_x; _x++) {
	    	for (int _y = 0; _y < height*scale_y; _y++) {
				cv::Point2f p = cv::Point2f(_x, _y);
				cv::Point2f transformed_p = affine_transform(&tile, p);
				int x = transformed_p.x;
				int y = transformed_p.y;
				if(y >= upper_y || y < lower_y || x >= upper_x || x < lower_x) {
					continue;
				}
				unsigned char val =
	        		tile.p_image->at<unsigned char>((int)(_y/scale_y), (int)(_x/scale_x));
				if(_y/scale_y >= height || _x/scale_x >= width) {
				//	std::cout << "OUT OF BOUNDS " << _y/scale_y << " " << _x/scale_x << std::endl;
				}
				section->p_out->at<unsigned char>(y-lower_y, x-lower_x) = val;
				if(x-lower_x> max_x) max_x = x-lower_x;
	    	}	
	  	}
		std::cout << "MAX X: " << max_x << " , width " << img.size().width << std::endl;
	  	tile.p_image->release();
	}
	cv::imwrite(filename, (*section->p_out));
}
void output_section_image_affine(section_data_t* section, std::string filename, int lower_x, int upper_x, int lower_y, int upper_y, bool thumbnail) {
    //create new matrix
	int nrows = upper_y-lower_y; 
    int ncols = upper_x-lower_x; 
	section->p_out = new cv::Mat();
	(*section->p_out).create(nrows, ncols, CV_8UC1);
    for (int y = 0; y < nrows; y++) {
      	for (int x = 0; x < ncols; x++) {
        	section->p_out->at<unsigned char>(y,x) = 0;
      	}
    }
 
	//loop through tiles
	for (int i = 0; i < section->n_tiles; i++) {
	  	tile_data_t tile = section->tiles[i];
	  	(*tile.p_image) = cv::imread(
	    tile.filepath,
	    CV_LOAD_IMAGE_UNCHANGED);
		
	
		for (int _x = 0; _x < 3128; _x++) {
	    	for (int _y = 0; _y < 2724; _y++) {
				cv::Point2f p = cv::Point2f(_x, _y);
				cv::Point2f transformed_p = affine_transform(&tile, p);
				int x = transformed_p.x;
				int y = transformed_p.y;
				if(y >= upper_y || y < lower_y || x >= upper_x || x < lower_x) {
					continue;
				}
				unsigned char val =
	        		tile.p_image->at<unsigned char>(_y, _x);
				section->p_out->at<unsigned char>(y-lower_y, x-lower_x) = val;
	    	}	
	  	}
	  	tile.p_image->release();
	}
	cv::Mat outImage2 = resize(0.5, *section->p_out);
	cv::imwrite(filename, outImage2);
}

void output_section_image_affine_elastic(section_data_t* section, std::string filename, int lower_x, int upper_x, int lower_y, int upper_y, bool thumbnail) {
    //create new matrix
	int nrows = upper_y-lower_y; 
    int ncols = upper_x-lower_x; 
	section->p_out = new cv::Mat();
	(*section->p_out).create(nrows, ncols, CV_8UC1);
    for (int y = 0; y < nrows; y++) {
      	for (int x = 0; x < ncols; x++) {
        	section->p_out->at<unsigned char>(y,x) = 0;
      	}
    }
 
	//loop through tiles
	for (int i = 0; i < section->n_tiles; i++) {
	  	tile_data_t tile = section->tiles[i];
	  	(*tile.p_image) = cv::imread(
	    tile.filepath,
	    CV_LOAD_IMAGE_UNCHANGED);
		
	
		for (int _x = 0; _x < 3128; _x++) {
	    	for (int _y = 0; _y < 2724; _y++) {
				cv::Point2f p = cv::Point2f(_x, _y);
				//cv::Point2f transformed_p = affine_transform(&tile, p);
				//std::cout << "calling elastic... " << std::endl;
				cv::Point2f	transformed_p = elastic_transform(&tile, p);
				int x = transformed_p.x;
				int y = transformed_p.y;
				if(y >= upper_y || y < lower_y || x >= upper_x || x < lower_x) {
					continue;
				}
				unsigned char val =
	        		tile.p_image->at<unsigned char>(_y, _x);
				section->p_out->at<unsigned char>(y-lower_y, x-lower_x) = val;
	    	}	
	  	}
	  	tile.p_image->release();
	}
			
	cv::Mat outImage2 = resize(0.5, *section->p_out);
	cv::imwrite(filename, outImage2);
}

void output_section_image(section_data_t* section, std::string filename) {
	int min_x = section->tiles[0].x_start;
	int max_x = section->tiles[0].x_finish;
	int min_y = section->tiles[0].y_start;
	int max_y = section->tiles[0].y_finish;
	for (int i = 1; i < section->n_tiles; i ++) {
		tile_data_t tile = section->tiles[i];
		if(min_x > tile.x_start) {
			min_x = tile.x_start;
		}
		if(max_x < tile.x_finish) {
			max_x = tile.x_finish;
		}
		if(min_y > tile.y_start) {
			min_y = tile.y_start;
		}
		if(max_y < tile.y_finish) {
			max_y = tile.y_finish;
		}
	}
	int nrows = max_y-min_y;
	int ncols = max_x-min_x;	
	section->p_out = new cv::Mat();
	(*section->p_out).create(nrows, ncols, CV_8UC1);
	
	for (int i = 0; i < section->n_tiles; i++) {
	  tile_data_t tile = section->tiles[i];
	  (*tile.p_image).create(3128, 2724, CV_8UC1);
	  (*tile.p_image) = cv::imread(
	      tile.filepath,
	      CV_LOAD_IMAGE_UNCHANGED);
	
	  for (int y = tile.y_start; y < tile.y_finish; y++) {
	    for (int x = tile.x_start; x < tile.x_finish; x++) {
			unsigned char val =
	        	tile.p_image->at<unsigned char>((int)(y-tile.y_start), (int)(x-tile.x_start));
	      //add check to see if in bounds (box)
			section->p_out->at<unsigned char>(y-min_y, x-min_x) = val;
	    }
	  }
	  tile.p_image->release();
	}
	
	cv::Mat outImage2;
	cv::resize((*section->p_out), outImage2, cv::Size(), 0.5,0.5);
	cv::imwrite(filename, outImage2);
} 

void output_section_image_bounded(section_data_t* section, std::string filename, int lower_x, int upper_x, int lower_y, int upper_y, bool thumbnail) {
//find min and max (currently default inputted)
	int min_x = section->tiles[0].x_start;
	int max_x = section->tiles[0].x_finish;
	int min_y = section->tiles[0].y_start;
	int max_y = section->tiles[0].y_finish;
	for (int i = 1; i < section->n_tiles; i ++) {
		tile_data_t tile = section->tiles[i];
		if(min_x > tile.x_start) {
			min_x = tile.x_start;
		}
		if(max_x < tile.x_finish) {
			max_x = tile.x_finish;
		}
		if(min_y > tile.y_start) {
			min_y = tile.y_start;
		}
		if(max_y < tile.y_finish) {
			max_y = tile.y_finish;
		}
	}
	if(min_x > lower_x || max_x < upper_x || min_y > lower_y || max_y < upper_y) {
		std::cout << "BOUNDS BAD " << std::endl;
		return;
	}
	//create new matrix
	int ncols = upper_x-lower_x;	
	int nrows = upper_y-lower_y;
	section->p_out = new cv::Mat();
	(*section->p_out).create(nrows, ncols, CV_8UC1);
    for (int y = 0; y < nrows; y++) {
      	for (int x = 0; x < ncols; x++) {
        	section->p_out->at<unsigned char>(y,x) = 0;
      	}
    }
        
	std::set<std::pair<int,int> > set_pixels;
	for (int i = 0; i < section->n_tiles; i++) {
	  	tile_data_t tile = section->tiles[i];
	  	(*tile.p_image).create(3128, 2724, CV_8UC1);
	  	(*tile.p_image) = cv::imread(
	  	    tile.filepath,
	  	    CV_LOAD_IMAGE_UNCHANGED);
	  	for (int y = tile.y_start; y < tile.y_finish; y++) {
	  	  	for (int x = tile.x_start; x < tile.x_finish; x++) {
	  	  		if(y >= upper_y || y < lower_y || x >= upper_x || x < lower_x) {
	  	  			continue;
	  	  		}
      	  	    if (set_pixels.find(std::make_pair(x,y)) != set_pixels.end()) continue;
      	  	    set_pixels.insert(std::make_pair(x,y));
	  	  		unsigned char val =
	  	  	    	tile.p_image->at<unsigned char>((int)(y-tile.y_start), (int)(x-tile.x_start));
	  	  		section->p_out->at<unsigned char>(y-lower_y, x-lower_x) = val;
	  	  	}
	  	}
	  	tile.p_image->release();
	}
	
	cv::Mat outImage2;
	cv::resize((*section->p_out), outImage2, cv::Size(), 0.5,0.5);
	cv::imwrite(filename, outImage2);
}


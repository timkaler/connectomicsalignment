#include <iostream>

#include <fstream>
//#include "mesh.h"

enum Resolution {THUMBNAIL, FULL};
enum Averaging {VOTING, GEOMETRIC};

static float Dot(cv::Point2f a, cv::Point2f b) {
  return a.x*b.x + a.y*b.y;
}

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

std::tuple<bool, float, float, float> findTriangle(std::vector<renderTriangle>*mesh_triangles, cv::Point2f point) {
	for(int i = 0; i < mesh_triangles->size(); i ++) {
      float u, v, w;	  
	  cv::Point2f a = (*mesh_triangles)[i].p[0];
      cv::Point2f b = (*mesh_triangles)[i].p[1];
      cv::Point2f c = (*mesh_triangles)[i].p[2];
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

cv::Point2f elastic_transform(tile_data_t *t, cv::Point2f point) {
	std::vector<renderTriangle>*mesh_triangles = t->mesh_triangles;
	auto  foundTriangle = findTriangle(mesh_triangles, point);
	if(!std::get<0>(foundTriangle)) {
		std::cout << " elastic TRIANGLE NOT FOUND " << std::endl;
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
		//std::cout << c1.x << " (" << lower_x << "," << upper_x << ")" << c1.y << " (" << lower_y << "," << upper_y << ")" << std::endl;

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

cv::Mat render(section_data_t* section, std::string filename, int input_lower_x, int input_upper_x, int input_lower_y, int input_upper_y, Resolution res, bool write) {
	int lower_y, lower_x, upper_y, upper_x;
	int nrows, ncols;
	double scale_x, scale_y;
	//calculate scale
	if(res == THUMBNAIL) {
    	std::string thumbnailpath = std::string(section->tiles[0].filepath);
    	thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
    	thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");
    	cv::Mat thumbnail_img = cv::imread(thumbnailpath,CV_LOAD_IMAGE_UNCHANGED);
    	cv::Mat img = cv::imread(section->tiles[0].filepath,CV_LOAD_IMAGE_UNCHANGED);
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

	section->p_out = new cv::Mat();
	(*section->p_out).create(nrows, ncols, CV_8UC1);
    for (int y = 0; y < nrows; y++) {
      	for (int x = 0; x < ncols; x++) {
        	section->p_out->at<unsigned char>(y,x) = -1;
      	}
    }
		
	for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
	  	tile_data_t tile = section->tiles[i];

		if(!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
			//std::cout << "tile not in bounds" << std::endl;
			continue;
		} else {
			//std::cout << "in bounds" << std::endl;
		}
		if(res == THUMBNAIL) {	
    		std::string path = std::string(tile.filepath);
    		path = path.replace(path.find(".bmp"), 4,".jpg");             		
			path = path.insert(path.find_last_of("/") + 1, "thumbnail_");
			(*tile.p_image) = cv::imread(path,CV_LOAD_IMAGE_GRAYSCALE);
		} 
		if(res == FULL) {
			 (*tile.p_image) = cv::imread(
    	         tile.filepath,
 		         CV_LOAD_IMAGE_UNCHANGED);
		}
		for (int _x = 0; _x < (*tile.p_image).size().width; _x++) {
	    	for (int _y = 0; _y < (*tile.p_image).size().height; _y++) {
				cv::Point2f p = cv::Point2f(_x*scale_x, _y*scale_y);
				cv::Point2f transformed_p = affine_transform(&tile, p);
				transformed_p = elastic_transform(&tile, transformed_p);
				int x = (int)(transformed_p.x/scale_x + 0.5);
				int y = (int)(transformed_p.y/scale_y + 0.5);
				unsigned char val =
	        		tile.p_image->at<unsigned char>(_y, _x);
				//std::cout << "value : " << val << ", ";
				
				if(y-lower_y >= 0 && y-lower_y < nrows && x-lower_x >= 0 && x-lower_x < ncols) {
					section->p_out->at<unsigned char>(y-lower_y, x-lower_x) = val;
				}
	    	}	
	  	}
	  	tile.p_image->release();
	}
	for (int y = 0; y < nrows; y++) {
      	for (int x = 0; x < ncols; x++) {
        	if(section->p_out->at<unsigned char>(y,x) < 0) {
				std::cout << y << " " << x << " unassigned" << std::endl;
			}
      	}
    }
	if(write) {
		cv::imwrite(filename, (*section->p_out));
	}
	return (*section->p_out);
}

float matchTemplate(cv::Mat img1, cv::Mat img2) {
    cv::Mat result_SQDIFF, result_SQDIFF_NORMED, result_CCORR, result_CCORR_NORMED,
        result_CCOEFF, result_CCOEFF_NORMED;
    cv::matchTemplate(img1, img2, result_SQDIFF, CV_TM_SQDIFF);
    cv::matchTemplate(img1, img2, result_SQDIFF_NORMED, CV_TM_SQDIFF_NORMED);
    cv::matchTemplate(img1, img2, result_CCORR, CV_TM_CCORR);
    cv::matchTemplate(img1, img2, result_CCORR_NORMED, CV_TM_CCORR_NORMED);
    cv::matchTemplate(img1, img2, result_CCOEFF, CV_TM_CCOEFF);
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

std::set<std::pair<int, int> > find_bad_triangles_geometric(std::vector<renderTriangle> * triangles, section_data_t* prev_section, section_data_t* section, int lower_x, int upper_x, int lower_y, int upper_y, int box_width, int box_height, Resolution res) {
	std::set<std::pair<int, int> > bad_triangles;
	int count = 0;
	std::map<std::pair<int, int>, float> total_corr;
	std::map<std::pair<int, int>, int>  total_boxes;
	std::cout << "called find bad triangles " << std::endl;
	std::cout << "box dimentions: " << box_width << " " << box_height << std::endl;
	for(int i = lower_y; i < upper_y - box_height; i += box_height) {
		for(int j = lower_x; j < upper_x - box_width; j += box_width) {
			std::string file1 = std::string("1box") + std::to_string(count) + std::string(".tif");
			std::string file2 = std::string("2box") + std::to_string(count) + std::string(".tif");

			cv::Mat im1 = render(section, file1, j, j + box_width, i, i + box_height, res, false);
			count ++;
			cv::Mat im2 = render(prev_section, file2, j, j + box_width, i, i + box_height, res, false);

			float corr = matchTemplate(im1, im2);
	
			cv::Point2f middle(j-box_width/2, i-box_height/2);
			auto tri = findTriangle(triangles, middle); //this is super slow prob
			if(!std::get<0>(tri)) {
				std::cout << "TRIANGLE NOT FOUND " << middle.x << " " << middle.y << std::endl;
				//exit(0);
			}
			std::pair<int, int> key = (*triangles)[0].key;
			if(total_corr.find(key) == total_corr.end()) {
				total_corr[key] = 0;
				total_boxes[key] = 0;
			}
			total_corr[key] += corr;
			total_boxes[key] ++;
		}
	}
	for(int i = 0; i < triangles->size(); i ++) {
		std::pair<int, int> key = (*triangles)[i].key;
		float avg = total_corr[key]/total_boxes[key];
		if(avg > 0.1) {
			bad_triangles.insert(key);
		}
	}
	
	return bad_triangles;
}



std::set<std::pair<int, int> > find_bad_triangles(std::vector<renderTriangle> * triangles, section_data_t* prev_section, section_data_t* section, int lower_x, int upper_x, int lower_y, int upper_y, int box_width, int box_height, Resolution res) {
	std::set<std::pair<int, int> > bad_triangles;
	int count = 0;
	std::map<std::pair<int, int>, int>  num_valid;
	std::map<std::pair<int, int>, int>  num_invalid;
	for(int i = lower_y; i < upper_y - box_height; i += box_height) {
		for(int j = lower_x; j < upper_x - box_width; j += box_width) {
			std::string file1 = std::string("1box") + std::to_string(count) + std::string(".tif");
			std::string file2 = std::string("2box") + std::to_string(count) + std::string(".tif");

			cv::Mat im1 = render(section, file1, j, j + box_width, i, i + box_height, res, false);
			count ++;
			cv::Mat im2 = render(prev_section, file2, j, j + box_width, i, i + box_height, res, false);

			float corr = matchTemplate(im1, im2);
	
			cv::Point2f middle(j-box_width/2, i-box_height/2);
			auto tri = findTriangle(triangles, middle); //this is super slow prob
			if(!std::get<0>(tri)) {
				std::cout << "TRIANGLE NOT FOUND " << middle.x << " " << middle.y << std::endl;
				//exit(0);
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
			std::cout << "triangle valid " << num_valid[key] << " invalid " << num_invalid[key] << " key " << key.first << " " << key.second  << std::endl;
		}
		if(num_valid[key] < num_invalid[key]) {
			bad_triangles.insert(key);
		}
	}
	
	return bad_triangles;
}

cv::Mat render_error(section_data_t* prev_section, section_data_t* section, std::string filename, int input_lower_x, int input_upper_x, int input_lower_y, int input_upper_y, int box_width, int box_height, Resolution res, Averaging avg_method) {

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
    	cv::Mat thumbnail_img = cv::imread(thumbnailpath,CV_LOAD_IMAGE_UNCHANGED);
    	cv::Mat img = cv::imread(section->tiles[0].filepath,CV_LOAD_IMAGE_UNCHANGED);
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
	
	if(avg_method == VOTING) {
		bad_triangles = find_bad_triangles(&triangles, prev_section, section, input_lower_x, input_upper_x, input_lower_y, input_upper_y, box_height, box_width, res);	
	}
	if(avg_method == GEOMETRIC) {
		bad_triangles = find_bad_triangles_geometric(&triangles, prev_section, section, input_lower_x, input_upper_x, input_lower_y, input_upper_y, box_height, box_width, res);
	}
	
	
	section->p_out = new cv::Mat();
	(*section->p_out).create(nrows, ncols, CV_8UC1);
    for (int y = 0; y < nrows; y++) {
      	for (int x = 0; x < ncols; x++) {
        	section->p_out->at<unsigned char>(y,x) = -1;
      	}
    }
	for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
	  	tile_data_t tile = section->tiles[i];

		if(!tile_in_bounds(tile, input_lower_x, input_upper_x, input_lower_y, input_upper_y)) {
			//std::cout << "tile not in bounds" << std::endl;
			continue;
		} else {
			//std::cout << "in bounds" << std::endl;
		}

		if(res == THUMBNAIL) {	
    		std::string path = std::string(tile.filepath);
    		path = path.replace(path.find(".bmp"), 4,".jpg");             		
			path = path.insert(path.find_last_of("/") + 1, "thumbnail_");
			(*tile.p_image) = cv::imread(path,CV_LOAD_IMAGE_GRAYSCALE);
		} 
		if(res == FULL) {
			 (*tile.p_image) = cv::imread(
    	         tile.filepath,
 		         CV_LOAD_IMAGE_UNCHANGED);
		}
		for (int _x = 0; _x < (*tile.p_image).size().width; _x++) {
	    	for (int _y = 0; _y < (*tile.p_image).size().height; _y++) {
				cv::Point2f p = cv::Point2f(_x*scale_x, _y*scale_y);
				cv::Point2f transformed_p = affine_transform(&tile, p);
				transformed_p = elastic_transform(&tile, transformed_p);
				
				renderTriangle tri = (*tile.mesh_triangles)[0]; //should have the triangle its in be correct
                bool bad_triangle = bad_triangles.find(tri.key) != bad_triangles.end(); // TODO

				int x = (int)(transformed_p.x/scale_x + 0.5);
				int y = (int)(transformed_p.y/scale_y + 0.5);
				unsigned char val = tile.p_image->at<unsigned char>(_y, _x);
				if(bad_triangle) {
					val = 255;
				}				
				if(y-lower_y >= 0 && y-lower_y < nrows && x-lower_x >= 0 && x-lower_x < ncols) {
					section->p_out->at<unsigned char>(y-lower_y, x-lower_x) = val;
				}
	    	}	
	  	}
	  	tile.p_image->release();
	}
	for (int y = 0; y < nrows; y++) {
      	for (int x = 0; x < ncols; x++) {
        	if(section->p_out->at<unsigned char>(y,x) < 0) {
				std::cout << y << " " << x << " unassigned" << std::endl;
			}
      	}
    }
	cv::imwrite(filename, (*section->p_out));
	return (*section->p_out);
}



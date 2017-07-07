
//#include "mesh.h"

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
/*
void reorder(std::vector<renderTriangle>*mesh_triangles, int lower_x, int upper_x, int lower_y, int upper_y) {
	int next_index = 0;
	for(int i = 0; i < mesh_triangles->size(); i ++) {
		if ((*mesh_triangles)[i].p[0].x >= lower_x && (*mesh_triangles)[i].p[0].x <= upper_x &&
			(*mesh_triangles)[i].p[1].x >= lower_x && (*mesh_triangles)[i].p[1].x <= upper_x &&
			(*mesh_triangles)[i].p[2].x >= lower_x && (*mesh_triangles)[i].p[2].x <= upper_x &&	
			(*mesh_triangles)[i].p[0].y >= lower_y && (*mesh_triangles)[i].p[0].y <= upper_y &&			
            (*mesh_triangles)[i].p[1].y >= lower_y && (*mesh_triangles)[i].p[1].y <= upper_y &&
            (*mesh_triangles)[i].p[2].y >= lower_y && (*mesh_triangles)[i].p[2].y <= upper_y) {
			int j = i;
			while(j > next_index) {
				renderTriangle * temp = &(*mesh_triangles)[j - 1];
				(*mesh_triangles)[j - 1] = (*mesh_triangles)[j];
				(*mesh_triangles)[j] = *temp;
				j --;
			}
			next_index = i + 1;
		}
	}
}

int triangleIndex(cv::Point2f point, std::vector<renderTriangle>*mesh_triangles, int lower_x, int upper_x, int lower_y, int upper_y) {
	int new_upper_x, new_lower_x, new_lower_y, new_upper_y;
	if(point.x < (upper_x + lower_x)/2) {
		new_upper_x = (upper_x + lower_x)/2;
		new_lower_x = lower_x;
	} else {
		new_upper_x = upper_x;
		new_lower_x = (upper_x + lower_x)/2;
	}
	if(point.y < (upper_y + lower_y)/2) {
		new_upper_y = (upper_y + lower_y)/2;
		new_lower_y = lower_y;
	} else {
		new_upper_y = upper_y;
		new_lower_y = (upper_y + lower_y)/2;
	}
	
	if(upper_x - lower_x <= 100 && upper_y - lower_y <= 100) {	
		reorder(mesh_triangles, new_lower_x, new_upper_x, new_lower_y, new_upper_y);
		return triangleIndex(point, mesh_triangles, new_lower_x, new_upper_x, new_lower_y, new_upper_x);
	} else if(upper_x - lower_x <= 100) {
		reorder(mesh_triangles, new_lower_x, new_upper_x, lower_y, upper_y);
		return triangleIndex(point, mesh_triangles, new_lower_x, new_upper_x, lower_y, upper_y);
	} else if(upper_y - lower_y <= 100) {
		reorder(mesh_triangles, lower_x, upper_x, new_lower_y, new_upper_y);
		return triangleIndex(point, mesh_triangles, lower_x, upper_x, new_lower_y, new_upper_y);
	}

	for(int i = 0; i < mesh_triangles->size(); i ++) {
		if(inTriangle(point, (*mesh_triangles)[i].p)) {
			return i;
			//re-order some more?
		}
	}

	std::cout << "TRIANGLE FINDING FAILED" << std::endl;
	return -1;
}*/

cv::Point2f elastic_transform(tile_data_t *t, cv::Point2f point) {
	std::vector<renderTriangle>*mesh_triangles = t->mesh_triangles;
	int index = -1;
	float u, v, w;
	cv::Point2f a, b, c;
	for(int i = 0; i < mesh_triangles->size(); i ++) {
      a = (*mesh_triangles)[i].p[0];
      b = (*mesh_triangles)[i].p[1];
      c = (*mesh_triangles)[i].p[2];
      Barycentric(point, a,b,c,u,v,w);
      if (u >= 0 && v >= 0 && w >= 0) {
			index = 0;
			int j = i;
			while(j > 0) {
				renderTriangle temp = (*mesh_triangles)[j - 1];
				(*mesh_triangles)[j - 1] = (*mesh_triangles)[j];
				(*mesh_triangles)[j] = temp;
				j --;
			} 
			break;
		}
	}
	if(index < 0) {
		std::cout << " TRIANGLE NOT FOUND " << std::endl;
		return point;
	}

    if (u < 0 || v < 0 || w < 0) {
        printf("Error point wasn't in triangle we found\n");
        printf("Error u %f v %f w %f index %d \n", u,v,w, index);
        exit(1);
    } 

	float new_x = u*(*mesh_triangles)[index].q[0].x + v*(*mesh_triangles)[index].q[1].x+w*(*mesh_triangles)[index].q[2].x;
	float new_y = u*(*mesh_triangles)[index].q[0].y + v*(*mesh_triangles)[index].q[1].y + w*(*mesh_triangles)[index].q[2].y;

	if(std::abs(new_x-point.x) > 0.1 /*|| std::abs(new_y - point.y) > 0.1*/) {
//		std::cout << "off by too much " << new_x << " " << point.x << std::endl;
//		 std::cout << "off by too much " << new_y << " " << point.y << std::endl;
	//	exit(0);
	}
	return cv::Point2f(new_x, new_y);
}

cv::Point2f affine_transform(tile_data_t * t, cv::Point2f point) {
	float new_x = point.x*t->a00 + point.y*t->a01 + t->offset_x;
	float new_y = point.x*t->a10 + point.y*t->a11 + t->offset_y;
	return cv::Point2f(new_x, new_y);
}

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
        	section->p_out->at<unsigned char>(y,x) = -1;
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
	//apply transform for the coners of the tile - check to see if its actually worth rendering (overallaping with the region)  
	//loop through tiles
	for (int i = 0; i < section->n_tiles; i++) {
	  	tile_data_t tile = section->tiles[i];
		cv::Point2f c1 = cv::Point2f(tile.x_start, tile.y_start);
		cv::Point2f c2 = cv::Point2f(tile.x_start, tile.y_finish);
		cv::Point2f c3 = cv::Point2f(tile.x_finish, tile.y_start);
		cv::Point2f c4 = cv::Point2f(tile.x_finish, tile.y_finish);
		c1 = affine_transform(&tile, c1);		
		c2 = affine_transform(&tile, c2);
		c3 = affine_transform(&tile, c3);
		c4 = affine_transform(&tile, c4);
		if((c1.y >= upper_y || c1.y < lower_y || c1.x >= upper_x || c1.x < lower_x) &&
			(c2.y >= upper_y || c2.y < lower_y || c2.x >= upper_x || c2.x < lower_x) &&
			(c3.y >= upper_y || c3.y < lower_y || c3.x >= upper_x || c3.x < lower_x) &&
			(c4.y >= upper_y || c4.y < lower_y || c4.x >= upper_x || c4.x < lower_x)) {	
			//printf("tile v (%f, %f)  (%f, %f)  (%f, %f)  (%f, %f) \n", c1.x, c1.y, c2.x, c2.y, c3.x, c3.y, c4.x, c4.y);
			//continue;
		} else {
			printf("tile v (%f, %f)  (%f, %f)  (%f, %f)  (%f, %f) \n", c1.x, c1.y, c2.x, c2.y, c3.x, c3.y, c4.x, c4    .y);
		}
	  	(*tile.p_image) = cv::imread(
	    tile.filepath,
	    CV_LOAD_IMAGE_UNCHANGED);
		
	
		for (int _x = 0; _x < 3128; _x++) {
	    	for (int _y = 0; _y < 2724; _y++) {
				cv::Point2f p = cv::Point2f(_x, _y);
				cv::Point2f transformed_p = affine_transform(&tile, p);
				{
					int x = transformed_p.x;
					int y = transformed_p.y;
					if(y >= upper_y + 1000.0 || y < lower_y - 1000.0 || x >= upper_x +1000.0 || x < lower_x - 1000.0) {
						continue;
					}
				}
				//std::cout << "calling elastic... " << std::endl;
				transformed_p = elastic_transform(&tile, transformed_p);
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
struct Point
{
    int x;
    int y;
};
 
// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
bool onSegment(Point p, Point q, Point r)
{
    if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
        q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
       return true;
 
    return false;
}
 
// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(Point p, Point q, Point r)
{
    // See http://www.geeksforgeeks.org/orientation-3-ordered-points/
    // for details of below formula.
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);
 
    if (val == 0) return 0;  // colinear
 
    return (val > 0)? 1: 2; // clock or counterclock wise
}
 
// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
bool doIntersect(Point p1, Point q1, Point p2, Point q2)
{
    // Find the four orientations needed for general and
    // special cases
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);
 
    // General case
    if (o1 != o2 && o3 != o4)
        return true;
 
    // Special Cases
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;
 
    // p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;
 
    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;
 
     // p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;
 
    return false; // Doesn't fall in any of the above cases
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

		if(c1.x < 20000 && c1.y < 20000) {
			std::cout << "SHOULD BE IN" << std::endl;
		}
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
		std::vector<cv::Point2i> a;
		std::vector<cv::Point2i> b;
		a.push_back(c1);
		a.push_back(c2);
		a.push_back(c3);
		a.push_back(c4);
		b.push_back(cv::Point2f(lower_x, lower_y));
		b.push_back(cv::Point2f(lower_x, upper_y));
		b.push_back(cv::Point2f(upper_x, lower_y));
		b.push_back(cv::Point2f(upper_x, upper_y));

		for(int a_i = 0; a_i < 4; a_i ++) {
			for(int a_j = 0; a_j < 4; a_j ++) {
				if(a_i == a_j) continue;
				for(int b_i = 0; b_i < 4; b_i ++) {
					for(int b_j = 0; b_j < 4; b_j ++) {
						if(b_j == b_i) continue;

						struct Point p1 = {a[a_i].x, a[a_i].y}, q1 = {a[a_j].x, a[a_j].y};
						struct Point p2 = {b[b_i].x, a[b_i].y}, q2 = {b[b_j].x, b[b_j].y}; 
						if(doIntersect(p1, q1, p2, q2)) return true;
						float x, y;	
						if(a[a_i].x-a[a_j].x == 0 && b[b_i].x-b[b_j].x == 0) {
							continue; //(I think) 
						} else if(a[a_i].x-a[a_j].x == 0) {
							x = a[a_i].x;
							float slope_b = (b[b_i].y-b[b_j].y)/(b[b_i].x-b[b_j].x);
							float intercept_b = b[b_i].y - slope_b*b[b_i].x;
							y = slope_b*x + intercept_b;
						} else if(b[b_i].x-b[b_j].x == 0) {
							x = b[b_i].x;
							float slope_a = (float)(a[a_i].y-a[a_j].y)/(a[a_i].x-a[a_j].x);
							float intercept_a = a[a_i].y - slope_a*a[a_i].x;
							y = slope_a*x + intercept_a;
						} else {
							float slope_a = (float)(a[a_i].y-a[a_j].y)/(a[a_i].x-a[a_j].x);
							float intercept_a = a[a_i].y - slope_a*a[a_i].x;
							float slope_b = (float)(b[b_i].y-b[b_j].y)/(b[b_i].x-b[b_j].x);
							float intercept_b = b[b_i].y - slope_b*b[b_i].x;
							if(slope_a == slope_b) continue;
							x = (intercept_b - intercept_a)/(slope_a - slope_b);
							y = slope_a * x + intercept_a;
							if(y != slope_b * x + intercept_b) {
	//							std::cout << "ERROR IN SLOPE" << slope_a << " " << intercept_a << " " << slope_b << " " << intercept_b << " " << x << " " << y << std::endl;
								//exit(0);
							}
						}
	
						if((x >= min( a[a_j].x, a[a_i].x) && x <= max(a[a_i].x, a[a_j].x)) &&
							(y >= min(a[a_j].y, a[a_i].y) && y <= max(a[a_i].y, a[a_j].y)) &&
							(x >= min(b[b_j].x, b[b_i].x) && x <= max(b[b_i].x, b[b_j].x)) &&
							(y >= min(b[b_j].y, b[b_i].y) && y <= max(b[b_i].y, b[b_j].y))) {
							return true;
						} else {
	//						std::cout << "not on segment (" << a[a_i].x << "," << a[a_i].y << ") (" << a[a_j].x << "," << a[a_j].y << ")          (" <<  b[b_i].x << "," << b[b_i].y << ") (" << b[b_j].x << "    ," << b[b_j].y << ")" << std::endl;
						}
					}
				}
			}
		}
		return false;
}

cv::Mat output_section_image_affine_elastic_thumbnail(section_data_t* section, std::string filename, int lower_x, int upper_x, int lower_y, int upper_y) {
	//calculate scale
    std::string thumbnailpath = std::string(section->tiles[0].filepath);
    thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
    thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");
    cv::Mat thumbnail_img = cv::imread(thumbnailpath,CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat img = cv::imread(section->tiles[0].filepath,CV_LOAD_IMAGE_UNCHANGED);
	double scale_x = (double)(img.size().width)/thumbnail_img.size().width;
    double scale_y = (double)(img.size().height)/thumbnail_img.size().height;

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
        	section->p_out->at<unsigned char>(y,x) = -1;
      	}
    }
	for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
	  	tile_data_t tile = section->tiles[i];
		if(!tile_in_bounds(tile, lower_x, upper_x, lower_y, upper_y)) {
			//std::cout << "tile not in bounds" << std::endl;
			continue;
		} else {
			//std::cout << "in bounds" << std::endl;
		}
	
    	std::string path = std::string(tile.filepath);
    	path = path.replace(path.find(".bmp"), 4,".jpg");             		
		path = path.insert(path.find_last_of("/") + 1, "thumbnail_");
		(*tile.p_image) = cv::imread(path,CV_LOAD_IMAGE_GRAYSCALE);
		int max_x = 0;		
		for (int _x = 0; _x < width*scale_x; _x++) {
	    	for (int _y = 0; _y < height*scale_y; _y++) {
				cv::Point2f p = cv::Point2f(_x, _y);
				cv::Point2f transformed_p = affine_transform(&tile, p);
				{
					int x = transformed_p.x;
					int y = transformed_p.y;
					if(y >= upper_y + 1000.0 || y < lower_y - 1000.0 || x >= upper_x +1000.0 || x < lower_x - 1000.0) {
						continue;
					}
				}
				//std::cout << "calling elastic... " << std::endl;
				transformed_p = elastic_transform(&tile, transformed_p);
				int x = transformed_p.x;
				int y = transformed_p.y;
				if(y >= upper_y || y < lower_y || x >= upper_x || x < lower_x) {
					continue;
				}
				//printf("tile c (%f, %f)  (%f, %f)  (%f, %f)  (%f, %f) \n", c1.x, c1.y, c2.x, c2.y, c3.x, c3.y, c4.x, c4.y);
					
				//std::cout << "tile t (" << tile.x_start << "," << tile.y_start << ") (" << tile.x_start << "," <<  tile.y_finish << ") (" <<  tile.x_finish << "," << tile.y_start << ") (" << tile.x_finish << "," << tile.y_finish << ")" << std::endl;
				unsigned char val =
	        		tile.p_image->at<unsigned char>((int)(_y/scale_y), (int)(_x/scale_x));
				std::cout << "value : " << val << ", ";
				section->p_out->at<unsigned char>(y-lower_y, x-lower_x) = val;
	    	}	
	  	}
	  	tile.p_image->release();
	}
	cv::imwrite(filename, (*section->p_out));
	return (*section->p_out);
}
cv::Mat output_section_image_affine_elastic_thumbnail_to_thumbnail(section_data_t* section, std::string filename, int lower_x, int upper_x, int lower_y, int upper_y) {
	//calculate scale
    std::string thumbnailpath = std::string(section->tiles[0].filepath);
    thumbnailpath = thumbnailpath.replace(thumbnailpath.find(".bmp"), 4,".jpg");
    thumbnailpath = thumbnailpath.insert(thumbnailpath.find_last_of("/") + 1, "thumbnail_");
    cv::Mat thumbnail_img = cv::imread(thumbnailpath,CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat img = cv::imread(section->tiles[0].filepath,CV_LOAD_IMAGE_UNCHANGED);
	double scale_x = (double)(img.size().width)/thumbnail_img.size().width;
    double scale_y = (double)(img.size().height)/thumbnail_img.size().height;

    std::cout << "SCALE OF THUMBNAIL: " << scale_x << " " << scale_y  << std::endl;
   	int width = thumbnail_img.size().width;
	int height = thumbnail_img.size().height; 
	std::cout << "thumbnail width: " << width << " height: " << height << " | original width: " << img.size().width << " height: " << img.size().height << std::endl;

	//create new matrix
	int nrows = (upper_y-lower_y)/scale_y;
    int ncols = (upper_x-lower_x)/scale_x; 
	section->p_out = new cv::Mat();
	(*section->p_out).create(nrows, ncols, CV_8UC1);
    for (int y = 0; y < nrows; y++) {
      	for (int x = 0; x < ncols; x++) {
        	section->p_out->at<unsigned char>(y,x) = -1;
      	}
    }
	for (int i = section->n_tiles; --i>=0;/*i < section->n_tiles; i++*/) {
	  	tile_data_t tile = section->tiles[i];
		if(!tile_in_bounds(tile, lower_x, upper_x, lower_y, upper_y)) {
			std::cout << "tile not in bounds" << std::endl;
			continue;
		} else {
			std::cout << "in bounds" << std::endl;
		}
	
    	std::string path = std::string(tile.filepath);
    	path = path.replace(path.find(".bmp"), 4,".jpg");             		
		path = path.insert(path.find_last_of("/") + 1, "thumbnail_");
		(*tile.p_image) = cv::imread(path,CV_LOAD_IMAGE_GRAYSCALE);
		int max_x = 0;		
		for (int _x = 0; _x < width; _x++) {
	    	for (int _y = 0; _y < height; _y++) {
				cv::Point2f p = cv::Point2f(_x*scale_x, _y*scale_y);
				cv::Point2f transformed_p = affine_transform(&tile, p);
				{
					int x = transformed_p.x;
					int y = transformed_p.y;
					if(y >= upper_y + 1000.0 || y < lower_y - 1000.0 || x >= upper_x +1000.0 || x < lower_x - 1000.0) {
						continue;
					}
				}
				//std::cout << "calling elastic... " << std::endl;
				transformed_p = elastic_transform(&tile, transformed_p);
				int x = transformed_p.x/scale_x;
				int y = transformed_p.y/scale_y;
				if(y >= upper_y || y < lower_y || x >= upper_x || x < lower_x) {
					continue;
				}
				//printf("tile c (%f, %f)  (%f, %f)  (%f, %f)  (%f, %f) \n", c1.x, c1.y, c2.x, c2.y, c3.x, c3.y, c4.x, c4.y);
					
				//std::cout << "tile t (" << tile.x_start << "," << tile.y_start << ") (" << tile.x_start << "," <<  tile.y_finish << ") (" <<  tile.x_finish << "," << tile.y_start << ") (" << tile.x_finish << "," << tile.y_finish << ")" << std::endl;
				unsigned char val =
	        		tile.p_image->at<unsigned char>(_y, _x);
				int v = val;
				std::cout << v << ",";
				section->p_out->at<unsigned char>(y-lower_y/scale_y, x-lower_x/scale_x) = val;
	    	}	
	  	}
	  	tile.p_image->release();
	}
	cv::imwrite(filename, (*section->p_out));
	return (*section->p_out);
}

void matchTemplate(cv::Mat img1, cv::Mat img2) {
    cv::Mat result_SQDIFF, result_SQDIFF_NORMED, result_CCORR, result_CCORR_NORMED,
        result_CCOEFF, result_CCOEFF_NORMED;
    
    cv::matchTemplate(img1, img2, result_SQDIFF, CV_TM_SQDIFF);
    cv::matchTemplate(img1, img2, result_SQDIFF_NORMED, CV_TM_SQDIFF_NORMED);
    cv::matchTemplate(img1, img2, result_CCORR, CV_TM_CCORR);
    cv::matchTemplate(img1, img2, result_CCORR_NORMED, CV_TM_CCORR_NORMED);
    cv::matchTemplate(img1, img2, result_CCOEFF, CV_TM_CCOEFF);
    cv::matchTemplate(img1, img2, result_CCOEFF_NORMED, CV_TM_CCOEFF_NORMED);
   
    std::cout << "---------- MATCH TEMPLATE RESULTS (image size: " << img1.size().width << " " <<
        img1.size().height << " " << img2.size().width << " " << img2.size().height << ") -----------" << std::endl;
    std::cout << "       SQDIFF: " <<        result_SQDIFF.at<float>(0,0) << std::endl;
    std::cout << "SQDIFF_NORMED: " << result_SQDIFF_NORMED.at<float>(0,0) << std::endl;
    std::cout << "        CCORR: " <<         result_CCORR.at<float>(0,0) << std::endl;
    std::cout << " CCORR_NORMED: " <<  result_CCORR_NORMED.at<float>(0,0) << std::endl;
    std::cout << "       CCOEFF: " <<        result_CCOEFF.at<float>(0,0) << std::endl;
    std::cout << "CCOEFF_NORMED: " << result_CCOEFF_NORMED.at<float>(0,0) << std::endl;

    
}

/*void cross_correlation(std::string filepath1, std::string filepath2, int box_width, int box_height) { */

void cross_correlation(cv::Mat img1, cv::Mat img2, int box_width, int box_height) {
	//cv::Mat img1 = cv::imread(filepath1,CV_LOAD_IMAGE_UNCHANGED);
    //cv::Mat img2 = cv::imread(filepath2,CV_LOAD_IMAGE_UNCHANGED);
	
	int upper_row1 = -1;
	int lower_row1 = -1;
	int upper_col1 = -1;
	int lower_col1 = -1;
    int upper_row2= -1;
    int lower_row2= -1;
    int upper_col2 = -1;
    int lower_col2 = -1;
	
    /* lower bound row for image 1*/
	for(int i = 0; i < img1.size().height; i ++) {
		bool blank = false;
		for(int j = 0; j < img1.size().width; j ++) {
			if(img1.at<float>(i, j) < 0) blank = true;
		}
        if(!blank) {
            lower_row1 = i;
            break;
        }
	}
    /* upper bound row for image 1*/
    for(int i = img1.size().height; i >= 0; i --) {
        bool blank = false;
        for(int j = 0; j < img1.size().width; j ++) {
            if(img1.at<float>(i, j) < 0) blank = true;
        }
        if(!blank) {
            upper_row1 = i;
            break;
        }
    }
    /* lower bound col for image 1*/
    for(int i = 0; i < img1.size().width; i ++) {
        bool blank = false;
        for(int j = 0; j < img1.size().height; j ++) {
            if(img1.at<float>(j, i) < 0) blank = true;
        }
        if(!blank) {
            lower_col1 = i;
            break;
        }
    }
    /* upper bound col for image 1*/
    for(int i = img1.size().width; i >= 0; i --) {
        bool blank = false;
        for(int j = 0; j < img1.size().height; j ++) {
            if(img1.at<float>(j, i) < 0) blank = true;
        }
        if(!blank) {
            upper_col1 = i;
            break;
        }
    }
    /* lower bound row for image 2*/
    for(int i = 0; i < img2.size().height; i ++) {
        bool blank = false;
        for(int j = 0; j < img2.size().width; j ++) {
            if(img2.at<float>(i, j) < 0) blank = true;
        }
        if(!blank) {
            lower_row2 = i;
            break;
        }
    }
    /* upper bound row for image 2*/
    for(int i = img2.size().height; i >= 0; i --) {
        bool blank = false;
        for(int j = 0; j < img2.size().width; j ++) {
            if(img2.at<float>(i, j) < 0) blank = true;
        }
        if(!blank) {
            upper_row2 = i;
            break;
        }
    }
    /* lower bound col for image 2*/
    for(int i = 0; i < img2.size().width; i ++) {
        bool blank = false;
        for(int j = 0; j < img2.size().height; j ++) {
            if(img2.at<float>(j, i) < 0) blank = true;
        }
        if(!blank) {
            lower_col2 = i;
            break;
        }
    }
    /* upper bound col for image 2*/
    for(int i = img2.size().width; i >= 0; i --) {
        bool blank = false;
        for(int j = 0; j < img2.size().height; j ++) {
            if(img2.at<float>(j, i) < 0) blank = true;
        }
        if(!blank) {
            upper_col2 = i;
            break;
        }
    }
    if(upper_row1 < 0 || lower_row1 < 0 || upper_col1 < 0 || lower_col1 < 0 ||
       upper_row2 < 0 || lower_row2 < 0 || upper_col2 < 0 || lower_col2 < 0) {
        std::cout << "Invalid bounds " << upper_row1 << " " << lower_row1 << "  "
		<< upper_col1 << " " << lower_col1 << " " << upper_row2 << " " << lower_row2 << " " << upper_col2 << " " << lower_col2 << std::endl;
    }
	std::cout << "bounds " << upper_row1 << " " << lower_row1 << "  "
		<< upper_col1 << " " << lower_col1 << " " << upper_row2 << " " << lower_row2 << " " << upper_col2 << " " << lower_col2 << std::endl;
 
	int upper_row = min(upper_row1, upper_row2);
	int lower_row = max(lower_row1, lower_row2);
	int upper_col = min(upper_col1, upper_col2);
	int lower_col = max(lower_col1, lower_col2);

	cv::Mat box1, box2;	
	box1.create(box_height, box_width, CV_8UC1);
	box2.create(box_height, box_width, CV_8UC1);
	for(int i = lower_row; i <= upper_row; i ++) {
		for(int j = lower_col; j <= upper_col; j ++) {
            int row = (i - lower_row)%box_height;
            int col = (j - lower_col)%box_width;
            if(i != lower_row && j != lower_col && row == 0 && col == 0) {
                /* boxes are filled in and ready to be compared */
                matchTemplate(box1, box2);
            }
			box1.at<float>(j%box_height, i%box_width) = img1.at<float>(j, i);
            box2.at<float>(j%box_height, i%box_width) = img2.at<float>(j, i);
        }
    }
	std::cout << "method finished" << std::endl;	
	cv::Mat result;
	//cv::matchTemplate	
}
//fucntion for same matrix size (to test correlation code)
//take a region in global space and a pair of section (for every pair) - require all callers to define coordinaes in global space for overlap
//check correlation code first (instead of rectangle code)
//quering patches (tree)
void cross_correlate_simple(cv::Mat img1, cv::Mat img2, int box_height, int box_width) {
}	

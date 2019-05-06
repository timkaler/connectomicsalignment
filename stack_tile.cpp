
#include "decode_jp2.cpp"

int DEBUG_total_read_count = 0;

void updateTile2DAlign(int vid, void* scheduler_void) {
  //double global_learning_rate = 0.49;

  Scheduler* scheduler = reinterpret_cast<Scheduler*>(scheduler_void);
  Graph* graph = reinterpret_cast<Graph*>(scheduler->graph_void);

  vdata* vertex_data = graph->getVertexData(vid);
  tfk::Tile* tile = (tfk::Tile*) vertex_data->tile;

  if (vid != tile->tile_id) printf("Failure!\n");

  tile->local2DAlignUpdate();

  if (vertex_data->iteration_count < 10000) {
    scheduler->add_task(vid, updateTile2DAlign);
  }
  vertex_data->iteration_count++;
}


float tfk::Tile::compute_deviation(Tile* b_tile) {

        cv::Point2f a_point =
            this->rigid_transform(this->ideal_points[b_tile->tile_id].first);
        cv::Point2f b_point =
            b_tile->rigid_transform(this->ideal_points[b_tile->tile_id].second);
        cv::Point2f delta =  a_point - b_point;
        cv::Point2f idelta = this->ideal_offsets[b_tile->tile_id];

	//cv::Point2f a_point = cv::Point2f(this->x_start + this->offset_x,
	//		this->y_start + this->offset_y);
	//cv::Point2f b_point = cv::Point2f(b_tile->x_start+b_tile->offset_x,
	//		b_tile->y_start+b_tile->offset_y);

	//cv::Point2f delta = a_point-b_point;
	//cv::Point2f idelta = this->ideal_offsets[b_tile->tile_id];
	double dx = delta.x-idelta.x;
	double dy = delta.y-idelta.y;
	//return std::max(dx,dy);
	return std::sqrt(dx*dx+dy*dy);
}


std::vector<float> tfk::Tile::tile_pair_feature(Tile *other) {
	//return 1.0;
	if (!(this->overlaps_with(other))) {
		return std::vector<float>();//-2;
   	}


        
        //cv::Mat tile_p_image_1 = this->get_tile_data(Resolution::FULL);
        //cv::Mat tile_p_image_2 = this->get_tile_data(Resolution::FULL);

        //std::pair<cv::Point2f, cv::Point2f> offset_info;
        //std::pair<cv::Mat, cv::Mat> overlap_matrix = this->get_overlap_matrix(other, 1.0, offset_info);

        float extra_feature = -1.0;
        float extra_feature2 = -1.0;
        //if (overlap_matrix.first.rows > 10 && overlap_matrix.first.cols > 10) {
        //  cv::Mat small_box1 = overlap_matrix.first(cv::Rect(0,0,10,10));
        //  cv::Mat small_box2 = overlap_matrix.second(cv::Rect(0,0,10,10));
        //  cv::Mat result_TM_SQDIFF;
        //  cv::Mat result2_TM_SQDIFF;
        //  cv::matchTemplate(small_box1, small_box2, result_TM_SQDIFF, CV_TM_CCORR);
        //  cv::matchTemplate(small_box1, small_box2, result2_TM_SQDIFF, CV_TM_CCORR_NORMED);
        //  extra_feature = result_TM_SQDIFF.at<float>(0,0);
        //  extra_feature2 = result2_TM_SQDIFF.at<float>(0,0);
        //  //printf("computed extra feature %f %f\n", extra_feature, extra_feature2);
        //} 
	//cv::Mat tile_p_image_1;
	//cv::Mat tile_p_image_2;

	//float scale = 0.3;
	//float scale_x = scale;
	//float scale_y = scale;
	//tile_p_image_1 = this->get_tile_data(Resolution::PERCENT30);
	//tile_p_image_2 = other->get_tile_data(Resolution::PERCENT30);
	////tile_p_image_1 = this->get_tile_data(Resolution::FULL);
	////tile_p_image_2 = other->get_tile_data(Resolution::FULL);

	//std::pair<cv::Point2f, cv::Point2f> tile_1_bounds = this->get_bbox();
	//std::pair<cv::Point2f, cv::Point2f> tile_2_bounds = other->get_bbox();

	//// scale the bbox.
	//tile_1_bounds.first.x *= scale_x;
	//tile_1_bounds.first.y *= scale_y;
	//tile_1_bounds.second.x *= scale_x;
	//tile_1_bounds.second.y *= scale_y;
	//tile_2_bounds.first.x *= scale_x;
	//tile_2_bounds.first.y *= scale_y;
	//tile_2_bounds.second.x *= scale_x;
	//tile_2_bounds.second.y *= scale_y;

	//int nrows = std::min(tile_1_bounds.second.y, tile_2_bounds.second.y) - std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);
	//int ncols = std::min(tile_1_bounds.second.x, tile_2_bounds.second.x) - std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
	////printf("rows = %d, cols = %d\n", nrows, ncols);
	//if ((nrows <= 0) || (ncols <= 0) ) {
	//  return std::vector<float>();
	//}
	//int offset_x = std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
	//int offset_y = std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);




	//cv::Mat transform_1 = cv::Mat::zeros(nrows, ncols, CV_8UC1);
	//cv::Mat transform_2 = cv::Mat::zeros(nrows, ncols, CV_8UC1);

	//// make the transformed images in the same size with the same cells in the same locations

	//// determine start coordinates

	//int start_y1 = 1000000;
	//int end_y1 = -1;
	//int start_x1 = 1000000;
	//int end_x1 = -1;
	//for (int _y = 0; _y < tile_p_image_1.rows; _y++) {
	//    int _x = 0;
	//    cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
	//    cv::Point2f transformed_p = this->rigid_transform(p)*scale;

	//    int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
	//    int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
	//    if ((y_c >= 0) && (y_c < nrows)) {
	//      if (_y < start_y1) {
	//        start_y1 = _y;
	//      }
	//      if (_y > end_y1) {
	//        end_y1 = _y;
	//      }
	//    }
	//}
	//for (int _x = 0; _x < tile_p_image_1.cols; _x++) {
	//    int _y = 0;
	//    cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
	//    cv::Point2f transformed_p = this->rigid_transform(p)*scale;

	//    int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
	//    int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
	//    if ((x_c >= 0) && (x_c < ncols)) {
	//      if (_x < start_x1) {
	//        start_x1 = _x;
	//      }
	//      if (_x > end_x1) {
	//        end_x1 = _x;
	//      }
	//    }
	//}

	//int start_y2 = 1000000;
	//int end_y2 = -1;
	//int start_x2 = 1000000;
	//int end_x2 = -1;
	//for (int _y = 0; _y < tile_p_image_2.rows; _y++) {
	//    int _x = 0;
	//    cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
	//    cv::Point2f transformed_p = other->rigid_transform(p)*scale;

	//    int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
	//    int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
	//    if ((y_c >= 0) && (y_c < nrows)) {
	//      if (_y < start_y2) {
	//        start_y2 = _y;
	//      }
	//      if (_y > end_y2) {
	//        end_y2 = _y;
	//      }
	//    }
	//}
	//for (int _x = 0; _x < tile_p_image_2.cols; _x++) {
	//    int _y = 0;
	//    cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
	//    cv::Point2f transformed_p = other->rigid_transform(p)*scale;

	//    int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
	//    int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
	//    if ((x_c >= 0) && (x_c < ncols)) {
	//      if (_x < start_x2) {
	//        start_x2 = _x;
	//      }
	//      if (_x > end_x2) {
	//        end_x2 = _x;
	//      }
	//    }
	//}


	//for (int _y = start_y1; _y < end_y1; _y++) {
	//  for (int _x = start_x1; _x < end_x1; _x++) {
	//    cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
	//    cv::Point2f transformed_p = this->rigid_transform(p)*scale;

	//    int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
	//    int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
	//    if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
	//      transform_1.at<unsigned char>(y_c, x_c) +=
	//         tile_p_image_1.at<unsigned char>(_y, _x);
	//    }
	//  }
	//}

	//for (int _y = start_y2; _y < end_y2; _y++) {
	//  for (int _x = start_x2; _x < end_x2; _x++) {
	//    cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
	//    cv::Point2f transformed_p = other->rigid_transform(p)*scale;

	//    int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
	//    int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
	//    if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
	//      transform_2.at<unsigned char>(y_c, x_c) +=
	//         tile_p_image_2.at<unsigned char>(_y, _x);
	//    }
	//  }
	//}



	//// clear any location which only has a value for one of them
	//// note that the transforms are the same size
	//int min_x = 1000000;
	//int min_y = 1000000;
	//int max_x = -1;
	//int max_y = -1;
	//for (int _y = 0; _y < transform_1.rows; _y++) {
	//  for (int _x = 0; _x < transform_1.cols; _x++) {
	//    if (transform_2.at<unsigned char>(_y, _x) == 0) {
	//     transform_1.at<unsigned char>(_y, _x) = 0;
	//    }
	//    else if (transform_1.at<unsigned char>(_y, _x) == 0) {
	//     transform_2.at<unsigned char>(_y, _x) = 0;
	//    } else {
	//      if (_y < min_y) min_y = _y;
	//      if (_y > max_y) max_y = _y;
	//      if (_x < min_x) min_x = _x;
	//      if (_x > max_x) max_x = _x;
	//    }
	//  }
	//}

	//int small_nrows = max_y - min_y + 1;
	//int small_ncols = max_x - min_x + 1;
	//if (small_nrows <= 0 || small_ncols <= 0) return std::vector<float>();
	//cv::Mat small_transform_1 = cv::Mat::zeros(small_nrows, small_ncols, CV_8UC1);
	//cv::Mat small_transform_2 = cv::Mat::zeros(small_nrows, small_ncols, CV_8UC1);

	//for (int _y = 0; _y < transform_1.rows; _y++) {
	//  for (int _x = 0; _x < transform_1.cols; _x++) {
	//    if (transform_2.at<unsigned char>(_y, _x) == 0) {
	//     transform_1.at<unsigned char>(_y, _x) = 0;
	//    }
	//    else if (transform_1.at<unsigned char>(_y, _x) == 0) {
	//     transform_2.at<unsigned char>(_y, _x) = 0;
	//    } else {
	//      small_transform_1.at<unsigned char>(_y-min_y, _x-min_x) =
	//          transform_1.at<unsigned char>(_y, _x);
	//      small_transform_2.at<unsigned char>(_y-min_y, _x-min_x) =
	//          transform_2.at<unsigned char>(_y, _x);
	//    }
	//  }
	//}

	////if (small_transform_1.rows < 10 || small_transform_2.rows < 10 ||
	////    small_transform_1.cols < 10 || small_transform_2.cols < 10) {
	////  return std::vector<float>();
	////}

	//int start_index = 0;
	//std::vector<float> coordinates;
	////for (int r = 0; r < 10; r++) {
	////  for (int c = 0; c < 10; c++) {
	////    coordinates.push_back(1.0*small_transform_1.at<unsigned char>(r,c));
	////    coordinates.push_back(1.0*small_transform_2.at<unsigned char>(r,c));
	////  }
	////}

	//cv::Mat result_CCOEFF_NORMED;
	//cv::Mat result_TM_SQDIFF;

	////cv::Mat _transform_1, _transform_2;
	////cv::resize(transform_1, _transform_1, cv::Size(), 1.0,1.0, CV_INTER_AREA);
	////cv::resize(transform_2, _transform_2, cv::Size(), 1.0,1.0, CV_INTER_AREA);

	////cv::matchTemplate(_transform_1, _transform_2, result_CCOEFF_NORMED, CV_TM_CCOEFF_NORMED);


	std::pair<cv::Point2f, cv::Point2f> tile_1_bounds = this->get_bbox();
	std::pair<cv::Point2f, cv::Point2f> tile_2_bounds = other->get_bbox();

	float end_x = std::min(tile_1_bounds.second.x, tile_2_bounds.second.x);
	float end_y = std::min(tile_1_bounds.second.y, tile_2_bounds.second.y);
	float start_x = std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
	float start_y = std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);

	float dx = end_x - start_x;
	float dy = end_y - start_y;

	//cv::matchTemplate(small_transform_1, small_transform_2, result_CCOEFF_NORMED, CV_TM_CCOEFF_NORMED);
	//cv::matchTemplate(small_transform_1, small_transform_2, result_TM_SQDIFF, CV_TM_CCORR_NORMED);

	//coordinates.push_back(result_CCOEFF_NORMED.at<float>(0,0));
	//coordinates.push_back(result_TM_SQDIFF.at<float>(0,0));
	std::vector<float> coordinates;
	coordinates.push_back(extra_feature);
	coordinates.push_back(extra_feature2);
	coordinates.push_back(dx*dy);
	coordinates.push_back(dx);
	coordinates.push_back(dy);

	//coordinates.push_back(small_transform_1.rows*small_transform_1.cols/(scale*scale));
	//coordinates.push_back(small_transform_1.rows/scale);
	//coordinates.push_back(small_transform_1.cols/scale);

	return coordinates;
	//cv::matchTemplate(small_transform_1, small_transform_2, result_CCOEFF_NORMED, CV_TM_CCOEFF_NORMED);
	//return result_CCOEFF_NORMED.at<float>(0,0);
}

float tfk::Tile::error_tile_pair(Tile *other) {
	//return 1.0;
	if (!(this->overlaps_with(other))) {
		return -2;
	}


	cv::Mat tile_p_image_1;
	cv::Mat tile_p_image_2;

	float scale = 0.3;
	float scale_x = scale;
	float scale_y = scale;
	tile_p_image_1 = this->get_tile_data(Resolution::PERCENT30);
	tile_p_image_2 = other->get_tile_data(Resolution::PERCENT30);

	std::pair<cv::Point2f, cv::Point2f> tile_1_bounds = this->get_bbox();
	std::pair<cv::Point2f, cv::Point2f> tile_2_bounds = other->get_bbox();

	// scale the bbox.
	tile_1_bounds.first.x *= scale_x;
	tile_1_bounds.first.y *= scale_y;
	tile_1_bounds.second.x *= scale_x;
	tile_1_bounds.second.y *= scale_y;
	tile_2_bounds.first.x *= scale_x;
	tile_2_bounds.first.y *= scale_y;
	tile_2_bounds.second.x *= scale_x;
	tile_2_bounds.second.y *= scale_y;

	int nrows = std::min(tile_1_bounds.second.y, tile_2_bounds.second.y) - std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);
	int ncols = std::min(tile_1_bounds.second.x, tile_2_bounds.second.x) - std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
	//printf("rows = %d, cols = %d\n", nrows, ncols);
	if ((nrows <= 0) || (ncols <= 0) ) {
		return -2;
	}
	int offset_x = std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
	int offset_y = std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);




	cv::Mat transform_1 = cv::Mat::zeros(nrows, ncols, CV_8UC1);
	cv::Mat transform_2 = cv::Mat::zeros(nrows, ncols, CV_8UC1);

	// make the transformed images in the same size with the same cells in the same locations

	// determine start coordinates

	int start_y1 = 1000000;
	int end_y1 = -1;
	int start_x1 = 1000000;
	int end_x1 = -1;
	for (int _y = 0; _y < tile_p_image_1.rows; _y++) {
		int _x = 0;
		cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
		cv::Point2f transformed_p = this->rigid_transform(p)*scale;

		//int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
		int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
		if ((y_c >= 0) && (y_c < nrows)) {
			if (_y < start_y1) {
				start_y1 = _y;
			}
			if (_y > end_y1) {
				end_y1 = _y;
			}
		}
	}
	for (int _x = 0; _x < tile_p_image_1.cols; _x++) {
		int _y = 0;
		cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
		cv::Point2f transformed_p = this->rigid_transform(p)*scale;

		int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
		//int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
		if ((x_c >= 0) && (x_c < ncols)) {
			if (_x < start_x1) {
				start_x1 = _x;
			}
			if (_x > end_x1) {
				end_x1 = _x;
			}
		}
	}

	int start_y2 = 1000000;
	int end_y2 = -1;
	int start_x2 = 1000000;
	int end_x2 = -1;
	for (int _y = 0; _y < tile_p_image_2.rows; _y++) {
		int _x = 0;
		cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
		cv::Point2f transformed_p = other->rigid_transform(p)*scale;

		//int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
		int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
		if ((y_c >= 0) && (y_c < nrows)) {
			if (_y < start_y2) {
				start_y2 = _y;
			}
			if (_y > end_y2) {
				end_y2 = _y;
			}
		}
	}
	for (int _x = 0; _x < tile_p_image_2.cols; _x++) {
		int _y = 0;
		cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
		cv::Point2f transformed_p = other->rigid_transform(p)*scale;

		int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
		//int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
		if ((x_c >= 0) && (x_c < ncols)) {
			if (_x < start_x2) {
				start_x2 = _x;
			}
			if (_x > end_x2) {
				end_x2 = _x;
			}
		}
	}


	for (int _y = start_y1; _y < end_y1; _y++) {
		for (int _x = start_x1; _x < end_x1; _x++) {
			cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
			cv::Point2f transformed_p = this->rigid_transform(p)*scale;

			int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
			int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
			if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
				transform_1.at<unsigned char>(y_c, x_c) +=
					tile_p_image_1.at<unsigned char>(_y, _x);
			}
		}
	}

	for (int _y = start_y2; _y < end_y2; _y++) {
		for (int _x = start_x2; _x < end_x2; _x++) {
			cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
			cv::Point2f transformed_p = other->rigid_transform(p)*scale;

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
	int min_x = 1000000;
	int min_y = 1000000;
	int max_x = -1;
	int max_y = -1;
	for (int _y = 0; _y < transform_1.rows; _y++) {
		for (int _x = 0; _x < transform_1.cols; _x++) {
			if (transform_2.at<unsigned char>(_y, _x) == 0) {
				transform_1.at<unsigned char>(_y, _x) = 0;
			}
			else if (transform_1.at<unsigned char>(_y, _x) == 0) {
				transform_2.at<unsigned char>(_y, _x) = 0;
			} else {
				if (_y < min_y) min_y = _y;
				if (_y > max_y) max_y = _y;
				if (_x < min_x) min_x = _x;
				if (_x > max_x) max_x = _x;
			}
		}
	}

	int small_nrows = max_y - min_y + 1;
	int small_ncols = max_x - min_x + 1;
	if (small_nrows <= 0 || small_ncols <= 0) return -2;
	cv::Mat small_transform_1 = cv::Mat::zeros(small_nrows, small_ncols, CV_8UC1);
	cv::Mat small_transform_2 = cv::Mat::zeros(small_nrows, small_ncols, CV_8UC1);

	for (int _y = 0; _y < transform_1.rows; _y++) {
		for (int _x = 0; _x < transform_1.cols; _x++) {
			if (transform_2.at<unsigned char>(_y, _x) == 0) {
				transform_1.at<unsigned char>(_y, _x) = 0;
			}
			else if (transform_1.at<unsigned char>(_y, _x) == 0) {
				transform_2.at<unsigned char>(_y, _x) = 0;
			} else {
				small_transform_1.at<unsigned char>(_y-min_y, _x-min_x) =
					transform_1.at<unsigned char>(_y, _x);
				small_transform_2.at<unsigned char>(_y-min_y, _x-min_x) =
					transform_2.at<unsigned char>(_y, _x);
			}
		}
	}

	cv::Mat result_CCOEFF_NORMED;

	//cv::Mat _transform_1, _transform_2;
	//cv::resize(transform_1, _transform_1, cv::Size(), 1.0,1.0, CV_INTER_AREA);
	//cv::resize(transform_2, _transform_2, cv::Size(), 1.0,1.0, CV_INTER_AREA);

	//cv::matchTemplate(_transform_1, _transform_2, result_CCOEFF_NORMED, CV_TM_CCOEFF_NORMED);
	//cv::matchTemplate(transform_1, transform_2, result_CCOEFF_NORMED, CV_TM_CCOEFF_NORMED);
	cv::matchTemplate(small_transform_1, small_transform_2, result_CCOEFF_NORMED, CV_TM_CCOEFF_NORMED);
	return result_CCOEFF_NORMED.at<float>(0,0);
}

std::pair<cv::Mat, cv::Mat> tfk::Tile::get_overlap_matrix(Tile* other, float scale,
		std::pair<cv::Point2f, cv::Point2f>& offset_info) {
	if (!(this->overlaps_with(other))) {
		return std::make_pair(cv::Mat(0,0,CV_8UC1), cv::Mat(0,0,CV_8UC1));
	}

	cv::Mat tile_p_image_1;
	cv::Mat tile_p_image_2;

	float scale_x = scale;
	float scale_y = scale;
	tile_p_image_1 = this->get_tile_data(Resolution::FULL);
	tile_p_image_2 = other->get_tile_data(Resolution::FULL);

	std::pair<cv::Point2f, cv::Point2f> tile_1_bounds = this->get_bbox();
	std::pair<cv::Point2f, cv::Point2f> tile_2_bounds = other->get_bbox();

	// scale the bbox.
	tile_1_bounds.first.x *= scale_x;
	tile_1_bounds.first.y *= scale_y;
	tile_1_bounds.second.x *= scale_x;
	tile_1_bounds.second.y *= scale_y;
	tile_2_bounds.first.x *= scale_x;
	tile_2_bounds.first.y *= scale_y;
	tile_2_bounds.second.x *= scale_x;
	tile_2_bounds.second.y *= scale_y;

	int nrows = std::min(tile_1_bounds.second.y, tile_2_bounds.second.y) - std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);
	int ncols = std::min(tile_1_bounds.second.x, tile_2_bounds.second.x) - std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
	//printf("rows = %d, cols = %d\n", nrows, ncols);
	if ((nrows <= 0) || (ncols <= 0) ) {
		return std::make_pair(cv::Mat(0,0,CV_8UC1), cv::Mat(0,0,CV_8UC1));
	}
	int offset_x = std::max(tile_1_bounds.first.x, tile_2_bounds.first.x);
	int offset_y = std::max(tile_1_bounds.first.y, tile_2_bounds.first.y);




	cv::Mat transform_1 = cv::Mat::zeros(nrows, ncols, CV_8UC1);
	cv::Mat transform_2 = cv::Mat::zeros(nrows, ncols, CV_8UC1);

	// make the transformed images in the same size with the same cells in the same locations

	// determine start coordinates

	int start_y1 = 1000000;
	int end_y1 = -1;
	int start_x1 = 1000000;
	int end_x1 = -1;
	for (int _y = 0; _y < tile_p_image_1.rows; _y++) {
		int _x = 0;
		cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
		cv::Point2f transformed_p = this->rigid_transform(p)*scale;

		//int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
		int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
		if ((y_c >= 0) && (y_c < nrows)) {
			if (_y < start_y1) {
				start_y1 = _y;
			}
			if (_y > end_y1) {
				end_y1 = _y;
			}
		}
	}
	for (int _x = 0; _x < tile_p_image_1.cols; _x++) {
		int _y = 0;
		cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
		cv::Point2f transformed_p = this->rigid_transform(p)*scale;

		int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
		//int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
		if ((x_c >= 0) && (x_c < ncols)) {
			if (_x < start_x1) {
				start_x1 = _x;
			}
			if (_x > end_x1) {
				end_x1 = _x;
			}
		}
	}

	int start_y2 = 1000000;
	int end_y2 = -1;
	int start_x2 = 1000000;
	int end_x2 = -1;
	for (int _y = 0; _y < tile_p_image_2.rows; _y++) {
		int _x = 0;
		cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
		cv::Point2f transformed_p = other->rigid_transform(p)*scale;

		//int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
		int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
		if ((y_c >= 0) && (y_c < nrows)) {
			if (_y < start_y2) {
				start_y2 = _y;
			}
			if (_y > end_y2) {
				end_y2 = _y;
			}
		}
	}
	for (int _x = 0; _x < tile_p_image_2.cols; _x++) {
		int _y = 0;
		cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
		cv::Point2f transformed_p = other->rigid_transform(p)*scale;

		int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
		//int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
		if ((x_c >= 0) && (x_c < ncols)) {
			if (_x < start_x2) {
				start_x2 = _x;
			}
			if (_x > end_x2) {
				end_x2 = _x;
			}
		}
	}


	for (int _y = start_y1; _y < end_y1; _y++) {
		for (int _x = start_x1; _x < end_x1; _x++) {
			cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
			cv::Point2f transformed_p = this->rigid_transform(p)*scale;

			int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
			int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
			if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
				transform_1.at<unsigned char>(y_c, x_c) +=
					tile_p_image_1.at<unsigned char>(_y, _x);
			}
		}
	}

	for (int _y = start_y2; _y < end_y2; _y++) {
		for (int _x = start_x2; _x < end_x2; _x++) {
			cv::Point2f p = cv::Point2f(_x/scale, _y/scale);
			cv::Point2f transformed_p = other->rigid_transform(p)*scale;

			int x_c = ((int)(transformed_p.x + 0.5)) - offset_x;
			int y_c = ((int)(transformed_p.y + 0.5)) - offset_y;
			if ((y_c >= 0) && (y_c < nrows) && (x_c >= 0) && (x_c < ncols)) {
				transform_2.at<unsigned char>(y_c, x_c) +=
					tile_p_image_2.at<unsigned char>(_y, _x);
			}
		}
	}

	offset_info =
		std::make_pair(cv::Point2f(1.0*start_x1, 1.0*start_y1),
				cv::Point2f(1.0*start_x2, 1.0*start_y2));
	//printf("offset info %d %d %d %d\n", start_x1, start_y1, start_x2, start_y2);

	// clear any location which only has a value for one of them
	// note that the transforms are the same size
	int min_x = 1000000;
	int min_y = 1000000;
	int max_x = -1;
	int max_y = -1;
	for (int _y = 0; _y < transform_1.rows; _y++) {
		for (int _x = 0; _x < transform_1.cols; _x++) {
			if (transform_2.at<unsigned char>(_y, _x) == 0) {
				transform_1.at<unsigned char>(_y, _x) = 0;
			}
			else if (transform_1.at<unsigned char>(_y, _x) == 0) {
				transform_2.at<unsigned char>(_y, _x) = 0;
			} else {
				if (_y < min_y) min_y = _y;
				if (_y > max_y) max_y = _y;
				if (_x < min_x) min_x = _x;
				if (_x > max_x) max_x = _x;
			}
		}
	}

	int small_nrows = max_y - min_y + 1;
	int small_ncols = max_x - min_x + 1;
	if (small_nrows <= 0 || small_ncols <= 0) {
		return std::make_pair(cv::Mat(0,0,CV_8UC1), cv::Mat(0,0,CV_8UC1));
	}
	cv::Mat small_transform_1 = cv::Mat::zeros(small_nrows, small_ncols, CV_8UC1);
	cv::Mat small_transform_2 = cv::Mat::zeros(small_nrows, small_ncols, CV_8UC1);

	for (int _y = 0; _y < transform_1.rows; _y++) {
		for (int _x = 0; _x < transform_1.cols; _x++) {
			if (transform_2.at<unsigned char>(_y, _x) == 0) {
				transform_1.at<unsigned char>(_y, _x) = 0;
			}
			else if (transform_1.at<unsigned char>(_y, _x) == 0) {
				transform_2.at<unsigned char>(_y, _x) = 0;
			} else {
				small_transform_1.at<unsigned char>(_y-min_y, _x-min_x) =
					transform_1.at<unsigned char>(_y, _x);
				small_transform_2.at<unsigned char>(_y-min_y, _x-min_x) =
					transform_2.at<unsigned char>(_y, _x);
			}
		}
	}

	return std::make_pair(small_transform_1, small_transform_2);
}


void tfk::Tile::release_full_image() {
	full_image_lock->lock();
	percent30_lock->lock();
	full_image.release();
	percent30_image.release();
	has_percent30_image = false;
	has_full_image = false;
	percent30_lock->unlock();
	full_image_lock->unlock();
}

void tfk::Tile::release_3d_keypoints() {
	this->p_kps_3d->clear();
	std::vector<cv::KeyPoint>().swap(*(this->p_kps_3d));
	this->p_kps_desc_3d->release();
}


std::vector<cv::Point2f> tfk::Tile::get_corners() {

	std::vector<cv::Point2f> post_corners;

	double dx = this->shape_dx;
	double dy = this->shape_dy;

	cv::Point2f corners[4];
	corners[0] = cv::Point2f(0.0,0.0);
	corners[1] = cv::Point2f(dx,0.0);
	corners[2] = cv::Point2f(0.0,dy);
	corners[3] = cv::Point2f(dx,dy);

	for (int i = 0; i < 4; i++) {
		post_corners.push_back(this->rigid_transform(corners[i]));
	}

	return post_corners;
}

// format is min_x,min_y , max_x,max_y
std::pair<cv::Point2f, cv::Point2f> tfk::Tile::get_bbox() {

	std::vector<cv::Point2f> corners = this->get_corners();
	float min_x = corners[0].x;
	float max_x = corners[0].x;
	float min_y = corners[0].y;
	float max_y = corners[0].y;
	for (int i = 1; i < corners.size(); i++) {
		if (corners[i].x < min_x) min_x = corners[i].x;
		if (corners[i].x > max_x) max_x = corners[i].x;

		if (corners[i].y < min_y) min_y = corners[i].y;
		if (corners[i].y > max_y) max_y = corners[i].y;
	}
	return std::make_pair(cv::Point2f(min_x,min_y), cv::Point2f(max_x, max_y));
}

void tfk::Tile::get_3d_keypoints_limit(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Mat>& desc, int limit) {
	if (this->p_kps_3d->size() <= 0) return;


        float strongest_keypoint = 0.0;
 
	for (int pt_idx = 0; pt_idx < this->p_kps_3d->size(); ++pt_idx) {
		//cv::Point2f pt = this->rigid_transform((*(this->p_kps_3d))[pt_idx].pt);
		cv::KeyPoint kpt = (*(this->p_kps_3d))[pt_idx];
                if (strongest_keypoint < kpt.response) {
                  strongest_keypoint = kpt.response;
                }
	}
        

	for (int pt_idx = 0; pt_idx < this->p_kps_3d->size(); ++pt_idx) {
		cv::Point2f pt = this->rigid_transform((*(this->p_kps_3d))[pt_idx].pt);
		cv::KeyPoint kpt = (*(this->p_kps_3d))[pt_idx];
		kpt.pt = pt;
                if (kpt.response >= strongest_keypoint*0.99) {
		  keypoints.push_back(kpt);
		  desc.push_back(this->p_kps_desc_3d->row(pt_idx).clone());
                }
	}
}

void tfk::Tile::get_3d_keypoints(std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Mat>& desc) {
	if (this->p_kps_3d->size() <= 0) return;

	for (int pt_idx = 0; pt_idx < this->p_kps_3d->size(); ++pt_idx) {
		cv::Point2f pt = this->rigid_transform((*(this->p_kps_3d))[pt_idx].pt);
		cv::KeyPoint kpt = (*(this->p_kps_3d))[pt_idx];
		kpt.pt = pt;
		keypoints.push_back(kpt);
		desc.push_back(this->p_kps_desc_3d->row(pt_idx).clone());
	}
}


void tfk::Tile::local2DAlignUpdateLimited(std::set<Tile*>* active_set) {
	//if (this->bad_2d_alignment) return;

	if (active_set->find(this) == active_set->end()) return;

	if (this->bad_2d_alignment) return;
	//std::vector<edata>& edges = graph->edgeData[vid];
	double global_learning_rate = 0.05;
	if (this->edges.size() == 0) return;
	if (this->edges.size() == 0) return;

	double learning_rate = global_learning_rate;
	double grad_error_x = 0.0;
	double grad_error_y = 0.0;
	double weight_sum = 1.0;

	for (int i = 0; i < this->edges.size(); i++) {
		std::vector<cv::Point2f>* v_points = this->edges[i].v_points;
		std::vector<cv::Point2f>* n_points = this->edges[i].n_points;
		//vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
		Tile* neighbor = (Tile*) this->edges[i].neighbor_tile;
		if (active_set->find(neighbor) == active_set->end()) continue;
		if (neighbor->bad_2d_alignment) continue;
		//if (!(dynamic_cast<MatchTilesTask*>(this->match_tiles_task))->neighbor_to_success[neighbor]) continue;
		if (this->ideal_offsets.find(neighbor->tile_id) == this->ideal_offsets.end() &&
				neighbor->ideal_offsets.find(this->tile_id) == neighbor->ideal_offsets.end()) continue;



		cv::Point2f a_point = cv::Point2f(this->x_start+this->offset_x,
				this->y_start+this->offset_y);
		cv::Point2f b_point = cv::Point2f(neighbor->x_start+neighbor->offset_x,
				neighbor->y_start+neighbor->offset_y);
		cv::Point2f delta = a_point-b_point;
		// c_a_point - c_b_point - a_point + b_point
		// c_a_point - a_point + b_point-c_b_point
		if (v_points->size() == 0 && n_points->size() == 0) continue; 
		cv::Point2f deviation;
		if (this->ideal_offsets.find(neighbor->tile_id) != this->ideal_offsets.end()) {
			deviation = this->ideal_offsets[neighbor->tile_id] - delta;
      } else {
        //          c_b_point - c_a_point - b_point + a_point
        //          a_point - b_point + c_b_point - c_a_point
        //          b_point - a_point + c_a_point - c_b_point


        //            c_b_point - c_a_point + b_point - a_point
        //            c_a_point - c_b_point + b_point - a_point
        //            c_a_point - a_point + b_point - c_b_point
        
        //            - c_b_point + c_a_point - a_point + b_point
        deviation = -1*neighbor->ideal_offsets[this->tile_id] - delta;
      }

      weight_sum += 1;
      if (std::abs(deviation.x) > 2.0 || true) {
        grad_error_x += 2*deviation.x;
      }
      if (std::abs(deviation.y) > 2.0 || true ) {
        grad_error_y += 2*deviation.y;
      }
      continue;
    //double curr_weight = 1.0/v_points->size();

    //for (int j = 0; j < v_points->size(); j++) {
    //  cv::Point2f ptx1 = this->rigid_transform((*v_points)[j]); //transform_point(vertex_data, (*v_points)[j]);
    //  cv::Point2f ptx2 = neighbor->rigid_transform((*n_points)[j]);//transform_point(neighbor_vertex, (*n_points)[j]);

    //  double delta_x = ptx2.x - ptx1.x;
    //  double delta_y = ptx2.y - ptx1.y;
    //  grad_error_x += 2 * delta_x * curr_weight;
    //  grad_error_y += 2 * delta_y * curr_weight;
    //  weight_sum += curr_weight;
    //}
  }

  //printf("gradient %f %f\n", grad_error_x, grad_error_y);

  // update the gradients.
  //this->offset_x += grad_error_x*learning_rate/(weight_sum);
  this->offset_y += grad_error_y*learning_rate/(weight_sum);

}

double tfk::Tile::local2DAlignUpdateEnergyFaster() {
  if (this->bad_2d_alignment) return 0.0;
  if (this->edges.size() == 0) return 0.0;

  double energy = 0.0;
  for (int i = 0; i < this->filtered_edges.size(); i++) {
    Tile* neighbor = this->filtered_edges[i];
    cv::Point2f deviation;
    if (this->filtered_edges_sign[i]) {
      cv::Point2f a_point = this->rigid_transform(this->filtered_edges_points[i].first);
      cv::Point2f b_point = neighbor->rigid_transform(this->filtered_edges_points[i].second);
      cv::Point2f delta =  a_point - b_point;
      deviation = this->filtered_edges_offset[i] - delta;
    } else {
      cv::Point2f a_point =
          this->rigid_transform(this->filtered_edges_points[i].second);
      cv::Point2f b_point =
          neighbor->rigid_transform(this->filtered_edges_points[i].first);
      cv::Point2f delta =  a_point - b_point;
      deviation = -1*this->filtered_edges_offset[i] - delta;
    }
    energy += deviation.x*deviation.x + deviation.y*deviation.y;
  }

  return energy;
}

double tfk::Tile::local2DAlignUpdateEnergy() {
  if (this->bad_2d_alignment) return 0.0;
  //std::vector<edata>& edges = graph->edgeData[vid];
  //double global_learning_rate = 0.01;
  if (this->edges.size() == 0) return 0.0;
  if (this->edges.size() == 0) return 0.0;

  //double learning_rate = global_learning_rate;
  double grad_error_x = 0.0;
  double grad_error_y = 0.0;
  double weight_sum = 0.0;
  double energy = 0.0;

  //std::set<int> found_tile_ids;
  for (int i = 0; i < this->edges.size(); i++) {
    std::vector<cv::Point2f>* v_points = this->edges[i].v_points;
    std::vector<cv::Point2f>* n_points = this->edges[i].n_points;
    //vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
    Tile* neighbor = (Tile*) this->edges[i].neighbor_tile;
    if (neighbor->bad_2d_alignment) continue;

    if (this->ideal_offsets.find(neighbor->tile_id) == this->ideal_offsets.end() &&
        neighbor->ideal_offsets.find(this->tile_id) == neighbor->ideal_offsets.end()) continue;


    //if (found_tile_ids.find(neighbor->tile_id) != found_tile_ids.end()) {printf("does this ever happen?\n"); continue;}
    //found_tile_ids.insert(neighbor->tile_id);


      //cv::Point2f a_point = cv::Point2f(this->x_start+this->offset_x,
      //                                  this->y_start+this->offset_y);
      //cv::Point2f b_point = cv::Point2f(neighbor->x_start+neighbor->offset_x,
      //                                  neighbor->y_start+neighbor->offset_y);
      //cv::Point2f delta = a_point-b_point;
                              // c_a_point - c_b_point - a_point + b_point
                              // c_a_point - a_point + b_point-c_b_point
      if (v_points->size() == 0 && n_points->size() == 0) continue; 
      cv::Point2f deviation;
      if (this->ideal_offsets.find(neighbor->tile_id) != this->ideal_offsets.end()) {

        cv::Point2f a_point = this->rigid_transform(this->ideal_points[neighbor->tile_id].first);
        cv::Point2f b_point = neighbor->rigid_transform(this->ideal_points[neighbor->tile_id].second);
        cv::Point2f delta =  a_point - b_point;
        deviation = this->ideal_offsets[neighbor->tile_id] - delta;
      } else {
        cv::Point2f a_point =
            this->rigid_transform(neighbor->ideal_points[this->tile_id].second);
        cv::Point2f b_point =
            neighbor->rigid_transform(neighbor->ideal_points[this->tile_id].first);

        cv::Point2f delta =  a_point - b_point;

        //          c_b_point - c_a_point - b_point + a_point
        //          a_point - b_point + c_b_point - c_a_point
        //          b_point - a_point + c_a_point - c_b_point


        //            c_b_point - c_a_point + b_point - a_point
        //            c_a_point - c_b_point + b_point - a_point
        //            c_a_point - a_point + b_point - c_b_point

        //            - c_b_point + c_a_point - a_point + b_point
        deviation = -1*neighbor->ideal_offsets[this->tile_id] - delta;

      }
      //float dist = sqrt(deviation.x*deviation.x + deviation.y*deviation.y);
      //if (dist > 10.0) {
      //  printf("The dist is %f\n", dist);
      //}
      weight_sum = 1;
      //this->offset_x += deviation.x*learning_rate;
      //this->offset_y += deviation.y*learning_rate;
      energy += deviation.x*deviation.x + deviation.y*deviation.y;
      
      if (std::abs(deviation.x) > 3.0 || true ) {
        grad_error_x += 2*deviation.x;
      }
      if (std::abs(deviation.y) > 3.0 || true ) {
        grad_error_y += 2*deviation.y;
      }
      //continue;
    //double curr_weight = 1.0/v_points->size();

    //for (int j = 0; j < v_points->size(); j++) {
    //  cv::Point2f ptx1 = this->rigid_transform((*v_points)[j]); //transform_point(vertex_data, (*v_points)[j]);
    //  cv::Point2f ptx2 = neighbor->rigid_transform((*n_points)[j]);//transform_point(neighbor_vertex, (*n_points)[j]);

    //  double delta_x = ptx2.x - ptx1.x;
    //  double delta_y = ptx2.y - ptx1.y;
    //  grad_error_x += 2 * delta_x * curr_weight;
    //  grad_error_y += 2 * delta_y * curr_weight;
    //  weight_sum += curr_weight;
    //}
  }
  return energy;
  //printf("gradient %f %f\n", grad_error_x, grad_error_y);
  //if (weight_sum > 0.5) {
  //// update the gradients.
  ////if (grad_error_x / (weight_sum) > 10.0 ||
  ////    grad_error_y / (weight_sum) > 10.0) {
  ////  printf("grad error x is %f y is %f\n", grad_error_x, grad_error_y); 
  ////}
  //this->offset_x += grad_error_x*learning_rate/(weight_sum);
  //this->offset_y += grad_error_y*learning_rate/(weight_sum);
  //printf("offset x is %f\n", this->offset_x);
  //}
}

void tfk::Tile::local2DAlignUpdate_filter_edges_cleanup() {

   this->filtered_edges.clear();
   std::vector<Tile*>().swap(this->filtered_edges);
   this->filtered_edges_points.clear();
   std::vector<std::pair<cv::Point2f, cv::Point2f> >().swap(this->filtered_edges_points);

   this->filtered_edges_sign.clear();
   std::vector<bool>().swap(this->filtered_edges_sign);
}

void tfk::Tile::local2DAlignUpdate_filter_edges() {//, int num_angles) {
  this->filtered_edges.clear();
  this->filtered_edges_points.clear();
  this->filtered_edges_sign.clear();

  if (this->bad_2d_alignment) return;
  if (this->edges.size() == 0) return;

  for (int i = 0; i < this->edges.size(); i++) {
    Tile* neighbor = (Tile*) this->edges[i].neighbor_tile;
    std::vector<cv::Point2f>* v_points = this->edges[i].v_points;
    std::vector<cv::Point2f>* n_points = this->edges[i].n_points;
    if (neighbor->bad_2d_alignment) continue;
    if (this->ideal_offsets.find(neighbor->tile_id) == this->ideal_offsets.end() &&
        neighbor->ideal_offsets.find(this->tile_id) == neighbor->ideal_offsets.end()) continue;


    if (v_points->size() == 0 && n_points->size() == 0) continue;

    this->filtered_edges.push_back(neighbor);

    if (this->ideal_offsets.find(neighbor->tile_id) != this->ideal_offsets.end()) {
      this->filtered_edges_points.push_back(this->ideal_points[neighbor->tile_id]); 
      this->filtered_edges_sign.push_back(true);
      this->filtered_edges_offset.push_back(this->ideal_offsets[neighbor->tile_id]);
      //cv::Point2f a_point = this->rigid_transform(this->ideal_points[neighbor->tile_id].first);
      //cv::Point2f b_point = neighbor->rigid_transform(this->ideal_points[neighbor->tile_id].second);
      //cv::Point2f delta =  a_point - b_point;
      //deviation = this->ideal_offsets[neighbor->tile_id] - delta;
    } else {
      this->filtered_edges_points.push_back(neighbor->ideal_points[this->tile_id]);
      this->filtered_edges_sign.push_back(false);
      this->filtered_edges_offset.push_back(neighbor->ideal_offsets[this->tile_id]);
      //cv::Point2f a_point =
      //    this->rigid_transform(neighbor->ideal_points[this->tile_id].second);
      //cv::Point2f b_point =
      //    neighbor->rigid_transform(neighbor->ideal_points[this->tile_id].first);

      //cv::Point2f delta =  a_point - b_point;

      //deviation = -1*neighbor->ideal_offsets[this->tile_id] - delta;
      //continue;
    }

  }

}

void tfk::Tile::local2DAlignUpdateFaster(double lr) {//, int num_angles) {
  if (this->bad_2d_alignment) return;

  if (this->edges.size() == 0) return;
  if (this->edges.size() == 0) return;


    double grad_error_x = 0.0;
    double grad_error_y = 0.0;
    double weight_sum = 0.0;

    for (int i = 0; i < this->filtered_edges.size(); i++) {
      Tile* neighbor = this->filtered_edges[i];
      cv::Point2f deviation;
      if (this->filtered_edges_sign[i]) {

        cv::Point2f a_point = this->rigid_transform(this->filtered_edges_points[i].first);
        cv::Point2f b_point = neighbor->rigid_transform(this->filtered_edges_points[i].second);
        cv::Point2f delta =  a_point - b_point;
        deviation = this->filtered_edges_offset[i] - delta;
      } else {
        cv::Point2f a_point =
            this->rigid_transform(this->filtered_edges_points[i].second);
        cv::Point2f b_point =
            neighbor->rigid_transform(this->filtered_edges_points[i].first);

        cv::Point2f delta =  a_point - b_point;
        deviation = -1*this->filtered_edges_offset[i] - delta;
      }
      weight_sum = 1;
      grad_error_x += 2*deviation.x;
      grad_error_y += 2*deviation.y;
    }
    this->grad_error_x = grad_error_x / this->edges.size();
    this->grad_error_y = grad_error_y / this->edges.size();

}


void tfk::Tile::local2DAlignUpdate() {
  if (this->bad_2d_alignment) return;
  //std::vector<edata>& edges = graph->edgeData[vid];
  double global_learning_rate = 0.45;//0.45;
  if (this->edges.size() == 0) return;
  if (this->edges.size() == 0) return;

  double learning_rate = global_learning_rate;
  double grad_error_x = 0.0;
  double grad_error_y = 0.0;
  double weight_sum = 0.0;


  //std::set<int> found_tile_ids;
  //int index = (this->iteration_count++)%this->edges.size();
  for (int i = 0; i < this->edges.size(); i++) {
    //if (i != index) continue;
    std::vector<cv::Point2f>* v_points = this->edges[i].v_points;
    std::vector<cv::Point2f>* n_points = this->edges[i].n_points;
    //vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
    Tile* neighbor = (Tile*) this->edges[i].neighbor_tile;
    if (neighbor->bad_2d_alignment) continue;

    if (this->ideal_offsets.find(neighbor->tile_id) == this->ideal_offsets.end() &&
        neighbor->ideal_offsets.find(this->tile_id) == neighbor->ideal_offsets.end()) continue;


    //if (found_tile_ids.find(neighbor->tile_id) != found_tile_ids.end()) continue;
    //found_tile_ids.insert(neighbor->tile_id);


      cv::Point2f a_point = cv::Point2f(this->x_start+this->offset_x,
                                        this->y_start+this->offset_y);
      cv::Point2f b_point = cv::Point2f(neighbor->x_start+neighbor->offset_x,
                                        neighbor->y_start+neighbor->offset_y);
      cv::Point2f delta = a_point-b_point;
                              // c_a_point - c_b_point - a_point + b_point
                              // c_a_point - a_point + b_point-c_b_point
      if (v_points->size() == 0 && n_points->size() == 0) continue; 
      cv::Point2f deviation;
      if (this->ideal_offsets.find(neighbor->tile_id) != this->ideal_offsets.end()) {
        deviation = this->ideal_offsets[neighbor->tile_id] - delta;
      } else {
        //          c_b_point - c_a_point - b_point + a_point
        //          a_point - b_point + c_b_point - c_a_point
        //          b_point - a_point + c_a_point - c_b_point


        //            c_b_point - c_a_point + b_point - a_point
        //            c_a_point - c_b_point + b_point - a_point
        //            c_a_point - a_point + b_point - c_b_point

        //            - c_b_point + c_a_point - a_point + b_point
        deviation = -1*neighbor->ideal_offsets[this->tile_id] - delta;
        //continue;
      }
      float dist = sqrt(deviation.x*deviation.x + deviation.y*deviation.y);
      //if (dist > 10.0) {
      //  printf("The dist is %f\n", dist);
      //}
      //this->offset_x += deviation.x*learning_rate;
      //this->offset_y += deviation.y*learning_rate;
      double damp = 1.0;


      weight_sum += damp;

      //if (this->mfov_id != neighbor->mfov_id) {
      //  damp = 0.5;
      //  damp = damp*damp;
      //}


      if (dist > 0.0) {
        if (std::abs(deviation.x) > 3.0 || true ) {
          //grad_error_x += 2*deviation.x*damp;
          grad_error_x += deviation.x;
        }
        if (std::abs(deviation.y) > 3.0 || true ) {
          grad_error_y += deviation.y;
          //grad_error_y += 2*deviation.y*damp;
        }
     }
      //continue;
    //double curr_weight = 1.0/v_points->size();

    //for (int j = 0; j < v_points->size(); j++) {
    //  cv::Point2f ptx1 = this->rigid_transform((*v_points)[j]); //transform_point(vertex_data, (*v_points)[j]);
    //  cv::Point2f ptx2 = neighbor->rigid_transform((*n_points)[j]);//transform_point(neighbor_vertex, (*n_points)[j]);

    //  double delta_x = ptx2.x - ptx1.x;
    //  double delta_y = ptx2.y - ptx1.y;
    //  grad_error_x += 2 * delta_x * curr_weight;
    //  grad_error_y += 2 * delta_y * curr_weight;
    //  weight_sum += curr_weight;
    //}
  }

  //printf("gradient %f %f\n", grad_error_x, grad_error_y);
  //if (weight_sum > 0.5) {
  //// update the gradients.
  ////if (grad_error_x / (weight_sum) > 10.0 ||
  ////    grad_error_y / (weight_sum) > 10.0) {
  ////  printf("grad error x is %f y is %f\n", grad_error_x, grad_error_y); 
  ////}

  //if (this->offset_x > 1.0) {
  //  grad_error_x -= this->offset_x / std::pow(std::abs(this->offset_x),0.25);
  //}
  //if (this->offset_y > 1.0) {
  //  grad_error_y -= this->offset_y / std::pow(std::abs(this->offset_y),0.25);
  //}
  if (weight_sum > 0.5) {
    this->offset_x += grad_error_x*learning_rate/(weight_sum);
    this->offset_y += grad_error_y*learning_rate/(weight_sum);
  }
  //printf("offset x is %f\n", this->offset_x);
  //}
}


void tfk::Tile::local2DAlignUpdate(double lr) {//, int num_angles) {
  if (this->bad_2d_alignment) return;

  if (this->edges.size() == 0) return;
  if (this->edges.size() == 0) return;

  //double best_delta_angle = 0.0;
  double best_grad_error_x = 0.0;
  double best_grad_error_y = 0.0;
  double best_deviation_energy = 0.0;
  bool first_step = true;

  //for (int angle_step = 1-num_angles; angle_step < num_angles; angle_step++) {
  for (int angle_step = 0; angle_step < 1; angle_step++) {

    //double delta_angle = angle_step*0.00001;

    double grad_error_x = 0.0;
    double grad_error_y = 0.0;
    double weight_sum = 0.0;
    //std::set<int> found_tile_ids;
    //double angle_sum = 0.0;
    //double angle_sum_count = 0.0;

    for (int i = 0; i < this->edges.size(); i++) {
      std::vector<cv::Point2f>* v_points = this->edges[i].v_points;
      std::vector<cv::Point2f>* n_points = this->edges[i].n_points;
      Tile* neighbor = (Tile*) this->edges[i].neighbor_tile;
      if (neighbor->bad_2d_alignment) continue;
      if (this->ideal_offsets.find(neighbor->tile_id) == this->ideal_offsets.end() &&
          neighbor->ideal_offsets.find(this->tile_id) == neighbor->ideal_offsets.end()) continue;

      //if (found_tile_ids.find(neighbor->tile_id) != found_tile_ids.end()) {printf("does this ever happen?\n"); continue;}
      //found_tile_ids.insert(neighbor->tile_id);

      if (v_points->size() == 0 && n_points->size() == 0) continue; 
      cv::Point2f deviation;
      if (this->ideal_offsets.find(neighbor->tile_id) != this->ideal_offsets.end()) {
        cv::Point2f a_point = this->rigid_transform(this->ideal_points[neighbor->tile_id].first);
        cv::Point2f b_point = neighbor->rigid_transform(this->ideal_points[neighbor->tile_id].second);
        cv::Point2f delta =  a_point - b_point;
        deviation = this->ideal_offsets[neighbor->tile_id] - delta;
      } else {
        cv::Point2f a_point =
            this->rigid_transform(neighbor->ideal_points[this->tile_id].second);
        cv::Point2f b_point =
            neighbor->rigid_transform(neighbor->ideal_points[this->tile_id].first);

        cv::Point2f delta =  a_point - b_point;

        deviation = -1*neighbor->ideal_offsets[this->tile_id] - delta;
        //continue;
      }
      weight_sum = 1;

      if (std::abs(deviation.x) > 3.0 || true ) {
        grad_error_x += 2*deviation.x;
      }
      if (std::abs(deviation.y) > 3.0 || true ) {
        grad_error_y += 2*deviation.y;
      }
    }
    cv::Point2f deviation = cv::Point2f(grad_error_x, grad_error_y);

    //if (deviation.x*deviation.x + deviation.y*deviation.y < best_deviation_energy || first_step) {
    if (first_step /*deviation.x*deviation.x + deviation.y*deviation.y < best_deviation_energy || first_step*/) {
      //best_deviation_energy = deviation.x*deviation.x + deviation.y*deviation.y;
      best_grad_error_x = grad_error_x / this->edges.size();
      best_grad_error_y = grad_error_y / this->edges.size();
    }
    if (deviation.x*deviation.x + deviation.y*deviation.y < best_deviation_energy || first_step) {
      best_deviation_energy = deviation.x*deviation.x + deviation.y*deviation.y;
    }
    first_step = false;
  }

  this->grad_error_x = best_grad_error_x;
  this->grad_error_y = best_grad_error_y;
  //printf("gradient %f %f\n", grad_error_x, grad_error_y);
  //if (weight_sum > 0.5) {
  //// update the gradients.
  ////if (grad_error_x / (weight_sum) > 10.0 ||
  ////    grad_error_y / (weight_sum) > 10.0) {
  ////  printf("grad error x is %f y is %f\n", grad_error_x, grad_error_y); 
  ////}
  //this->offset_x += grad_error_x*learning_rate/(weight_sum);
  //this->offset_y += grad_error_y*learning_rate/(weight_sum);
  //printf("offset x is %f\n", this->offset_x);
  //}
}


//void tfk::Tile::local2DAlignUpdate() {
//  if (this->bad_2d_alignment) return;
//  //std::vector<edata>& edges = graph->edgeData[vid];
//  double global_learning_rate = 0.45;//0.45;
//  if (this->edges.size() == 0) return;
//  if (this->edges.size() == 0) return;
//
//  double learning_rate = global_learning_rate;
//  double grad_error_x = 0.0;
//  double grad_error_y = 0.0;
//  double weight_sum = 0.0;
//
//
//  //std::set<int> found_tile_ids;
//  //int index = (this->iteration_count++)%this->edges.size();
//  for (int i = 0; i < this->edges.size(); i++) {
//    //if (i != index) continue;
//    std::vector<cv::Point2f>* v_points = this->edges[i].v_points;
//    std::vector<cv::Point2f>* n_points = this->edges[i].n_points;
//    //vdata* neighbor_vertex = graph->getVertexData(edges[i].neighbor_id);
//    Tile* neighbor = (Tile*) this->edges[i].neighbor_tile;
//    if (neighbor->bad_2d_alignment) continue;
//
//    if (this->ideal_offsets.find(neighbor->tile_id) == this->ideal_offsets.end() &&
//        neighbor->ideal_offsets.find(this->tile_id) == neighbor->ideal_offsets.end()) continue;
//
//
//    //if (found_tile_ids.find(neighbor->tile_id) != found_tile_ids.end()) continue;
//    //found_tile_ids.insert(neighbor->tile_id);
//
//
//      cv::Point2f a_point = cv::Point2f(this->x_start+this->offset_x,
//                                        this->y_start+this->offset_y);
//      cv::Point2f b_point = cv::Point2f(neighbor->x_start+neighbor->offset_x,
//                                        neighbor->y_start+neighbor->offset_y);
//      cv::Point2f delta = a_point-b_point;
//                              // c_a_point - c_b_point - a_point + b_point
//                              // c_a_point - a_point + b_point-c_b_point
//      if (v_points->size() == 0 && n_points->size() == 0) continue; 
//      cv::Point2f deviation;
//      if (this->ideal_offsets.find(neighbor->tile_id) != this->ideal_offsets.end()) {
//        deviation = this->ideal_offsets[neighbor->tile_id] - delta;
//      } else {
//        //          c_b_point - c_a_point - b_point + a_point
//        //          a_point - b_point + c_b_point - c_a_point
//        //          b_point - a_point + c_a_point - c_b_point
//
//
//        //            c_b_point - c_a_point + b_point - a_point
//        //            c_a_point - c_b_point + b_point - a_point
//        //            c_a_point - a_point + b_point - c_b_point
//
//        //            - c_b_point + c_a_point - a_point + b_point
//        deviation = -1*neighbor->ideal_offsets[this->tile_id] - delta;
//        //continue;
//      }
//      float dist = sqrt(deviation.x*deviation.x + deviation.y*deviation.y);
//      //if (dist > 10.0) {
//      //  printf("The dist is %f\n", dist);
//      //}
//      //this->offset_x += deviation.x*learning_rate;
//      //this->offset_y += deviation.y*learning_rate;
//      double damp = 1.0;
//
//
//      weight_sum += damp;
//
//      //if (this->mfov_id != neighbor->mfov_id) {
//      //  damp = 0.5;
//      //  damp = damp*damp;
//      //}
//
//
//      if (dist > 0.0) {
//        if (std::abs(deviation.x) > 3.0 || true ) {
//          //grad_error_x += 2*deviation.x*damp;
//          grad_error_x += deviation.x;
//        }
//        if (std::abs(deviation.y) > 3.0 || true ) {
//          grad_error_y += deviation.y;
//          //grad_error_y += 2*deviation.y*damp;
//        }
//     }
//      //continue;
//    //double curr_weight = 1.0/v_points->size();
//
//    //for (int j = 0; j < v_points->size(); j++) {
//    //  cv::Point2f ptx1 = this->rigid_transform((*v_points)[j]); //transform_point(vertex_data, (*v_points)[j]);
//    //  cv::Point2f ptx2 = neighbor->rigid_transform((*n_points)[j]);//transform_point(neighbor_vertex, (*n_points)[j]);
//
//    //  double delta_x = ptx2.x - ptx1.x;
//    //  double delta_y = ptx2.y - ptx1.y;
//    //  grad_error_x += 2 * delta_x * curr_weight;
//    //  grad_error_y += 2 * delta_y * curr_weight;
//    //  weight_sum += curr_weight;
//    //}
//  }
//
//  //printf("gradient %f %f\n", grad_error_x, grad_error_y);
//  //if (weight_sum > 0.5) {
//  //// update the gradients.
//  ////if (grad_error_x / (weight_sum) > 10.0 ||
//  ////    grad_error_y / (weight_sum) > 10.0) {
//  ////  printf("grad error x is %f y is %f\n", grad_error_x, grad_error_y); 
//  ////}
//
//  //if (this->offset_x > 1.0) {
//  //  grad_error_x -= this->offset_x / std::pow(std::abs(this->offset_x),0.25);
//  //}
//  //if (this->offset_y > 1.0) {
//  //  grad_error_y -= this->offset_y / std::pow(std::abs(this->offset_y),0.25);
//  //}
//  if (weight_sum > 0.5) {
//    this->offset_x += grad_error_x*learning_rate/(weight_sum);
//    this->offset_y += grad_error_y*learning_rate/(weight_sum);
//  }
//  //printf("offset x is %f\n", this->offset_x);
//  //}
//}


void tfk::Tile::make_symmetric(int phase, std::vector<Tile*>& tile_list) {
  if (phase == 0) {
    if (this->bad_2d_alignment) return;
    for (int i = 0; i < tile_list.size(); i++) {
      Tile* other = tile_list[i];
      if (other->bad_2d_alignment) continue;
      for (int j = 0; j < other->edges.size(); j++) {
        edata edge = other->edges[j];
        if (edge.neighbor_id == this->tile_id) {
          edata edge2;
          edge2.v_points = edge.n_points;
          edge2.n_points = edge.v_points;
          edge2.neighbor_id = other->tile_id;
          edge2.neighbor_tile = other;
          edge2.weight = 1.0;
          //printf("adding symmetricn edge %d %d\n", this->tile_id, other->tile_id);
          this->add_edges.push_back(edge2);
        }
      }
    }
  } else if (phase == 1) {
    if (this->bad_2d_alignment) return;
    for (int i = 0; i < this->add_edges.size(); i++) {
      this->edges.push_back(this->add_edges[i]);
    }
  }
}


void tfk::Tile::insert_matches(Tile* neighbor, std::vector<cv::Point2f>& points_a, std::vector<cv::Point2f>& points_b) {
  std::vector<cv::Point2f>* vedges = new std::vector<cv::Point2f>();
  std::vector<cv::Point2f>* nedges = new std::vector<cv::Point2f>();


  // NOTE(TFK): This count thing is just quick hack to reduce space since we use ideal-offsets now...
  int count = 0;
  for (int i = 0; i < points_a.size(); i++) {
    vedges->push_back(cv::Point2f(points_a[i]));
    if (count++ > 4) break;
  }
  count = 0;
  for (int i = 0; i < points_b.size(); i++) {
    nedges->push_back(cv::Point2f(points_b[i]));
    if (count++ > 4) break;
  }

  edata edge1;
  edge1.v_points = vedges;
  edge1.n_points = nedges;
  edge1.neighbor_id = neighbor->tile_id;
  edge1.neighbor_tile = (void*) neighbor;
  edge1.weight = 1.0;

  this->edges.push_back(edge1);



  //this->insertEdge(atile_id, edge1);
  // make bi-directional later.
  //edata edge2;
  //edge2.v_points = nedges;
  //edge2.n_points = vedges;
  //edge2.neighbor_id = atile_id;
  //edge2.weight = weight;
  //this->insertEdge(btile_id, edge2);
}

tfk::Tile::Tile(int section_id, int tile_id, int index, std::string filepath,
    int x_start, int x_finish, int y_start, int y_finish) {
  this->section_id = section_id;
  this->tile_id = tile_id;
  this->index = index;
  this->filepath = filepath;
  this->x_start = x_start;
  this->x_finish = x_start + 3128;//x_finish;
  this->y_start = y_start;
  this->y_finish = y_start + 2724;//y_finish;
  this->offset_x = 0.0;
  this->offset_y = 0.0;
  //this->angle = 0.0;
  this->image_data_replaced = false;
}

cv::Point2f tfk::Tile::rigid_transform_d(cv::Point2f pt) {
  cv::Point2f pt2 = cv::Point2f(pt.x+this->offset_x+this->x_start, pt.y+this->offset_y+this->y_start);
  //cv::Point2f pt2 = cv::Point2f(pt.x+this->x_start, pt.y+this->y_start);
  return pt2;
}

cv::Point2f rotate2d(const cv::Point2f& inPoint, const double& angRad)
{
    cv::Point2f outPoint;
    //CW rotation
    outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
    outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;
    return outPoint;
}

cv::Point2f rotatePoint(const cv::Point2f& inPoint, const cv::Point2f& center, const double& angRad)
{
    return rotate2d(inPoint - center, angRad) + center;
}

cv::Point2f tfk::Tile::rigid_transform(cv::Point2f pt) {
  // apply rotation.
  //double dx = this->shape_dx;
  //double dy = this->shape_dy;

  //cv::Point2f corners[4];
  //corners[0] = cv::Point2f(0.0,0.0);
  //corners[1] = cv::Point2f(dx,0.0);
  //corners[2] = cv::Point2f(0.0,dy);
  //corners[3] = cv::Point2f(dx,dy);

  //cv::Point2f midpoint = 0.5*cv::Point2f(dx,dy);

  //double _angle = this->angle;
  //cv::Point rotated_pt = rotatePoint(pt, midpoint, _angle);

  //cv::Point2f pt2 = cv::Point2f(rotated_pt.x+this->offset_x+this->x_start, rotated_pt.y+this->offset_y+this->y_start);

  cv::Point2f pt2 = cv::Point2f(pt.x+this->offset_x+this->x_start, pt.y+this->offset_y+this->y_start);
  //cv::Point2f pt2 = cv::Point2f(pt.x+this->x_start, pt.y+this->y_start);
  return pt2;
}

//cv::Point2f tfk::Tile::rigid_transform_angle(cv::Point2f pt, double _angle) {
//  //return rigid_transform(pt);
//  // apply rotation.
//  double dx = this->shape_dx;
//  double dy = this->shape_dy;
//
//  //cv::Point2f corners[4];
//  //corners[0] = cv::Point2f(0.0,0.0);
//  //corners[1] = cv::Point2f(dx,0.0);
//  //corners[2] = cv::Point2f(0.0,dy);
//  //corners[3] = cv::Point2f(dx,dy);
//  cv::Point2f midpoint = 0.5*cv::Point2f(dx,dy);
//  cv::Point rotated_pt = rotatePoint(pt, midpoint, _angle);
//
//  cv::Point2f pt2 = cv::Point2f(rotated_pt.x+this->offset_x+this->x_start, rotated_pt.y+this->offset_y+this->y_start);
//
//  //cv::Point2f pt2 = cv::Point2f(pt.x+this->offset_x+this->x_start, pt.y+this->offset_y+this->y_start);
//  //cv::Point2f pt2 = cv::Point2f(pt.x+this->x_start, pt.y+this->y_start);
//  return pt2;
//}




bool tfk::Tile::overlaps_with(std::pair<cv::Point2f, cv::Point2f> bbox) {
    float x1_start = this->x_start + this->offset_x;
    float x1_finish = this->x_finish + this->offset_x;
    float y1_start = this->y_start + this->offset_y;
    float y1_finish = this->y_finish + this->offset_y;

    float x2_start = bbox.first.x;
    float x2_finish = bbox.second.x;
    float y2_start = bbox.first.y;
    float y2_finish = bbox.second.y;

    bool res = false;
    if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
        (y1_start < y2_finish) && (y1_finish > y2_start)) {
        res = true;
    }
    return res;
}

bool tfk::Tile::overlaps_with_threshold(Tile* other, int min_dim_overlap) {
    int x1_start = this->x_start + this->offset_x;
    int x1_finish = this->x_finish + this->offset_x;
    int y1_start = this->y_start + this->offset_y;
    int y1_finish = this->y_finish + this->offset_y;

    int x2_start = other->x_start + other->offset_x;
    int x2_finish = other->x_finish + other->offset_x;
    int y2_start = other->y_start + other->offset_y;
    int y2_finish = other->y_finish + other->offset_y;


    // overlap needs to be big enough.

    /*
       x1s-----x1e
          x2s-------x2e
       (x1e - x2s) 

           x1s-----x2e
       x2s------x2e
       (x2e - x1s)
    */

    int x_size = min(x1_finish - x2_start, x2_finish - x1_start);
    int y_size = min(y1_finish - y2_start, y2_finish - y1_start);


    if (min_dim_overlap == 51) {
      printf("x_size %d, y_size %d\n", x_size, y_size);
      printf("x1_start %d, x1_finish %d, x2_start %d x2_finish %d\n", x1_start, x1_finish, x2_start, x2_finish);
      printf("x1_offset %f, y1_offset %f, x2_offset %f y2_offset %f\n", this->offset_x, this->offset_y, other->offset_x, other->offset_y);
      min_dim_overlap = 50;
    }


    bool res = false;
    if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
        (y1_start < y2_finish) && (y1_finish > y2_start) &&
        x_size >= min_dim_overlap && y_size >= min_dim_overlap) {
        res = true;
    }
    return res;
}

bool tfk::Tile::overlaps_with(Tile* other) {
    int x1_start = this->x_start + this->offset_x;
    int x1_finish = this->x_finish + this->offset_x;
    int y1_start = this->y_start + this->offset_y;
    int y1_finish = this->y_finish + this->offset_y;

    int x2_start = other->x_start + other->offset_x;
    int x2_finish = other->x_finish + other->offset_x;
    int y2_start = other->y_start + other->offset_y;
    int y2_finish = other->y_finish + other->offset_y;

    bool res = false;
    if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
        (y1_start < y2_finish) && (y1_finish > y2_start)) {
        res = true;
    }
    return res;
}

tfk::Tile::Tile(TileData& tile_data) {
    //tile_data_t *p_cur_tile = &(p_sec_data->tiles[p_sec_data->n_tiles]);
    //p_sec_data->n_tiles++;
    this->has_full_image = false;
    this->has_percent30_image = false;
    this->full_image_lock = new std::mutex();
    this->percent30_lock = new std::mutex();
    this->image_data_replaced = false;
    //this->shape_dx = tile_data.x_finish() - tile_data.x_start();
    //this->shape_dy = tile_data.y_finish() - tile_data.y_start();
    this->shape_dx = 3128;//tile_data.x_finish() - tile_data.x_start();
    this->shape_dy = 2724;//tile_data.y_finish() - tile_data.y_start();
    this->bad_2d_alignment = false;
    this->section_id = tile_data.section_id();
    this->mfov_id = tile_data.tile_mfov();
    this->index = tile_data.tile_index();
    this->x_start = tile_data.x_start();
    this->x_finish = this->x_start + this->shape_dx;//tile_data.x_finish();
    this->y_start = tile_data.y_start();
    this->y_finish = this->y_start + this->shape_dy;//tile_data.y_finish();
    this->offset_x = 0.0;
    this->offset_y = 0.0;
    //this->angle = 0.0;
    this->filepath = tile_data.tile_filepath();
    //printf("mfov id is %d\n", mfov_id);

    // NOTE TFK HACK

    //this->filepath = this->filepath.replace(this->filepath.find("/efs"), 4, "/home/gridsan/groups/supertech/connectomix");


    //printf("filepath %s\n", this->filepath.c_str());

    #ifdef IARPAFULL
      this->filepath = this->filepath.replace(this->filepath.find(".bmp"),4,".jpg");
      this->filepath = this->filepath.replace(this->filepath.find("IARPA_FULL"),10,"IARPA_Dataset");
      this->filepath = this->filepath.replace(this->filepath.find("sep14iarpa"),10,"compressed2");
    #endif

    #ifdef NOPATHCHANGE

    #else

      #ifdef HUMANTEST
      this->filepath = this->filepath.replace(this->filepath.find(".bmp"),4,".bmp.jp2");
      //this->filepath = this->filepath.replace(this->filepath.find(".png"),4,".j2k");
      #else
      //this->filepath = this->filepath.replace(this->filepath.find(".bmp"),4,".bmp.jp2");
      this->filepath = this->filepath.replace(this->filepath.find(".bmp"),4,".jpg");
      this->filepath = this->filepath.replace(this->filepath.find("sep14iarpa"),10,"compressed2");
      #endif
    #endif



    this->p_image = new cv::Mat();

    this->p_kps_3d = new std::vector<cv::KeyPoint>();
    this->p_kps_desc_3d = new cv::Mat();

}

void tfk::Tile::recompute_3d_keypoints(std::vector<cv::KeyPoint>& atile_all_kps,
                                       std::vector<cv::Mat>& atile_all_kps_desc,
                                       tfk::params sift_parameters) {

  cv::Mat local_p_image;
  local_p_image.create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  local_p_image = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);


  //this->p_kps_3d = new std::vector<cv::KeyPoint>();

  int rows = local_p_image.rows;
  int cols = local_p_image.cols;

  ASSERT((rows % SIFT_D1_SHIFT_3D) == 0);
  ASSERT((cols % SIFT_D2_SHIFT_3D) == 0);

  std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
  cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];
  int n_sub_images;


  cv::Ptr<cv::Feature2D> p_sift;
  p_sift = new cv::xfeatures2d::SIFT_Impl(
            sift_parameters.num_features,  // num_features --- unsupported.
            sift_parameters.num_octaves,  // number of octaves
            sift_parameters.contrast_threshold,  // contrast threshold.
            sift_parameters.edge_threshold,  // edge threshold.
            sift_parameters.sigma);  // sigma.

    int max_rows = rows / SIFT_D1_SHIFT_3D;
    int max_cols = cols / SIFT_D2_SHIFT_3D;
    n_sub_images = max_rows * max_cols;
        cv::Mat sub_im_mask = cv::Mat::ones(0,0,
            CV_8UC1);
        int sub_im_id = 0;
        // Detect the SIFT features within the subimage.
        //fasttime_t tstart = gettime();
        p_sift->detectAndCompute((local_p_image), sub_im_mask, v_kps[sub_im_id],
            m_kps_desc[sub_im_id], false);

        //fasttime_t tend = gettime();
        //totalTime += tdiff(tstart, tend);
    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    int point_count_3d = 0;
    for (int _i = 0; _i < n_sub_images; _i++) {
        for (int _j = 0; _j < v_kps[_i].size(); _j++) {
            cv::Point2f pt = this->rigid_transform(v_kps[_i][_j].pt);
            cv::KeyPoint kpt = v_kps[_i][_j];
            kpt.pt = pt;
            atile_all_kps.push_back(kpt);
            atile_all_kps_desc.push_back(m_kps_desc[_i].row(_j).clone());
            point_count_3d++;
        }
    }

  //cv::Mat m_kps_desc_filtered = m_kps_desc[0].clone();
  //*(this)->p_kps_desc_3d = m_kps_desc[0].clone();

  //atile_all_kps_desc.push_back(m_kps_desc[0].clone());

  //printf("Number of 3d points is %d\n", point_count_3d);
  local_p_image.release();
}


void tfk::Tile::compute_sift_keypoints3d(bool recomputation) {
  //(*this->p_image).create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  //(*this->p_image) = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);

  cv::Mat tmp_image;
  tmp_image.create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  tmp_image = this->get_tile_data(Resolution::FULL); //cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);

  //(*this->p_image).create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);

  float scale_x = 1.0/8;
  float scale_y = 1.0/8;

  //float scale_x = 1.0;
  //float scale_y = 1.0;
  cv::resize(tmp_image, (*this->p_image), cv::Size(), scale_x,scale_y,CV_INTER_AREA);


  this->p_kps_3d->clear();// = new std::vector<cv::KeyPoint>();

  //int rows = this->p_image->rows;
  //int cols = this->p_image->cols;
  //ASSERT((rows % SIFT_D1_SHIFT_3D) == 0);
  //ASSERT((cols % SIFT_D2_SHIFT_3D) == 0);
  cv::Ptr<cv::Feature2D> p_sift;
  std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
  cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];
  int n_sub_images;


  if (true || !recomputation) {
  // NOTE(TFK): I need to check these parameters against the prefix_ cached ones.
  p_sift = new cv::xfeatures2d::SIFT_Impl(
            1001,  // num_features --- unsupported.
            6,  // number of octaves
            //24,  // number of octaves
            CONTRAST_THRESH_3D,  // contrast threshold.
            EDGE_THRESH_3D,  // edge threshold.
            1.6*2);  // sigma.
 } else {

  p_sift = new cv::xfeatures2d::SIFT_Impl(
            16,  // num_features --- unsupported.
            6,  // number of octaves
            CONTRAST_THRESH_3D,  // contrast threshold.
            EDGE_THRESH_3D,  // edge threshold.
            1.6);  // sigma.

 }

    //int max_rows = rows / SIFT_D1_SHIFT_3D;
    //int max_cols = cols / SIFT_D2_SHIFT_3D;
    int max_rows = 1;
    int max_cols = 1;
    n_sub_images = max_rows * max_cols;
        cv::Mat sub_im_mask = cv::Mat::ones(0,0,
            CV_8UC1);
        int sub_im_id = 0;
        // Detect the SIFT features within the subimage.
        //fasttime_t tstart = gettime();
        p_sift->detectAndCompute((*this->p_image), sub_im_mask, v_kps[sub_im_id],
            m_kps_desc[sub_im_id], false);

        //fasttime_t tend = gettime();
        //totalTime += tdiff(tstart, tend);
    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    int point_count_3d = 0;
    for (int _i = 0; _i < n_sub_images; _i++) {
        for (int _j = 0; _j < v_kps[_i].size(); _j++) {
            v_kps[_i][_j].pt.x /= scale_x;
            v_kps[_i][_j].pt.y /= scale_y;
            (*this->p_kps_3d).push_back(v_kps[_i][_j]);
            point_count_3d++;
        }
    }

  //cv::Mat m_kps_desc_filtered = m_kps_desc[0].clone();
  *(this)->p_kps_desc_3d = m_kps_desc[0].clone();

  //printf("Number of 3d points is %d\n", point_count_3d);
  this->p_image->release();

}

// Just returns a mat with the full resolution tile data.
cv::Mat tfk::Tile::read_tile_image() {
  return cv::imread(this->filepath, CV_LOAD_IMAGE_GRAYSCALE);
}


cv::Mat tfk::Tile::get_tile_data_lockfree(Resolution res) {

  std::string thumbnailpath = std::string(this->filepath);
  switch(res) {
    case THUMBNAIL: {
      cv::Mat tmp = this->get_tile_data(Resolution::FULL);//cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      cv::Mat ret;
      cv::Mat dst;
      cv::resize(tmp, ret, cv::Size(), 0.1,0.1,CV_INTER_AREA);
      return ret;
      break;
    }
    case THUMBNAIL2: {
      return get_tile_data(Resolution::THUMBNAIL);
      cv::Mat src = cv::imread(thumbnailpath, CV_LOAD_IMAGE_GRAYSCALE);
      return src;
      break;
    }
    case FILEIOTEST: {
      std::vector<int> params;
      params.push_back(CV_IMWRITE_JPEG_QUALITY);
      params.push_back(70);
      std::string new_path;
      new_path = this->filepath.replace(0,5, "/home/gridsan/groups/supertech/connectomix/");
      new_path = this->filepath + "_.jpg";
      cv::Mat full_image = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      cv::imwrite(new_path, full_image, params);
      return full_image;
      break;
    }

    case FULL: {
      /*full_image_lock->lock();
      if (!has_full_image) {
        std::string new_path;
        new_path = this->filepath;
        //printf("filepath is %s\n", new_path.c_str());
        //full_image = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);
        //full_image = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);

        #ifdef NOPATHCHANGE
        cv::Mat tmp = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);
        #else 
          #ifdef HUMANTEST
          cv::Mat tmp;// = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);
          getJP2Image(new_path.c_str(), tmp);
          #else
          cv::Mat tmp = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);
          #endif
        #endif

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(10.0);
        clahe->apply(tmp,full_image);
        has_full_image = true; //uncomment for cashing
      }*/
      //cv::Mat ret = full_image.clone();
      //full_image_lock->unlock();
      return full_image;
      //return ret;
      break;
    }
    case PERCENT30: {
      percent30_lock->lock();
      if (has_percent30_image) {
        cv::Mat ret = percent30_image.clone();
        percent30_lock->unlock();
        return ret;
      }

      cv::Mat tmp = this->get_tile_data(Resolution::FULL);//cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      cv::Mat ret;
      cv::Mat dst;
      cv::resize(tmp, ret, cv::Size(), 0.3,0.3,CV_INTER_AREA);
      percent30_image = ret.clone();
      has_percent30_image = true;
      percent30_lock->unlock();
      return ret;
      break;
    }
    default: {
      printf("Error in get_tile_data, invalid resolution specified, got %d\n", res);
      exit(1);
      return cv::Mat();
    }
  }
}




cv::Mat tfk::Tile::get_tile_data(Resolution res) {

  std::string thumbnailpath = std::string(this->filepath);
  switch(res) {
    case THUMBNAIL: {
      cv::Mat tmp = this->get_tile_data(Resolution::FULL);//cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      cv::Mat ret;
      cv::Mat dst;
      cv::resize(tmp, ret, cv::Size(), 0.1,0.1,CV_INTER_AREA);
      return ret;
      break;
    }
    case THUMBNAIL2: {
      return get_tile_data(Resolution::THUMBNAIL);
      cv::Mat src = cv::imread(thumbnailpath, CV_LOAD_IMAGE_GRAYSCALE);
      return src;
      break;
    }
    case FILEIOTEST: {
      std::vector<int> params;
      params.push_back(CV_IMWRITE_JPEG_QUALITY);
      params.push_back(70);
      std::string new_path;
      new_path = this->filepath.replace(0,5, "/home/gridsan/groups/supertech/connectomix/");
      new_path = this->filepath + "_.jpg";
      cv::Mat full_image = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      cv::imwrite(new_path, full_image, params);
      return full_image;
      break;
    }

    case FULL: {
      full_image_lock->lock();
      if (!has_full_image) {
        std::string new_path;
        new_path = this->filepath;
        //printf("filepath is %s\n", new_path.c_str());
        //full_image = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);
        //full_image = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);

        #ifdef NOPATHCHANGE
        cv::Mat tmp = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);
        #else 
          //#ifdef HUMANTEST
          //cv::Mat tmp;// = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);
          //getJP2Image(new_path.c_str(), tmp);
          //#else
          cv::Mat tmp = cv::imread(new_path, CV_LOAD_IMAGE_GRAYSCALE);
          //#endif
        #endif

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(10.0);
        clahe->apply(tmp,full_image);
        has_full_image = true; //uncomment for cashing
      }
      cv::Mat ret = full_image.clone();
      full_image_lock->unlock();
      return ret;
      break;
    }
    case PERCENT30: {
      percent30_lock->lock();
      if (has_percent30_image) {
        cv::Mat ret = percent30_image.clone();
        percent30_lock->unlock();
        return ret;
      }

      cv::Mat tmp = this->get_tile_data(Resolution::FULL);//cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
      cv::Mat ret;
      cv::Mat dst;
      cv::resize(tmp, ret, cv::Size(), 0.3,0.3,CV_INTER_AREA);
      percent30_image = ret.clone();
      has_percent30_image = true;
      percent30_lock->unlock();
      return ret;
      break;
    }
    default: {
      printf("Error in get_tile_data, invalid resolution specified, got %d\n", res);
      exit(1);
      return cv::Mat();
    }
  }
}

void tfk::Tile::compute_sift_keypoints2d_params(tfk::params params,
    std::vector<cv::KeyPoint>& local_keypoints, cv::Mat& local_desc) {
    compute_sift_keypoints2d_params(params, local_keypoints, local_desc, this);
}


void tfk::Tile::compute_sift_keypoints2d_params_cache(tfk::params params, Tile* other_tile) {
  //printf("computing sift keypoints 2d with params\n");

  #ifdef FSJTRAINING
  cv::Mat _tmp_image = this->get_tile_data(FULL);
  #else
  cv::Mat& _tmp_image = this->full_image;// this->get_tile_data_lockfree(FULL);
  #endif




  if (!this->overlaps_with_threshold(other_tile, 50)) {
    printf("skippin because we don't overlap\n");
    return;
  }

  std::vector<cv::KeyPoint>& local_keypoints = keypoint2d_cache[other_tile].first;
  cv::Mat& local_desc = keypoint2d_cache[other_tile].second;


  //_tmp_image.create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  //_tmp_image = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
  int buffer_slack = 50;
  cv::Mat local_p_image;

  int my_x_start = this->x_start;
  int my_x_finish = this->x_finish;

  int other_x_start = other_tile->x_start - buffer_slack;
  int other_x_finish = other_tile->x_finish + buffer_slack;

  while (my_x_start < other_x_start) {
    my_x_start += 1;
  }
  while (my_x_finish > other_x_finish) {
    my_x_finish -= 1;
  }

  int my_y_start = this->y_start;
  int my_y_finish = this->y_finish;

  int other_y_start = other_tile->y_start - buffer_slack;
  int other_y_finish = other_tile->y_finish + buffer_slack;

  while (my_y_start < other_y_start) {
    my_y_start += 1;
  }
  while (my_y_finish > other_y_finish) {
    my_y_finish -= 1;
  }

  int new_x_start = my_x_start - ((int)this->x_start);
  int new_x_finish = ((int)this->x_finish) - my_x_finish;

  int new_y_start = my_y_start - ((int)this->y_start);
  int new_y_finish = ((int)this->y_finish) - my_y_finish;

  if (new_x_start < 0 || new_y_start < 0) {
    printf("The starts are less than zero!\n");
    return;
  }
  if (_tmp_image.cols-new_x_finish < 0 || _tmp_image.rows - new_y_finish < 0) {
    printf("The ends are less than zero!\n");
    return;
  }
  if (_tmp_image.cols-new_x_finish > _tmp_image.cols || _tmp_image.rows - new_y_finish > _tmp_image.rows) {
    printf("The ends are greater than dims!\n");
    return;
  }


  float scale_x = params.scale_x;
  float scale_y = params.scale_y;


  if (_tmp_image.cols - new_x_finish-new_x_start <= 1.0/scale_x + 1.0 ||
      _tmp_image.rows - new_y_finish-new_y_start <= 1.0/scale_y + 1.0) {
    printf("skipping tile pair due to insufficient overlap. %d, %d\n", _tmp_image.cols - new_x_finish-new_x_start, _tmp_image.rows - new_y_finish-new_y_start);
    printf("skipping tile pair due to insufficient overlap info: %d - %d - %d\n", _tmp_image.cols, new_x_finish, new_x_start);
    printf("this offsets %f %f, other offsets %f %f\n", this->offset_x, this->offset_y, other_tile->offset_x, other_tile->offset_y);
    printf("this starts %f %f, other starts %f %f\n", this->x_start, this->y_start, other_tile->x_start, other_tile->y_start);
    this->overlaps_with_threshold(other_tile, 51);
    return;
  }

  if (this != other_tile) {
    cv::Mat tmp_image = _tmp_image(cv::Rect(new_x_start, new_y_start,
                                            _tmp_image.cols - new_x_finish-new_x_start,
                                            _tmp_image.rows - new_y_finish-new_y_start));
    cv::resize(tmp_image, local_p_image, cv::Size(), scale_x,scale_y,CV_INTER_AREA);
  } else {
    cv::resize(_tmp_image, local_p_image, cv::Size(), scale_x,scale_y,CV_INTER_AREA);
  }
  //_tmp_image.release();

  int rows = local_p_image.rows;
  int cols = local_p_image.cols;

  cv::Ptr<cv::Feature2D> p_sift;

  std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
  cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];

  int n_sub_images = 1;
  if ((this->tile_id > MFOV_BOUNDARY_THRESH)) {
    p_sift = new cv::xfeatures2d::SIFT_Impl(
            params.num_features,  // num_features --- unsupported.
            params.num_octaves,  // number of octaves
            params.contrast_threshold,  // contrast threshold.
            params.edge_threshold,  // edge threshold.
            params.sigma);  // sigma.

    // THEN: This tile is on the boundary, we need to compute SIFT features
    // on the entire section.
    //int max_rows = rows / SIFT_D1_SHIFT;
    //int max_cols = cols / SIFT_D2_SHIFT;
    int max_rows = 1;
    int max_cols = 1;
    n_sub_images = max_rows * max_cols;

    //cilk_for (int cur_d1 = 0; cur_d1 < rows; cur_d1 += SIFT_D1_SHIFT) {
    //  cilk_for (int cur_d2 = 0; cur_d2 < cols; cur_d2 += SIFT_D2_SHIFT) {
    {
      int cur_d1 = 0;
      int cur_d2 = 0;
        //printf("cur_d2 is %d cur_d1 is %d\n", cur_d2, cur_d1);
        // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
        cv::Mat sub_im = local_p_image(cv::Rect(cur_d2, cur_d1,
            cols, rows));

        // Mask for subimage
        cv::Mat sum_im_mask = cv::Mat::ones(rows, cols,
            CV_8UC1);

        // Compute a subimage ID, refering to a tile within larger
        //   2d image.
        //int cur_d1_id = 0;//cur_d1 / SIFT_D1_SHIFT;
        //int cur_d2_id = 0;//cur_d2 / SIFT_D2_SHIFT;
        int sub_im_id = 0;//cur_d1_id * max_cols + cur_d2_id;

        // Detect the SIFT features within the subimage.
        //fasttime_t tstart = gettime();
        p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
            m_kps_desc[sub_im_id]);

        //fasttime_t tend = gettime();
        //totalTime += tdiff(tstart, tend);

        for (size_t i = 0; i < v_kps[sub_im_id].size(); i++) {
          v_kps[sub_im_id][i].pt.x += cur_d2;
          v_kps[sub_im_id][i].pt.y += cur_d1;
        }
      }
    //}
    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    for (int i = 0; i < n_sub_images; i++) {
        for (int j = 0; j < v_kps[i].size(); j++) {
            //v_kps[i][j].pt.x += 0.5;
            //v_kps[i][j].pt.y += 0.5;
            v_kps[i][j].pt.x /= scale_x;
            v_kps[i][j].pt.y /= scale_y;
            v_kps[i][j].pt.x += new_x_start;
            v_kps[i][j].pt.y += new_y_start;
            local_keypoints.push_back(v_kps[i][j]);
        }
    }

  } else {
    printf("Assert false becasue this is unsupported code path.\n");
    assert(false);
  }

  cv::vconcat(m_kps_desc, n_sub_images, local_desc);
  local_p_image.release();
}





void tfk::Tile::compute_sift_keypoints2d_params(tfk::params params,
std::vector<cv::KeyPoint>& local_keypoints, cv::Mat& local_desc, Tile* other_tile) {
  //printf("computing sift keypoints 2d with params\n");


  #ifdef FSJTRAINING
  cv::Mat _tmp_image = this->get_tile_data(FULL);
  #else
  cv::Mat& _tmp_image = this->full_image;// this->get_tile_data_lockfree(FULL);
  #endif


  if (!this->overlaps_with_threshold(other_tile, 50)) {
    printf("skippin because we don't overlap\n");
    return;
  }


  //if (keypoint2d_cache.find(other_tile) != keypoint2d_cache.end()) {
  //  local_keypoints = keypoint2d_cache[other_tile].first;
  //  local_desc = keypoint2d_cache[other_tile].second;


  //  keypoint2d_cache.erase(other_tile);

  //  return;
  //}

  //_tmp_image.create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  //_tmp_image = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);
  int buffer_slack = 50;
  cv::Mat local_p_image;

  int my_x_start = this->x_start;
  int my_x_finish = this->x_finish;

  int other_x_start = other_tile->x_start - buffer_slack;
  int other_x_finish = other_tile->x_finish + buffer_slack;

  while (my_x_start < other_x_start) {
    my_x_start += 1;
  }
  while (my_x_finish > other_x_finish) {
    my_x_finish -= 1;
  }

  int my_y_start = this->y_start;
  int my_y_finish = this->y_finish;

  int other_y_start = other_tile->y_start - buffer_slack;
  int other_y_finish = other_tile->y_finish + buffer_slack;

  while (my_y_start < other_y_start) {
    my_y_start += 1;
  }
  while (my_y_finish > other_y_finish) {
    my_y_finish -= 1;
  }

  int new_x_start = my_x_start - ((int)this->x_start);
  int new_x_finish = ((int)this->x_finish) - my_x_finish;

  int new_y_start = my_y_start - ((int)this->y_start);
  int new_y_finish = ((int)this->y_finish) - my_y_finish;

  if (new_x_start < 0 || new_y_start < 0) {
    printf("The starts are less than zero!\n");
    return;
  }
  if (_tmp_image.cols-new_x_finish < 0 || _tmp_image.rows - new_y_finish < 0) {
    printf("The ends are less than zero!\n");
    return;
  }
  if (_tmp_image.cols-new_x_finish > _tmp_image.cols || _tmp_image.rows - new_y_finish > _tmp_image.rows) {
    printf("The ends are greater than dims!\n");
    return;
  }


  float scale_x = params.scale_x;
  float scale_y = params.scale_y;


  if (_tmp_image.cols - new_x_finish-new_x_start <= 1.0/scale_x + 1.0 ||
      _tmp_image.rows - new_y_finish-new_y_start <= 1.0/scale_y + 1.0) {
    printf("skipping tile pair due to insufficient overlap. %d, %d\n", _tmp_image.cols - new_x_finish-new_x_start, _tmp_image.rows - new_y_finish-new_y_start);
    printf("skipping tile pair due to insufficient overlap info: %d - %d - %d\n", _tmp_image.cols, new_x_finish, new_x_start);
    printf("this offsets %f %f, other offsets %f %f\n", this->offset_x, this->offset_y, other_tile->offset_x, other_tile->offset_y);
    printf("this starts %f %f, other starts %f %f\n", this->x_start, this->y_start, other_tile->x_start, other_tile->y_start);
    this->overlaps_with_threshold(other_tile, 51);
    return;
  }

  if (this != other_tile) {
    cv::Mat tmp_image = _tmp_image(cv::Rect(new_x_start, new_y_start,
                                            _tmp_image.cols - new_x_finish-new_x_start,
                                            _tmp_image.rows - new_y_finish-new_y_start));
    cv::resize(tmp_image, local_p_image, cv::Size(), scale_x,scale_y,CV_INTER_AREA);
  } else {
    cv::resize(_tmp_image, local_p_image, cv::Size(), scale_x,scale_y,CV_INTER_AREA);
  }
  //_tmp_image.release();

  int rows = local_p_image.rows;
  int cols = local_p_image.cols;

  cv::Ptr<cv::Feature2D> p_sift;

  std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
  cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];

  int n_sub_images = 1;
  if ((this->tile_id > MFOV_BOUNDARY_THRESH)) {
    p_sift = new cv::xfeatures2d::SIFT_Impl(
            params.num_features,  // num_features --- unsupported.
            params.num_octaves,  // number of octaves
            params.contrast_threshold,  // contrast threshold.
            params.edge_threshold,  // edge threshold.
            params.sigma);  // sigma.

    // THEN: This tile is on the boundary, we need to compute SIFT features
    // on the entire section.
    //int max_rows = rows / SIFT_D1_SHIFT;
    //int max_cols = cols / SIFT_D2_SHIFT;
    int max_rows = 1;
    int max_cols = 1;
    n_sub_images = max_rows * max_cols;

    //cilk_for (int cur_d1 = 0; cur_d1 < rows; cur_d1 += SIFT_D1_SHIFT) {
    //  cilk_for (int cur_d2 = 0; cur_d2 < cols; cur_d2 += SIFT_D2_SHIFT) {
    {
      int cur_d1 = 0;
      int cur_d2 = 0;
        //printf("cur_d2 is %d cur_d1 is %d\n", cur_d2, cur_d1);
        // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
        cv::Mat sub_im = local_p_image(cv::Rect(cur_d2, cur_d1,
            cols, rows));

        // Mask for subimage
        cv::Mat sum_im_mask = cv::Mat::ones(rows, cols,
            CV_8UC1);

        // Compute a subimage ID, refering to a tile within larger
        //   2d image.
        //int cur_d1_id = 0;//cur_d1 / SIFT_D1_SHIFT;
        //int cur_d2_id = 0;//cur_d2 / SIFT_D2_SHIFT;
        int sub_im_id = 0;//cur_d1_id * max_cols + cur_d2_id;

        // Detect the SIFT features within the subimage.
        //fasttime_t tstart = gettime();
        p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
            m_kps_desc[sub_im_id]);

        //fasttime_t tend = gettime();
        //totalTime += tdiff(tstart, tend);

        for (size_t i = 0; i < v_kps[sub_im_id].size(); i++) {
          v_kps[sub_im_id][i].pt.x += cur_d2;
          v_kps[sub_im_id][i].pt.y += cur_d1;
        }
      }
    //}
    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    for (int i = 0; i < n_sub_images; i++) {
        for (int j = 0; j < v_kps[i].size(); j++) {
            //v_kps[i][j].pt.x += 0.5;
            //v_kps[i][j].pt.y += 0.5;
            v_kps[i][j].pt.x /= scale_x;
            v_kps[i][j].pt.y /= scale_y;
            v_kps[i][j].pt.x += new_x_start;
            v_kps[i][j].pt.y += new_y_start;
            local_keypoints.push_back(v_kps[i][j]);
        }
    }

  } else {
    printf("Assert false becasue this is unsupported code path.\n");
    assert(false);
  }

  cv::vconcat(m_kps_desc, n_sub_images, local_desc);
  local_p_image.release();
}


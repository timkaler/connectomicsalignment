/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <malloc.h>
#include <sys/time.h>
#include <functional>
#include "common.h"

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// TYPES
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// GLOBALS
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////





void init_tile(
    tile_data_t *p_tile,
    int in_section_id,
    int in_mfov_id,
    int in_index,
    int in_x_start,
    int in_x_finish,
    int in_y_start,
    int in_y_finish,
    char *in_filepath) {
    p_tile->section_id = in_section_id;
    p_tile->mfov_id = in_mfov_id;
    p_tile->index = in_index;
    p_tile->x_start = in_x_start;
    p_tile->x_finish = in_x_finish;
    p_tile->y_start = in_y_start;
    p_tile->y_finish = in_y_finish;
    strcpy(p_tile->filepath, in_filepath);
    p_tile->p_image = new cv::Mat();
    p_tile->p_kps = new std::vector<cv::KeyPoint>();
    p_tile->p_kps_desc = new cv::Mat();
}

void init_section(section_data_t *p_sec_data, int sec_id) {
    p_sec_data->section_id = sec_id;
    p_sec_data->n_tiles = 0;
    p_sec_data->out_d1 = OUT_D1_SIZE;
    p_sec_data->out_d2 = OUT_D2_SIZE;

   /*
    for (int i = 0; i < MAX_TILES; i++) {
        p_sec_data->p_transforms[i] = new cv::Mat();
        p_sec_data->p_warps[i] = new cv::Mat();
    }

    p_sec_data->p_warp_order = new std::vector<int>();
    p_sec_data->p_out = new cv::Mat();
   */ 
}

void init_align(align_data_t *p_align_data) {
    p_align_data->input_filepath = NULL;
    p_align_data->work_dirpath = NULL;
    for (int i = 0; i < MAX_SECTIONS; i++) {
        //p_align_data->p_section_transforms[i] = new cv::Mat();
        init_section(&(p_align_data->sec_data[i]), i);
    }
}

void start_timer(struct timeval *p_timer) {
    gettimeofday(p_timer, NULL);
}

void stop_timer(struct timeval *p_timer, const char *msg) {
    struct timeval timer2;
    double elapsed_time = 0;

    gettimeofday(&timer2, NULL);

    elapsed_time = (timer2.tv_sec - p_timer->tv_sec) * 1000000.0; // sec to microsecs
    elapsed_time += (timer2.tv_usec - p_timer->tv_usec); // microsecs

    printf("%s: %f [microsec]\n", msg, elapsed_time);

}

std::string CVMatType2Str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void log_kps(tile_data_t *p_tile_data) {
    char filepath[MAX_FILEPATH];    
    cv::Mat outImage;
        
    drawKeypoints(
        *(p_tile_data->p_image),
        *(p_tile_data->p_kps), 
        outImage, 
        cv::Scalar::all(-1), 
        cv::DrawMatchesFlags::DEFAULT);
    printf("logging image to path %s\n", LOG_DIR); 
    sprintf(filepath, "%s/kps_tile_%.4d_%.4d_%.4d.tif", 
        LOG_DIR, 
        p_tile_data->section_id,
        p_tile_data->mfov_id,
        p_tile_data->index);
    
    cv::imwrite(filepath, outImage);
}

void log_kps_matches(
    align_data_t *p_align_data,
    tile_data_t *p_tile_data_src,
    tile_data_t *p_tile_data_dst,
    std::vector< cv::DMatch > *p_matches) {
    
    char filepath[MAX_FILEPATH];    
    cv::Mat outImage;
    
    drawMatches(
        *(p_tile_data_src->p_image),
        *(p_tile_data_src->p_kps),
        *(p_tile_data_dst->p_image),
        *(p_tile_data_dst->p_kps),
        *p_matches,
        outImage);
        
    sprintf(filepath, "%s/kps_matches_tile_%.4d_%.4d_%.4d.tif", 
        LOG_DIR, 
        p_tile_data_src->section_id + p_align_data->base_section,
        p_tile_data_src->mfov_id,
        p_tile_data_src->index);
    
    cv::imwrite(filepath, outImage);
}

cv::Mat draw_pts_matches(
    cv::Mat im_src,
    cv::Mat im_dst,
    std::vector<cv::Point2f> *p_src_pts,
    std::vector<cv::Point2f> *p_dst_pts) {
    
    ASSERT(im_src.rows == im_dst.rows);
    ASSERT(im_src.cols == im_dst.cols);
    
    int rows = im_src.rows;
    int cols = im_src.cols;
    
    int out_rows = rows;
    int out_cols = cols * 2;
    
    cv::Mat out = cv::Mat::zeros(out_rows, out_cols, CV_8UC3);
    
    int d1_start = 0;
    int d2_start = 0;
    int d1_len = rows;
    int d2_len = cols;
    cv::Mat tmp = out(cv::Rect(d2_start, d1_start, d2_len, d1_len));
    cv::Mat tmp2;
    
    cv::cvtColor(im_src, tmp2, CV_GRAY2RGB, 3);
    tmp2.copyTo(tmp);
    
    d1_start = 0;
    d2_start = cols;
    d1_len = rows;
    d2_len = cols;
    tmp = out(cv::Rect(d2_start, d1_start, d2_len, d1_len));
    
    cv::cvtColor(im_dst, tmp2, CV_GRAY2RGB, 3);
    tmp2.copyTo(tmp);
    
    ASSERT(p_dst_pts->size() == p_src_pts->size());
    
    cv::Point2f src_pt;
    cv::Point2f dst_pt;
    for (size_t i = 0; i < p_src_pts->size(); i++) {
        src_pt = (*p_src_pts)[i];
        dst_pt = (*p_dst_pts)[i];
        dst_pt.x += cols;
        
        cv::line(out, src_pt, dst_pt, cv::Scalar( 110, 220, 0 ), 1, 8);
    }
    
    return out;
}

void log_pts_matches(
    align_data_t *p_align_data,
    tile_data_t *p_tile_data_src,
    tile_data_t *p_tile_data_dst,
    std::vector<cv::Point2f> *p_src_pts,
    std::vector<cv::Point2f> *p_dst_pts) {
    
    char filepath[MAX_FILEPATH];    
    cv::Mat outImage;
    
    outImage = draw_pts_matches(
        *(p_tile_data_src->p_image),
        *(p_tile_data_dst->p_image),
        p_src_pts,
        p_dst_pts);
        
    sprintf(filepath, "%s/pts_matches_tile_%.4d_%.4d_%.4d.tif", 
        LOG_DIR, 
        p_tile_data_src->section_id + p_align_data->base_section,
        p_tile_data_src->mfov_id,
        p_tile_data_src->index);
    
    cv::imwrite(filepath, outImage);
}


bool is_tiles_overlap_slack(tile_data_t *p_tile_data_1, tile_data_t *p_tile_data_2, double slack) {

    TRACE_3("is_tiles_overlap: start\n");
    TRACE_3("  -- tile_1: %d\n", p_tile_data_1->index);
    TRACE_3("  -- tile_2: %d\n", p_tile_data_2->index);
    
    int x1_start = p_tile_data_1->x_start - slack;
    int x1_finish = p_tile_data_1->x_finish + slack;
    int y1_start = p_tile_data_1->y_start - slack;
    int y1_finish = p_tile_data_1->y_finish + slack;
    
    int x2_start = p_tile_data_2->x_start;
    int x2_finish = p_tile_data_2->x_finish;
    int y2_start = p_tile_data_2->y_start;
    int y2_finish = p_tile_data_2->y_finish;
    
    bool res = false;
    
    if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
        (y1_start < y2_finish) && (y1_finish > y2_start)) {
        res = true;
    }
    
    TRACE_3("is_tiles_overlap: finish\n");
    
    return res;
    
}
bool is_tiles_overlap(tile_data_t *p_tile_data_1, tile_data_t *p_tile_data_2) {

    TRACE_3("is_tiles_overlap: start\n");
    TRACE_3("  -- tile_1: %d\n", p_tile_data_1->index);
    TRACE_3("  -- tile_2: %d\n", p_tile_data_2->index);
    
    int x1_start = p_tile_data_1->x_start;
    int x1_finish = p_tile_data_1->x_finish;
    int y1_start = p_tile_data_1->y_start;
    int y1_finish = p_tile_data_1->y_finish;
    
    int x2_start = p_tile_data_2->x_start;
    int x2_finish = p_tile_data_2->x_finish;
    int y2_start = p_tile_data_2->y_start;
    int y2_finish = p_tile_data_2->y_finish;
    
    bool res = false;
    
    if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
        (y1_start < y2_finish) && (y1_finish > y2_start)) {
        res = true;
    }
    
    TRACE_3("is_tiles_overlap: finish\n");
    
    return res;
    
}

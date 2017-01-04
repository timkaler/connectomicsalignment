/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef COMMON_H
#define COMMON_H 1

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <mutex>

#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/types_c.h>
#include <hdf5.h>
#include <opencv2/hdf/hdf5.hpp>
//#define FILE "dset.h5"


/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES 
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES 
/////////////////////////////////////////////////////////////////////////////////////////
#define likely(x) __builtin_expect ((x), 1)
#define unlikely(x) __builtin_expect ((x), 0)

#define prefetch_read(addr) __builtin_prefetch(addr, 0)

//#define ENABLE_TRACE_TIMER
//#define ENABLE_TRACE_TIMER_2
//
//#define ENABLE_TRACE_1
//#define ENABLE_TRACE_2
//#define ENABLE_TRACE_3
//#define ENABLE_LOG_KPS

#define MODE_COMPUTE_KPS_AND_MATCH (1)
#define MODE_COMPUTE_TRANSFORMS (2)
#define MODE_COMPUTE_WARPS (3)

#define LOG_DIR "log"

#define MAX_FILEPATH (512)
#define MAX_INPUT_BUF (2000)

//#define MAX_TILES (1024)
#define MAX_TILES (1024)
//#define MAX_SECTIONS (256)
#define MAX_SECTIONS (1024)

#define OVERLAP_2D (200)

#define MFOV_BOUNDARY_THRESH 37 // any section with id > 37 is on boundary.



#define LOWE_RATIO (0.65)
// TB: I don't know why the Python calls this ratio "rod".
#define ROD (0.92)
#define MAX_KPS_DIST (3000)

#define SIFT_MAX_SUB_IMAGES (1024)
//#define SIFT_D1_SHIFT (4096) //(256)
//#define SIFT_D2_SHIFT (4096) //(256)

//#define SIFT_D1_SHIFT (1024)
//#define SIFT_D2_SHIFT (1024)
#define SIFT_D1_SHIFT (681)
#define SIFT_D2_SHIFT (782)

#define OUT_D1_SIZE (20000)
#define OUT_D2_SIZE (20000)

//#define OUT_D1_SIZE (2048 * 2)
//#define OUT_D2_SIZE (2048 * 2)

//#define OUT_SHIFT_D1 (OUT_D1_SIZE/2)
//#define OUT_SHIFT_D2 (OUT_D2_SIZE/2)

#define BIN_ANGLE (20)

#define OUT_SHIFT_D1 (1000)
#define OUT_SHIFT_D2 (1000)

#define CONV_BASE_D1 (500)
#define CONV_BASE_D2 (500)

#define CONV_SHIFT_D1 (100)
#define CONV_SHIFT_D2 (100)

/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES - ASSERT AND DEBUG
/////////////////////////////////////////////////////////////////////////////////////////
#define TRACE(fmt, ...) \
    fprintf(stderr, "%s:%d:%s(): " fmt, \
            __FILE__, __LINE__, __func__, \
            ##__VA_ARGS__);

#define ABORT(fmt, ...) \
    TRACE(fmt, ##__VA_ARGS__); \
    abort();

#define ASSERT(cond) \
    if (unlikely(!(cond))) { \
        printf ("\n-----------------------------------------------\n"); \
        printf ("\nAssertion failure: %s:%d '%s'\n", __FILE__, __LINE__, #cond); \
        abort(); \
    }

#define ASSERT_MSG(cond, fmt, ...) \
    if (unlikely(!(cond))) { \
        printf ("\n-----------------------------------------------\n"); \
        printf ("\nAssertion failure: %s:%d '%s'\n", __FILE__, __LINE__, #cond); \
        TRACE(fmt, ##__VA_ARGS__); \
        abort(); \
    }

#ifdef ENABLE_TRACE_TIMER
#define TIMER_VAR(timer_name) struct timeval timer_name;
#define START_TIMER(timer) start_timer((timer))
#define STOP_TIMER(timer, msg) stop_timer((timer), msg)
#else
#define TIMER_VAR(timer_name)
#define START_TIMER(timer)
#define STOP_TIMER(timer, msg)
#endif

#ifdef ENABLE_TRACE_TIMER_2
#define TIMER_VAR_2(timer_name) struct timeval timer_name
#define START_TIMER_2(timer) start_timer((timer))
#define STOP_TIMER_2(timer, msg) stop_timer((timer), msg)
#else
#define TIMER_VAR_2(timer_name)
#define START_TIMER_2(timer)
#define STOP_TIMER_2(timer, msg)
#endif

#ifdef ENABLE_TRACE_1
#define TRACE_1(fmt, ...) TRACE(fmt, ##__VA_ARGS__)
#else
#define TRACE_1(fmt, ...)
#endif

#ifdef ENABLE_TRACE_2
#define TRACE_2(fmt, ...) TRACE(fmt, ##__VA_ARGS__)
#else
#define TRACE_2(fmt, ...)
#endif

#ifdef ENABLE_TRACE_3
#define TRACE_3(fmt, ...) TRACE(fmt, ##__VA_ARGS__)
#else
#define TRACE_3(fmt, ...)
#endif

#ifdef ENABLE_TRACE_4
#define TRACE_4(fmt, ...) TRACE(fmt, ##__VA_ARGS__)
#else
#define TRACE_4(fmt, ...)
#endif

#ifdef ENABLE_LOG_KPS
#define LOG_KPS(p_tile_data) log_kps(p_tile_data)
#else
#define LOG_KPS(p_tile_data)
#endif

// haoran's favorite ugly macros
#define rep(i,l,r) for (int i=(l); i<=(r); i++)
#define repd(i,r,l) for (int i=(r); i>=(l); i--)
#define rept(i,c) for (__typeof((c).begin()) i=(c).begin(); i!=(c).end(); i++)


/////////////////////////////////////////////////////////////////////////////////////////
// TYPES
/////////////////////////////////////////////////////////////////////////////////////////
typedef struct _tile_data {
    int section_id;
    int tile_id;
    int mfov_id;
    int index;
    char filepath[MAX_FILEPATH];
    cv::Mat *p_image;

    int x_start;
    int x_finish;
    int y_start;
    int y_finish;

    std::vector<cv::KeyPoint> *p_kps;
    cv::Mat *p_kps_desc;
        
} tile_data_t;

typedef struct _section_data {
    int section_id; 
    
    int n_tiles;
    tile_data_t tiles[MAX_TILES];
    
    int out_d1;
    int out_d2;
    cv::Mat *p_transforms[MAX_TILES];
    cv::Mat *p_warps[MAX_TILES];
    std::vector<int> *p_warp_order;
    cv::Mat *p_out;
    
    
} section_data_t;

typedef struct _align_data {
    
    int mode;
    
    char *input_filepath;
    char *work_dirpath;
    char *output_dirpath;
    
    int base_section;
    int n_sections;
    section_data_t sec_data[MAX_SECTIONS];
    
    cv::Mat *p_section_transforms[MAX_SECTIONS];
    
    
} align_data_t;

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////
void start_timer(struct timeval *p_timer);
void stop_timer(struct timeval *p_timer, const char *msg);

void init_tile(
    tile_data_t *p_tile,
    int in_section_id,
    int in_mfov_id,
    int in_index,
    int in_x_start,
    int in_x_finish,
    int in_y_start,
    int in_y_finish,
    char *in_filepath);

void init_section(section_data_t *p_sec_data, int sec_id);
void init_align(align_data_t *p_align_data);

std::string CVMatType2Str(int type);
void log_kps(tile_data_t *p_tile_data);
void log_kps_matches(
    align_data_t *p_align_data,
    tile_data_t *p_tile_data_src,
    tile_data_t *p_tile_data_dst,
    std::vector< cv::DMatch > *p_matches);
  
void log_pts_matches(
    align_data_t *p_align_data,
    tile_data_t *p_tile_data_src,
    tile_data_t *p_tile_data_dst,
    std::vector<cv::Point2f> *p_src_pts,
    std::vector<cv::Point2f> *p_dst_pts);

bool is_tiles_overlap(tile_data_t *p_tile_data_1, tile_data_t *p_tile_data_2);

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL MACROS
/////////////////////////////////////////////////////////////////////////////////////////

#endif // COMMON_H

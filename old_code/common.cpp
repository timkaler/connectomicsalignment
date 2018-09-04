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

void start_timer(struct timeval *p_timer) {
    gettimeofday(p_timer, NULL);
}

void stop_timer(struct timeval *p_timer, const char *msg) {
    struct timeval timer2;
    double elapsed_time = 0;

    gettimeofday(&timer2, NULL);

    elapsed_time = (timer2.tv_sec - p_timer->tv_sec) * 1000000.0; // sec to microsecs
    elapsed_time += (timer2.tv_usec - p_timer->tv_usec); // microsecs

    printf("%s: %f [sec]\n", msg, elapsed_time/1000000.0);

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

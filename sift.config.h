#ifndef __SIFT_CONFIG_H__
#define __SIFT_CONFIG_H__

#define SIFT_DATATYPE_CHAR 0	//not supported yet
#define SIFT_DATATYPE_SHORT 1
#define SIFT_DATATYPE_FLOAT 2

// choice of SIFT data type
#define SIFT_DATATYPE SIFT_DATATYPE_SHORT

// whether SIFT should use batching
// SIFT code will vectorize when batching
#define SIFT_USE_BATCHING 0

// comment out the line below to use default GaussianBlur instead of BoxBlur
#define USE_BOXBLUR_GAUSSIANPYRAMID

#if SIFT_DATATYPE == SIFT_DATATYPE_CHAR
#error Char data type not supported yet
#endif

#if SIFT_DATATYPE == SIFT_DATATYPE_FLOAT
#if SIFT_USE_BATCHING == 1
#error Batching with Float data type not supported yet
#endif
#endif

#if SIFT_USE_BATCHING == 1
	#if SIFT_DATATYPE == SIFT_DATATYPE_CHAR
	#define SIFT_BATCH_SIZE 32
	#elif SIFT_DATATYPE == SIFT_DATATYPE_SHORT
	#define SIFT_BATCH_SIZE 16
	#elif SIFT_DATATYPE == SIFT_DATATYPE_FLOAT
	#define SIFT_BATCH_SIZE 8
	#endif
#else
	#define SIFT_BATCH_SIZE 1
#endif

#if SIFT_DATATYPE == SIFT_DATATYPE_CHAR
typedef char sift_wt_elem;
static const int SIFT_FIXPT_SCALE = 1;
#elif SIFT_DATATYPE == SIFT_DATATYPE_SHORT
typedef short sift_wt_elem;
static const int SIFT_FIXPT_SCALE = 48;
#elif SIFT_DATATYPE == SIFT_DATATYPE_FLOAT
typedef float sift_wt_elem;
static const int SIFT_FIXPT_SCALE = 1;
#else
#error Unsupported SIFT data type
#endif

struct sift_wt
{
	sift_wt_elem chan[SIFT_BATCH_SIZE];
};

#endif

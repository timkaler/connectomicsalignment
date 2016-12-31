/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include "align.h"

#include "fasttime.h"

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

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
void parse_args(
    align_data_t *p_align_data, 
    int argc, 
    char **argv) {
    
    ASSERT_MSG(argc == 7, "Usage: %s [mode] [base_section] [num_sections] [input_filepath] [work_dir] [output_dir]\n", argv[0]);
    
    int idx = 1;
    
    p_align_data->mode = atoi(argv[idx]);
    idx++;
    
    ASSERT(((p_align_data->mode == MODE_COMPUTE_KPS_AND_MATCH) || 
            (p_align_data->mode == MODE_COMPUTE_TRANSFORMS) ||
            (p_align_data->mode == MODE_COMPUTE_WARPS)));
    
    p_align_data->base_section = atoi(argv[idx]);
    idx++;
    
    p_align_data->n_sections = atoi(argv[idx]);
    idx++;
    
    p_align_data->input_filepath = argv[idx];
    idx++;
    
    p_align_data->work_dirpath = argv[idx];
    idx++;
    
    p_align_data->output_dirpath = argv[idx];
    idx++;
    
    
}

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    
    cv::setNumThreads(0);
    
   // testcv();
    
   // exit(0);
    
    align_data_t *p_align_data;
    
    p_align_data = (align_data_t *)malloc(sizeof(align_data_t));
    
    init_align(p_align_data);
    
    parse_args(p_align_data, argc, argv);
#ifdef PROFILE
    ProfilerStart("profile.data");
#endif 
    align_execute(p_align_data);
#ifdef PROFILE
    ProfilerStop();
#endif
    return 0;
}

// Copyright 2016 - Supertech Research Group
//   Tim Kaler, Tao B. Schardl, Haoran Xu, Charles E. Leiserson, Alex Matveev.


////////////////////////////////////////////////////////////////////////////////
// INCLUDES
////////////////////////////////////////////////////////////////////////////////
#include <cilk/cilk.h>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

#include "./common.h"
#include "./align.h"
#include "./fasttime.h"
#include "./othersift2.cpp"

#include "AlignData.pb.h"
#include "AlignData.pb.cc"

#include "stack.hpp"

void align_execute(align_data_t *p_align_data) {
    TIMER_VAR(t_timer);
    TIMER_VAR(timer);
    //TFK_TIMER_VAR(timer_render);
    START_TIMER(&t_timer);

    tfk::Stack* stack = new tfk::Stack(p_align_data->base_section,
                                       p_align_data->n_sections,
                                       p_align_data->input_filepath,
                                       p_align_data->output_dirpath);

    stack->mode = p_align_data->mode;
    stack->output_dirpath = p_align_data->output_dirpath;
    stack->base_section = p_align_data->base_section;
    stack->n_sections = p_align_data->n_sections;
    stack->do_subvolume = p_align_data->do_subvolume;
    stack->input_filepath = p_align_data->input_filepath;
    stack->min_x = p_align_data->min_x;
    stack->min_y = p_align_data->min_y;
    stack->max_x = p_align_data->max_x;
    stack->max_y = p_align_data->max_y;
    stack->init();
    printf("Got past the init\n");
    printf("stack has sections %zu\n", stack->sections.size()); 
    stack->align_2d();
    stack->coarse_affine_align();
    stack->elastic_align();
//    stack->render(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(100000,100000)), "render", tfk::PERCENT30);
    stack->render_error(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(100000,100000)), "testrender");
    printf("Got to the end.\n");
    return;
}


// Copyright 2016 - Supertech Research Group
//   Tim Kaler, Tao B. Schardl, Haoran Xu, Charles E. Leiserson, Alex Matveev.


////////////////////////////////////////////////////////////////////////////////
// INCLUDES
////////////////////////////////////////////////////////////////////////////////

#include "./align.h"

#include <cilk/cilk.h>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

#include <iostream>
#include <cstdio>
#include <ctime>

#include "./common.h"


#include "./fasttime.h"
#include "./othersift2.cpp"

#include "./AlignData.pb.h"
#include "./AlignData.pb.cc"

#include "./stack.hpp"

#include "./render.hpp"
#include "./data.hpp"
#include "./ParamsDatabase.pb.h"

int sched_yield(void) {
for (int i=0; i< 4000; i++) _mm_pause(); usleep(1);

return 0;

}

tfk::Stack* make_stack(align_data_t* p_align_data) {
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
    stack->_bounding_box = p_align_data->bounding_box;
    stack->use_bbox_prefilter = false;
    stack->align_data = p_align_data;
    stack->init();

  {
    stack->paramdbs[MATCH_TILE_PAIR_TASK_ID]->align_data = p_align_data;
  }
  stack->align_data = p_align_data;

  return stack;
}


void align_execute(align_data_t *p_align_data) {
    tfk::Stack* stack = make_stack(p_align_data);
    printf("Got past the init\n");
    printf("stack has sections %zu\n", stack->sections.size());

    printf("starting align 2d\n");
    stack->align_2d();
    printf("Done with align 2d\n");
    stack->align_3d();
    printf("Done with align 3d\n");

    for (int i = 0; i < stack->sections.size(); i++) {
      stack->sections[i]->elastic_transform_ready = true;
    }

    //int size = 75000;
    std::clock_t start;
    double duration;

    start = std::clock();
    //int _start_x = 100000;
    //int _start_y = 100000;

    tfk::Render* render = new tfk::Render();
    auto entire_bbox = stack->sections[0]->get_bbox();
    render->render_stack(stack,entire_bbox,tfk::Resolution::THUMBNAIL,"out/stack");
    //float x1 = (entire_bbox.first.x + entire_bbox.second.x)/2 + 5000;  // -2500 + 5000;
    //float x2 = x1+5000;
    //float y1 = (entire_bbox.first.y+entire_bbox.second.y)/2+5000;  // -2500 + 5000;
    //float y2 = y1+5000;
    ////auto smaller_bbox = std::make_pair(cv::Point2f(x1, y1), cv::Point2f(x2, y2));


    //float delta_x = entire_bbox.second.x - entire_bbox.first.x;
    //float delta_y = entire_bbox.second.y - entire_bbox.first.y;
    //auto quadrant_1 = std::make_pair(cv::Point2f(entire_bbox.first.x,
    //                                             entire_bbox.first.y),
    //                                 cv::Point2f(entire_bbox.first.x + delta_x*0.3,
    //                                             entire_bbox.first.y+delta_y*0.3));


    printf("Right before render\n");
    printf("Num Sections: %zu\n", stack->sections.size());

    printf("Got to the end.\n");
    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
    std::cout << "printf: " << duration << '\n';
    return;
}

void train_fsj(align_data_t *p_align_data) {
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
    stack->_bounding_box = p_align_data->bounding_box;
    stack->use_bbox_prefilter = false;
    stack->align_data = p_align_data;
    stack->init();
  {
    stack->paramdbs[MATCH_TILE_PAIR_TASK_ID]->align_data = p_align_data;
  }

    printf("stack has sections %zu\n", stack->sections.size());
    std::vector<tfk::params> ps;
    int trials = 5000;

    stack->train_fsj(trials);
    return;
}


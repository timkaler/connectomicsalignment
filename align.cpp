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

#include <iostream>
#include <cstdio>
#include <ctime>

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

    for (int i = 0; i < stack->sections.size(); i++) {
      stack->sections[i]->elastic_transform_ready = true;
    }

    int size = 50000;
    std::clock_t start;
    double duration;

    start = std::clock();


    stack->render(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(50000 + size, 50000 + size)), "renderbefore", tfk::FULL);


    //stack->render_error(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(50000 + size, 50000 + size)), "testrender");


    //printf("Now am going to recompute the alignment\n");
    //stack->recompute_alignment();
    //printf("REcomputed the alignment, now am going to rerender with error markers.\n");
    //stack->render_error(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(50000 + size, 50000 + size)), "testrender_after");
    //stack->render(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(50000 + size, 50000 + size)), "renderafter", tfk::THUMBNAIL);
    printf("Got to the end.\n");
//<<<<<<< HEAD
//    printf("Counts of bad tiles replaced:\n");
//    for (int i = 0; i < stack->sections.size(); i++) {
//      printf("\tSection %d, %d tiles replaced\n", i, stack->sections[i]->num_tiles_replaced);
//    }
//=======
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout<<"printf: "<< duration <<'\n';
//>>>>>>> e7ca2dfba6e85351e7dcc4f891e0752c435b2524
    return;
}

void param_optimize(align_data_t *p_align_data) {

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
    std::vector<tfk::params> ps;
    

    for (int nf = 0; nf < 1; nf++) {
        int num_features = 1 << nf;
        for (int no = 6; no < 7; no++) {
            int num_octaves = no;
            for (float sigma = 1.6; sigma < 1.7; sigma+=.2) {
                //for (int r = 2; r < 3; r++) {
                    // can only run at full size
                    tfk::Resolution res = tfk::FULL;
                    /*
                    if (r==0) {
                        res = tfk::THUMBNAIL;
                    } else if (r==1) {
                        res = tfk::THUMBNAIL2;
                    } else if (r==2) {
                        res = tfk::FULL;
                    } else if (r==3) {
                        res = tfk::PERCENT30;
                    }
                    */
                      tfk::params p;
                      p.num_features = num_features;
                      p.num_octaves = num_octaves;
                      p.contrast_threshold = CONTRAST_THRESH;
                      p.edge_threshold = EDGE_THRESH_2D;
                      p.sigma = sigma;
                      p.res = res;
                      ps.push_back(p);
                //}
            }
        }
    }
    printf("testing %zu different paramter combinations\n", ps.size());
    std::vector<std::tuple<int, double, int>> current[stack->sections.size()];

    double threshold = 1.0;
    int trials = 10;
    for (int i = 0; i < stack->sections.size(); i++) {
      current[i] = stack->sections[i]->parameter_optimization(trials, threshold, ps);
      // to treat each section independently as a seperate trial
      for (int j = 0; j < ps.size(); j++) {
        printf("%d, %d, %f, %d, %d, %f, %f, %d\n",
            ps[j].num_features, ps[j].num_octaves, ps[j].sigma, 
            ps[j].res, std::get<0>(current[i][j]), threshold, std::get<1>(current[i][j]), std::get<2>(current[i][j]));
      }
    }
    // to sum up across the different sections
    /*
    for (int j = 0; j < ps.size(); j++) {
      for (int i = 1; i < stack->sections.size(); i++) {
        std::get<0>(current[0][j]) += std::get<0>(current[i][j]);
        std::get<1>(current[0][j]) += std::get<1>(current[i][j]);
        std::get<2>(current[0][j]) += std::get<2>(current[i][j]);
      }
      printf("%d, %d, %f, %d, %d, %f, %f, %d\n",
        ps[j].num_features, ps[j].num_octaves, ps[j].sigma, ps[j].res, 
        std::get<0>(current[0][j]), threshold, std::get<1>(current[0][j]), std::get<2>(current[0][j]));
    }
    */

    return;
}



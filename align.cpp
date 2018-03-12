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

namespace tfk {
  void train_match_tiles_task(Stack* stack) {
    int success = 0;
    int trials = 0;
    std::vector<int> param_adjustments(7);
    std::vector<int> param_train_deltas(7);

    for (int i = 0; i < param_adjustments.size(); i++) {
      param_adjustments[i] = 0;
      param_train_deltas[i] = 0;
    }

    cilk_for (int i = 0; i < 200; i++) {
      // pick random section.
      int section_id = rand()%stack->sections.size();
      Section* section = stack->sections[section_id];
      int tile_id = rand()%section->tiles.size();
      Tile* tile = section->tiles[tile_id];
      std::vector<Tile*> neighbors = section->get_all_close_tiles(tile);
      MatchTilesTask* task = new MatchTilesTask(tile, neighbors);
      task->compute(0.9, param_adjustments, param_train_deltas);
      bool res = task->error_check(0.9);
      if (res) {
        __sync_fetch_and_add(&success, 1);
      }
      __sync_fetch_and_add(&trials, 1);
      //trials++;
      //printf("prelim %f\n", (1.0*success)/trials);
    }
    printf("Result is %f\n", (1.0*success)/trials);

    float last_correct = (1.0*success)/trials;

    for (int j = 0; j < 100; j++) {
       success = 0;
       trials = 0;
         for (int k = 0; k < param_train_deltas.size(); k++) {
           param_train_deltas[k] = 0;
         }
       for (int k = 0; k < param_train_deltas.size(); k++) {
         int index = rand()%param_train_deltas.size();
         int sign = rand()%2 ? -1 : 1;
         param_train_deltas[index] += sign;
       }

       cilk_for (int i = 0; i < 200; i++) {
        // pick random section.
        int section_id = rand()%stack->sections.size();
        Section* section = stack->sections[section_id];
        int tile_id = rand()%section->tiles.size();
        Tile* tile = section->tiles[tile_id];
        std::vector<Tile*> neighbors = section->get_all_close_tiles(tile);
        MatchTilesTask* task = new MatchTilesTask(tile, neighbors);
        task->compute(0.9, param_adjustments, param_train_deltas);
        bool res = task->error_check(0.9);
        if (res) {
          __sync_fetch_and_add(&success, 1);
        }
        __sync_fetch_and_add(&trials, 1);
        //trials++;
        //printf("prelim %f\n", (1.0*success)/trials);
      }
      printf("Result is %f\n", (1.0*success)/trials);
      float next_correct = (1.0*success)/trials;
        for (int i = 0; i < param_train_deltas.size(); i++) {
            if (next_correct > last_correct) {
              param_adjustments[i] += param_train_deltas[i];
            } else {
              //param_adjustments[i] -= param_train_deltas[i];
            }
          param_train_deltas[i] = 0;
        }

      if (next_correct > last_correct) {
        last_correct = next_correct; 
      printf("params:\n");
      printf("scale_x %f\n", 0.1 + param_adjustments[0]*0.05);
      printf("scale_y %f\n", 0.1 + param_adjustments[1]*0.05);
      printf("num_features %f\n", 1.0 + param_adjustments[2]);
      printf("num_octaves %f\n", 6.0 + param_adjustments[3]);
      printf("contrast_thresh %f\n", 0.01 + param_adjustments[4]*0.001);
      printf("edge_thresh %f\n", 20.0 + param_adjustments[5]);
      printf("edge_thresh %f\n", 1.2 + param_adjustments[6]*0.05);

      }
    }

  }
}


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
    //tfk::train_match_tiles_task(stack);
    //return;
    //stack->test_io();
    //return;
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

    int _start_x = 50000;
    int _start_y = 50000;

    //stack->render(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(50000 + size, 50000 + size)), "renderfull", tfk::FULL);
    stack->render(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(50000 + size, 50000 + size)), "renderthumb", tfk::THUMBNAIL);
    //stack->render(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(50000 + size, 50000 + size)), "renderthumb", tfk::PERCENT30);
    //stack->render(std::make_pair(cv::Point2f(_start_x,_start_y),cv::Point2f(_start_x + size, _start_y + size)), "renderbefore", tfk::PERCENT30);


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



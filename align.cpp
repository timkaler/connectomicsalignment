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
#include "fasttime.h"

fasttime_t global_start; 

namespace tfk {
  void train_match_tiles_task(Stack* stack) {
    int success = 0;
    int trials = 0;



    MRParams* min_params = new MRParams();
    min_params->set_accuracy(0.1);
    min_params->set_cost(0.1);
    min_params->put_float_param("scale_x", 0.1);
    min_params->put_float_param("scale_y", 0.1);
    min_params->put_int_param("num_features", 1);
    min_params->put_int_param("num_octaves", 1);
    min_params->put_float_param("contrast_threshold", 0.01);
    min_params->put_float_param("edge_threshold", 0.01);
    min_params->put_float_param("sigma", 1.0);
 
    MRParams* max_params = new MRParams();
    max_params->set_accuracy(0.0);
    max_params->set_cost(1.0);
    max_params->put_float_param("scale_x", 1.0);
    max_params->put_float_param("scale_y", 1.0);
    max_params->put_int_param("num_features", 16);
    max_params->put_int_param("num_octaves", 16);
    max_params->put_float_param("contrast_threshold", 0.1);
    max_params->put_float_param("edge_threshold", 50.0);
    max_params->put_float_param("sigma", 10.0);
    
  
    MRParams* default_params = new MRParams();
    default_params->set_accuracy(1.0);
    default_params->set_cost(1.0);
    default_params->put_float_param("scale_x", 1.0);
    default_params->put_float_param("scale_y", 1.0);
    default_params->put_int_param("num_features", 1);
    default_params->put_int_param("num_octaves", 6);
    default_params->put_float_param("contrast_threshold", 0.04);
    default_params->put_float_param("edge_threshold", 5.0);
    default_params->put_float_param("sigma", 1.2);  

    ParamDB* paramDB = new ParamDB(default_params, min_params, max_params);

    float last_correct = (1.0*success)/trials;

    for (int j = 0; j < 100; j++) {
       success = 0;
       trials = 0;
       cilk_for (int i = 0; i < 500; i++) {
        // pick random section.
        int section_id = rand()%stack->sections.size();
        Section* section = stack->sections[section_id];
        int tile_id = rand()%section->tiles.size();
        Tile* tile = section->tiles[tile_id];
        std::vector<Tile*> neighbors = section->get_all_close_tiles(tile);
        MatchTilesTask* task = new MatchTilesTask(paramDB, tile, neighbors);
        task->compute(0.9);
        bool res = task->error_check(0.9);
        if (res) {
          __sync_fetch_and_add(&success, 1);
        }
        __sync_fetch_and_add(&trials, 1);
        tile->release_full_image();
        for (int k = 0; k < neighbors.size(); k++) {
          neighbors[k]->release_full_image();
        }
      }
      printf("Result is %f\n", (1.0*success)/trials);
      paramDB->print_possible_params();
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
    printf("2 bounding box is %f %f %f %f\n", p_align_data->bounding_box.first.x, p_align_data->bounding_box.first.y, p_align_data->bounding_box.second.x, p_align_data->bounding_box.second.y);
    stack->_bounding_box = p_align_data->bounding_box;
    stack->init();
    printf("Got past the init\n");
    printf("stack has sections %zu\n", stack->sections.size());
    global_start = gettime();
    //tfk::train_match_tiles_task(stack);
    //return;
    //stack->test_io();
    printf("starting align 2d\n");
    //return;
    stack->align_2d();
    printf("Done with align 2d\n");
    stack->align_3d();
    printf("Done with align 3d\n");
    //stack->coarse_affine_align();
    //stack->elastic_align();

    for (int i = 0; i < stack->sections.size(); i++) {
      stack->sections[i]->elastic_transform_ready = true;
    }

    int size = 75000;
    std::clock_t start;
    double duration;

    start = std::clock();
    int _start_x = 100000;
    int _start_y = 100000;


    auto entire_bbox = stack->sections[0]->get_bbox();

    float x1 = (entire_bbox.first.x + entire_bbox.second.x)/2;
    //float x1 = (entire_bbox.first.x);
    float x2 = x1+50000;
    float y1 = (entire_bbox.first.y+entire_bbox.second.y)/2;
    //float y1 = (entire_bbox.first.y);
    float y2 = y1+50000;
    auto smaller_bbox = std::make_pair(cv::Point2f(x1,y1), cv::Point2f(x2,y2)); 
    //stack->render(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(50000 + size, 50000 + size)), "renderfull", tfk::FULL);
    printf("Right before render\n");
    //stack->render(std::make_pair(cv::Point2f(_start_x,_start_y),cv::Point2f(_start_x + size, _start_y + size)), "renderthumb", tfk::THUMBNAIL);
    stack->render(entire_bbox, "renderthumb", tfk::THUMBNAIL);
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


    for (int nf = 0; nf < 3; nf++) {
        int num_features = 1 << nf;
        for (int no = 8; no < 16; no++) {
            int num_octaves = no;
            //for (float sigma = 1; sigma < 2; sigma+=.1) {
            
                //for (float ct = .001; ct < .025; ct+=.001) {
            
                    //for (float et = 4; et < 15; et+=3) {
                          float sigma = 1.6;
                          float ct = .015;
                          float et = 6;
                          for (float scale = .1; scale < 1.03; scale+=.1) {
                              tfk::params p;
                              p.num_features = num_features;//num_features;
                              p.num_octaves = num_octaves;
                              p.contrast_threshold = ct;//CONTRAST_THRESH;//10.0;
                              p.edge_threshold = et;//EDGE_THRESH_2D;
                              p.sigma = sigma;//sigma;
                              p.res = tfk::FULL;
                              p.scale_x = scale;
                              p.scale_y = scale;
                              ps.push_back(p);
                          }
                    //}
                //}

            //}
        }
    }
    std::random_shuffle ( ps.begin(), ps.end() );
    printf("testing %zu different paramter combinations\n", ps.size());
    std::vector<std::tuple<int, double, int>> current;

    double threshold = 5;
    int trials = 100;
    stack->parameter_optimization(trials, threshold, ps, current);
    /*for (int j = 0; j < ps.size(); j++) {
      printf("%d, %d, %f, %f, %f, %d, %d, %f, %f, %d\n",
          ps[j].num_features, ps[j].num_octaves, ps[j].sigma, ps[j].contrast_threshold  , ps[j].edge_threshold ,
          ps[j].res, std::get<0>(current[j]), threshold, std::get<1>(current[j]), std::get<2>(current[j]));
    }*/
    return;
}

void test_learning(align_data_t *p_align_data) {

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
    printf("stack has sections %zu\n", stack->sections.size());
    std::vector<tfk::params> ps;
    int trials = 10000;

    stack->test_learning(trials, 5, 2);
    return;
}


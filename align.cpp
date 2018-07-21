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
#include "render.hpp"
#include "data.hpp"
#include "ParamsDatabase.pb.h"
fasttime_t global_start; 


  //TODO(wheatman) this should fill both paramdb and train thwe error checker
  // use a known good set of parameters to train against
  // accuracy is how close
  // good if give same reult as known good
  void fill_match_tiles_task_pdb(align_data_t *p_align_data) {


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
    //printf("2 bounding box is %f %f %f %f\n", p_align_data->bounding_box.first.x, p_align_data->bounding_box.first.y, p_align_data->bounding_box.second.x, p_align_data->bounding_box.second.y);
    stack->_bounding_box = p_align_data->bounding_box;
    stack->init();
    

    
    std::vector<tfk::MRParams*> param_options;
    tfk::Section* section = stack->sections[0];
    tfk::Tile* tile = section->tiles[0];
    tile->paramdbs = stack->paramdbs;
    tile->ml_models = stack->ml_models;
    std::vector<tfk::Tile*> neighbors = section->get_all_close_tiles(tile);
    tfk::Tile* other_tile = neighbors[rand() % neighbors.size()];
    tfk::MatchTilePairTask* task = new tfk::MatchTilePairTask(tile, other_tile);
    task->setup_param_db_init(&param_options);

    //forget what we learned from the model
    // needed since we used load to get the data
    // stack->ml_models[MATCH_TILE_PAIR_TASK_ID]->enable_training();
    


    //paramDB->print_possible_params();
    for (int iter = 0; iter < 1; iter++) {
      if (false) {
        int apx_count = 0;
        cilk_for (int i = 0; i < 15000; i++) {
          //printf("working on random tile %d\n", i);
          //paramDB->print_possible_params();
          int section_id = rand()%stack->sections.size();
          tfk::Section* section = stack->sections[section_id];
          int tile_id = rand()%section->tiles.size();
          // pick random section.
          tfk::Tile* a_tile = section->tiles[tile_id];
          std::vector<tfk::Tile*> neighbors = section->get_all_close_tiles(a_tile);
          tfk::Tile* b_tile = neighbors[rand() % neighbors.size()];
          a_tile->paramdbs = stack->paramdbs;
          a_tile->ml_models = stack->ml_models;
          b_tile->paramdbs = stack->paramdbs;
          b_tile->ml_models = stack->ml_models;
          tfk::MatchTilePairTask* task = new tfk::MatchTilePairTask(a_tile, b_tile);
          a_tile->get_tile_data(tfk::FULL);
          b_tile->get_tile_data(tfk::FULL);
          tfk::MRParams *known_good = new tfk::MRParams();
          known_good->put_int_param("num_features", 1);
          known_good->put_int_param("num_octaves", 10);
          known_good->put_float_param("scale", 1);
          tfk::MatchTilePairTask* task_known_good = new tfk::MatchTilePairTask(a_tile, b_tile);
          task_known_good->compute_with_params(known_good);
          task_known_good->error_check(-1);
          
          task->setup_param_db(1, task_known_good);
          a_tile->release_full_image();
          b_tile->release_full_image();
          a_tile->release_2d_keypoints();
          b_tile->release_2d_keypoints();
          apx_count++;
          if (apx_count % 100 == 0) {
            printf("apx_count = %d\n", apx_count);
          }
        }
        stack->ml_models[MATCH_TILE_PAIR_TASK_ID]->save("ml_after_param_opt.ml.25000_5_4_pre");
      }
      
      printf("finished iter %d\n", iter);
      stack->ml_models[MATCH_TILE_PAIR_TASK_ID]->enable_training();
      stack->ml_models[MATCH_TILE_PAIR_TASK_ID]->train(true);
      stack->ml_models[MATCH_TILE_PAIR_TASK_ID]->save("ml_after_param_opt.ml.25000_5_4_post");
      //task->paramDB->print_possible_params();
    }
    
    //stack->ml_models[MATCH_TILE_PAIR_TASK_ID]->save("ml_after_param_opt.ml.25000_5_3");
    /*
    ParamsDatabase pdb_new;
    task->paramDB->to_proto(&pdb_new);
    printf("have %d diferent parameters in new\n",pdb_new.params_size());
    std::fstream output("match_tiles_task_pdb_big.pb", std::ios::out | std::ios::trunc | std::ios::binary);
    if (!pdb_new.SerializeToOstream(&output)) {
      std::cerr << "Failed to write pdb proto." << std::endl;
    }
    */
    /*
    for (float acc = .5; acc <=1; acc+=.01) {
      printf("acc = %f, cost =  %f\n",acc, task->paramDB->get_params_for_accuracy(acc)->get_cost());
      task->paramDB->get_params_for_accuracy(acc)->print();
    }
    */
    
  }

  void testing_corralation_test(align_data_t *p_align_data) {


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
    //printf("2 bounding box is %f %f %f %f\n", p_align_data->bounding_box.first.x, p_align_data->bounding_box.first.y, p_align_data->bounding_box.second.x, p_align_data->bounding_box.second.y);
    stack->_bounding_box = p_align_data->bounding_box;
    stack->init();
    

    
    std::vector<tfk::MRParams*> param_options;
    tfk::Section* section = stack->sections[0];
    tfk::Tile* tile = section->tiles[0];
    tile->paramdbs = stack->paramdbs;
    tile->ml_models = stack->ml_models;
    std::vector<tfk::Tile*> neighbors = section->get_all_close_tiles(tile);
    tfk::Tile* other_tile = neighbors[rand() % neighbors.size()];
    tfk::MatchTilePairTask* task = new tfk::MatchTilePairTask(tile, other_tile);
    task->setup_param_db_init(&param_options);
    int apx_count = 0;
    std::vector<float> pos_preds;
    std::vector<float> neg_preds;
    std::mutex mutex = std::mutex();
    cilk_for (int i = 0; i < 20; i++) {
      //printf("working on random tile %d\n", i);
      //paramDB->print_possible_params();
      int section_id = rand()%stack->sections.size();
      tfk::Section* section = stack->sections[section_id];
      int tile_id = rand()%section->tiles.size();
      // pick random section.
      tfk::Tile* a_tile = section->tiles[tile_id];
      std::vector<tfk::Tile*> neighbors = section->get_all_close_tiles(a_tile);
      tfk::Tile* b_tile = neighbors[rand() % neighbors.size()];
      a_tile->paramdbs = stack->paramdbs;
      a_tile->ml_models = stack->ml_models;
      b_tile->paramdbs = stack->paramdbs;
      b_tile->ml_models = stack->ml_models;
      tfk::MatchTilePairTask* task = new tfk::MatchTilePairTask(a_tile, b_tile);
      a_tile->get_tile_data(tfk::FULL);
      b_tile->get_tile_data(tfk::FULL);
      tfk::MRParams *known_good = new tfk::MRParams();
      known_good->put_int_param("num_features", 1);
      known_good->put_int_param("num_octaves", 10);
      known_good->put_float_param("scale", 1);
      tfk::MatchTilePairTask* task_known_good = new tfk::MatchTilePairTask(a_tile, b_tile);
      task_known_good->compute_with_params(known_good);
      task_known_good->error_check(-1);
      
      cilk_for (int j = 0; j < param_options.size(); j++) {
          tfk::MRParams *param = param_options[j];
          //printf("param option %d out of %zu\n", j, param_options->size());
          //int count = param->get_count();
          task->compute_with_params(param);
          task->error_check(1);
          bool correct = task->compare_results_and_update_model(task_known_good, 5);
          mutex.lock();
          if (correct) {
              pos_preds.push_back(a_tile->neighbor_correlations[b_tile->tile_id]);
          } else {
              neg_preds.push_back(a_tile->neighbor_correlations[b_tile->tile_id]);
          }
          mutex.unlock();


      }
      delete task;
      delete known_good;
      a_tile->release_full_image();
      b_tile->release_full_image();
      a_tile->release_2d_keypoints();
      b_tile->release_2d_keypoints();
      apx_count++;
      if (apx_count % 100 == 0) {
        printf("apx_count = %d\n", apx_count);
      }
    }
    FILE * pFile;
      pFile = fopen ("for_hist_corr.csv","w");
      for (int i = 0; i < pos_preds.size(); i++) {
        fprintf(pFile, "%.2f, ", pos_preds[i]);
      }
      fprintf(pFile, "\n");
      for (int i = 0; i < neg_preds.size(); i++) {
        fprintf(pFile, "%.2f, ", neg_preds[i]);
      }
      fprintf(pFile, "\n");
      delete stack;
    
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
    stack->use_bbox_prefilter = false;
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

    float x1 = (entire_bbox.first.x + entire_bbox.second.x)/2+5000;// -2500 + 5000;
    //float x1 = (entire_bbox.first.x);
    //float x2 = x1+7500;
    float x2 = x1+5000;
    float y1 = (entire_bbox.first.y+entire_bbox.second.y)/2+5000;// -2500 + 5000;
    //float y1 = (entire_bbox.first.y);
    //float y2 = y1+7500;
    float y2 = y1+5000;
    auto smaller_bbox = std::make_pair(cv::Point2f(x1,y1), cv::Point2f(x2,y2)); 
    //stack->render(std::make_pair(cv::Point2f(50000,50000),cv::Point2f(50000 + size, 50000 + size)), "renderfull", tfk::FULL);

    tfk::Render* render = new tfk::Render();
    render->render_stack(stack, entire_bbox, tfk::THUMBNAIL, ALIGN_OUTPUT_FILE_DIRECTORY + "/rendertest0");
    //render->render_stack(stack, entire_bbox, tfk::THUMBNAIL, ALIGN_OUTPUT_FILE_DIRECTORY+"/rendertest1");

    printf("Right before render\n");
    //printf("Is Overlap: %d\n",tfk::mesh_overlaps(stack));
    printf("Num Sections: %d\n", stack->sections.size());
    //overlay_triangles_stack(stack, entire_bbox, tfk::THUMBNAIL, "rendertest0");
    // tfk::Data* data = new tfk::Data();
    // data->sample_stack(stack, 10, 10000, "sampletest0");
    //stack->render(std::make_pair(cv::Point2f(_start_x,_start_y),cv::Point2f(_start_x + size, _start_y + size)), "renderthumb", tfk::THUMBNAIL);
    //stack->render(entire_bbox, "renderthumb", tfk::THUMBNAIL);
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
    stack->_bounding_box = p_align_data->bounding_box;
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
    stack->init();
    printf("stack has sections %zu\n", stack->sections.size());
    std::vector<tfk::params> ps;
    int trials = 10000;

    stack->train_fsj(trials);
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
    stack->_bounding_box = p_align_data->bounding_box;
    stack->init();
    printf("stack has sections %zu\n", stack->sections.size());
    std::vector<tfk::params> ps;
    int trials = 10000;

    stack->test_learning(trials, 5, 2);
    return;
}


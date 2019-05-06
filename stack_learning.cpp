
void tfk::Stack::train_fsj(int trials) {
  float PX_ERROR_THRESH = 1.0;
  params best_params;
  best_params.num_features = 1000;
  best_params.num_octaves = 6;
  best_params.contrast_threshold = .015;//CONTRAST_THRESH;
  best_params.edge_threshold = 10;//EDGE_THRESH_2D;
  best_params.sigma = 1.6;//1.6;
  best_params.scale_x = 0.5;//1.0;
  best_params.scale_y = 0.5;//1.0;
  best_params.res = FULL;


  params trial_params;
  trial_params.num_features = 1000;
  trial_params.num_octaves = 6;
  trial_params.contrast_threshold = .015;
  trial_params.edge_threshold = 10;
  trial_params.sigma = 1.6;
  trial_params.scale_x = 0.1;
  trial_params.scale_y = 0.1;
  trial_params.res = FULL;

  MRParams* best_mr_params = new MRParams();
  best_mr_params->put_int_param("num_features",1000);
  best_mr_params->put_int_param("num_octaves",6);
  best_mr_params->put_float_param("contrast_threshold",0.015);
  best_mr_params->put_float_param("edge_threshold",10.0);
  best_mr_params->put_float_param("sigma",1.6);
  best_mr_params->put_float_param("scale",1.0);

  MRParams* trial_mr_params = new MRParams();
  trial_mr_params->put_int_param("num_features",1000);
  trial_mr_params->put_int_param("num_octaves",6);
  trial_mr_params->put_float_param("contrast_threshold",0.015);
  trial_mr_params->put_float_param("edge_threshold",10.0);
  trial_mr_params->put_float_param("sigma",1.6);
  trial_mr_params->put_float_param("scale",0.1);


  std::vector<int> random_numbers_1;
  std::vector<int> random_numbers_2;
  std::vector<int> random_numbers_3;

  for (int i = 0; i < trials; i++) {
    int section_id = rand();
    // while (section_id%this->sections.size() == 30) {
    //   section_id = rand();
    // }
    random_numbers_1.push_back(section_id);
    random_numbers_2.push_back(rand());
    random_numbers_3.push_back(rand());
  }

  uint8_t padding1[128] __attribute__((unused));
  int64_t success_count = 0;
  uint8_t padding2[128] __attribute__((unused));
  
  int64_t failure_count = 0;
  uint8_t padding3[128] __attribute__((unused));
  int64_t failure_type2_count = 0;
  uint8_t padding4[128] __attribute__((unused));

  MLAnn* model = new MLAnn(12-4+6+4+1, TFK_TMP_DIR + "/tfk_test_model");
  //model->load("tfk_test_model", true);
  //model->enable_training();
  //model->train(false);

  //return;

  model->enable_training();
   

  cilk_for (int i = 0; i < trials; i++) {
    int section_index = random_numbers_1[i]%this->sections.size();


    if (this->sections[section_index]->tiles.size() == 0) {
      printf("section has zero tiles %d\n", this->sections[section_index]->real_section_id);
      exit(1);
      continue;
    }

    int tile_index = random_numbers_2[i]%this->sections[section_index]->tiles.size();

    Tile* _tile_a = this->sections[section_index]->tiles[tile_index];
    std::vector<Tile*> close_tiles = this->sections[section_index]->get_all_close_tiles_with_min_overlap(_tile_a, 50);

    if (close_tiles.size() == 0) continue;
    int n_index = random_numbers_3[i]%close_tiles.size();
    Tile* _tile_b = close_tiles[n_index];


    Tile _tile_a_copy = *_tile_a;
    Tile _tile_b_copy = *_tile_b;
    Tile* tile_a = &_tile_a_copy;
    Tile* tile_b = &_tile_b_copy;


    std::map<int, TileSiftTask*> dependencies;
    TileSiftTask* sift_task_a = new TileSiftTask(this->paramdbs[MATCH_TILE_PAIR_TASK_ID], tile_a);
    TileSiftTask* sift_task_b = new TileSiftTask(this->paramdbs[MATCH_TILE_PAIR_TASK_ID], tile_b);

    dependencies[tile_a->tile_id] = sift_task_a;
    dependencies[tile_b->tile_id] = sift_task_b;

    dependencies[tile_a->tile_id]->compute(0.9);
    dependencies[tile_b->tile_id]->compute(0.9);
    MatchTilePairTask* task2 = new MatchTilePairTask(tile_a, tile_b, true);
    task2->align_data = this->align_data;
    task2->dependencies = dependencies;
    task2->compute_with_params(trial_mr_params);
    bool res1 = task2->error_check(1000*1.9);
    delete dependencies[tile_a->tile_id];
    delete dependencies[tile_b->tile_id];
    cv::Point2f offset2 = task2->predicted_offset;


    MatchTilePairTask* task1 = new MatchTilePairTask(tile_a, tile_b, true);
    task1->align_data = this->align_data;
    task1->compute_with_params(best_mr_params);
    bool res2 = task1->error_check(1000*1.9);
    cv::Point2f offset1 = task1->predicted_offset;

      tile_a->release_full_image();
      tile_b->release_full_image();
    //if (!res1 || !res2) continue;
    //if (!res1 || !res2) {
    //  if (res1 != res2) {
    //    __sync_fetch_and_add(&failure_count, 1);
    //  } else {
    //    __sync_fetch_and_add(&success_count, 1);
    //  }
    //  delete task1;
    //  delete task2;
    //  continue;
    //}
    float dx = offset1.x - offset2.x;
    float dy = offset1.y - offset2.y;
    float dist = sqrt(dx*dx+dy*dy);
    int local_failure_count = failure_count;
    int local_success_count = success_count;
    int local_failure_type2_count = failure_type2_count;
    //if (!res1) __sync_fetch_and_add() continue; // if the fast pass doesn't succeed, that means we're falling onto slow pass anyways.

    //if (!res1) continue;

    if (res1 != res2 || dist > PX_ERROR_THRESH) {
      if (res1 == res2) {
        local_failure_type2_count = __sync_fetch_and_add(&failure_type2_count,1);
        printf("failure dist is %f\n", dist);
      } else {
        printf("failure, results don't match. %d, %d, dist %f\n", res1, res2, dist);
        dist = 100;
      }
      local_failure_count = __sync_fetch_and_add(&failure_count, 1)+1;
      model->add_training_example(task2->get_feature_vector(), 0, dist);
      //std::vector<float> vec = task2->get_feature_vector();
      //printf("fast:");
      //for (int x = 0; x < vec.size(); x++) {
      // printf("%f\t", vec[x]);
      //}
      // printf("\n");
      // vec = task1->get_feature_vector();
      //printf("slow:");
      //for (int x = 0; x < vec.size(); x++) {
      // printf("%f\t", vec[x]);
      //}
      // printf("\n\n");

      //printf("failure\n");
    } else {
      //if (!res1) printf("both failed\n");
      //printf("success\n");
      if (dist < PX_ERROR_THRESH/2) {
         local_success_count = __sync_fetch_and_add(&success_count, 1)+1;
         model->add_training_example(task2->get_feature_vector(), 1, -1.0);
      }
    }
    if (i%100 == 0 || i == trials-1) {
    printf("failures %d, successes %d, fraction %f%%; type2: %f%%\n", local_failure_count, local_success_count, (100.0*local_failure_count)/(local_failure_count+local_success_count), (100.0*local_failure_type2_count) / (local_failure_count + local_success_count));
    }
    delete task1;
    delete task2;
  }

  model->train(false);
  model->save(TFK_TMP_DIR + "/tfk_test_model");
}


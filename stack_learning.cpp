
// returns a vector
// the first item is how many were within the threshold
// the second item is the time it took
// the third item is the total number of matches
void tfk::Stack::parameter_optimization(int trials, double threshold, std::vector<params> ps,
        std::vector<std::tuple<int, double, int>>&results) {

  //std::vector<std::tuple<int, double, int>> results;
  results.reserve(ps.size());
  FILE * pFile;
  pFile = fopen("param_results.csv", "w");
  fprintf(pFile, "num_features,num_octaves,sigma,contrast_threshold,edge_threshold,scale,number_correct,threshold,time,memory\n");

  std::vector<Tile*[2]> tile_pairs(trials);


  // choose all the random pairs of overlapping tiles
  cilk_for (int i = 0; i < trials; i++) {
    // pick a random section
    Section* sec = this->sections[rand()%this->n_sections];
    // pick a random tile in that section
    Tile* tile_a = sec->tiles[rand()%sec->n_tiles];
    std::vector<int> neighbor_ids = sec->get_all_close_tiles(tile_a->tile_id);
    //pick a random neighbor of that tile
    while (neighbor_ids.size() == 0) {
      tile_a = sec->tiles[rand()%sec->n_tiles];
      neighbor_ids = sec->get_all_close_tiles(tile_a->tile_id);
    }
    Tile* tile_b = sec->tiles[neighbor_ids[rand()%neighbor_ids.size()]];
    tile_pairs[i][0] = tile_a;
    tile_pairs[i][1] = tile_b;
  }
  //printf("found pairs for testing\n");



  std::vector<cv::Point2f> correct_movement(trials);

  params best_params;
  best_params.num_features = 1;
  best_params.num_octaves = 6;
  best_params.contrast_threshold = .015;//CONTRAST_THRESH;
  best_params.edge_threshold = 10;//EDGE_THRESH_2D;
  best_params.sigma = 1.6;//1.6;
  best_params.scale_x = 1.0;
  best_params.scale_y = 1.0;
  best_params.res = FULL;


  // find the best movement for each pair using the best params
  cilk_for (int i = 0; i < trials; i++) {
    Tile* tile_a = new Tile(tile_pairs[i][0]->tile_data);
    Tile* tile_b = new Tile(tile_pairs[i][1]->tile_data);
    std::vector<cv::KeyPoint> a_tile_keypoints;
    cv::Mat a_tile_desc;
    std::vector<cv::KeyPoint> b_tile_keypoints;
    cv::Mat b_tile_desc;
    tile_a->compute_sift_keypoints2d_params(best_params, a_tile_keypoints, a_tile_desc, tile_b);
    tile_b->compute_sift_keypoints2d_params(best_params, b_tile_keypoints, b_tile_desc, tile_a);
    std::vector< cv::Point2f > filtered_match_points_a(0);
    std::vector< cv::Point2f > filtered_match_points_b(0);

    cv::Point2f relative_offset = this->sections[tile_a->section_id]->compute_tile_matches_pair(tile_a, tile_b,
      a_tile_keypoints, b_tile_keypoints,
      a_tile_desc, b_tile_desc,
      filtered_match_points_a,
      filtered_match_points_b, 5);
    tile_a->offset_x += relative_offset.x + (tile_b->x_start - tile_a->x_start);
    tile_a->offset_y += relative_offset.y + (tile_b->y_start - tile_a->y_start);
    correct_movement[i] = relative_offset;

    if (filtered_match_points_a.size() == 0) {
      printf("no matches for set params\n");
    } else {
      //printf("accuracy score of %f\n", tile_a->error_tile_pair(tile_b));
    }


    //printf("%f, %f, ", correct_movement[i][0], correct_movement[i][1]);
    tile_a->release_2d_keypoints();
    tile_b->release_2d_keypoints();
    delete tile_a;
    delete tile_b;
  }
  //printf("\n");
  //printf("found correct movements\n");


  int ps_size = ps.size();
  std::vector<std::vector<std::tuple<int, double, int> > > worker_results(ps_size);
  for (int k = 0; k < ps_size; k++) {
    double duration;
    std::clock_t  start = std::clock();
    int valid_moves = 0;
    int matches_count = 0;

    std::vector<int> worker_valid_moves(trials);
    std::vector<int> worker_matches_count(trials);


    cilk_for (int i = 0; i < trials; i++) {

      Tile* tile_a = new Tile(tile_pairs[i][0]->tile_data);
      Tile* tile_b = new Tile(tile_pairs[i][1]->tile_data);
      std::vector<cv::KeyPoint> a_tile_keypoints;
      cv::Mat a_tile_desc;
      std::vector<cv::KeyPoint> b_tile_keypoints;
      cv::Mat b_tile_desc;
      tile_a->compute_sift_keypoints2d_params(ps[k], a_tile_keypoints, a_tile_desc, tile_b);
      tile_b->compute_sift_keypoints2d_params(ps[k], b_tile_keypoints, b_tile_desc, tile_a);
      std::vector< cv::Point2f > filtered_match_points_a(0);
      std::vector< cv::Point2f > filtered_match_points_b(0);

      cv::Point2f relative_offset = this->sections[tile_a->section_id]->compute_tile_matches_pair(tile_a, tile_b,
        a_tile_keypoints, b_tile_keypoints,
        a_tile_desc, b_tile_desc,
        filtered_match_points_a,
        filtered_match_points_b, 5);

      if (filtered_match_points_a.size() > 0) {
        //printf("found matching points\n");
        //tile_a->offset_x += relative_offset.x + (tile_b->x_start - tile_a->x_start);
        //tile_a->offset_y += relative_offset.y + (tile_b->y_start - tile_a->y_start);
        //float accuracy = tile_a->error_tile_pair(tile_b);
      }

      cv::Point2f error_vec = relative_offset - correct_movement[i];
      if (std::abs(error_vec.x) <= threshold && std::abs(error_vec.y) <= threshold) {
        worker_valid_moves[i]++;
      }
      worker_matches_count[i] += (tile_a->p_kps->size() + tile_b->p_kps->size());
      tile_a->release_2d_keypoints();
      tile_b->release_2d_keypoints();
      delete tile_a;
      delete tile_b;
    }
    //printf("\n");
    for (int i = 0; i < worker_matches_count.size(); i++) {
      matches_count += worker_matches_count[i];
      valid_moves += worker_valid_moves[i];
    }
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    worker_results[k].push_back(std::make_tuple(valid_moves, duration, matches_count));
    fprintf(pFile,"%d,%d,%f,%f,%f,%f,%d,%f,%f,%d\n",
          ps[k].num_features, ps[k].num_octaves, ps[k].sigma, ps[k].contrast_threshold  , ps[k].edge_threshold ,
          ps[k].scale_x, valid_moves , threshold, duration  , matches_count );
    //printf("%d,%d,%f,%f,%f,%f,%d,%f,%f,%d\n",
    //      ps[k].num_features, ps[k].num_octaves, ps[k].sigma, ps[k].contrast_threshold  , ps[k].edge_threshold ,
    //      ps[k].scale_x, valid_moves , threshold, duration  , matches_count );
    fflush(pFile);
    printf("\r%d out of %zu", k, ps.size());
    fflush(stdout);
  }
  fclose(pFile);

  for (int i = 0; i < worker_results.size(); i++) {
    for (int j = 0; j < worker_results[i].size(); j++) {
      results.push_back(worker_results[i][j]);
    }
  }

}



void tfk::Stack::train_fsj(int trials) {
  params best_params;
  best_params.num_features = 1;
  best_params.num_octaves = 6;
  best_params.contrast_threshold = .015;//CONTRAST_THRESH;
  best_params.edge_threshold = 10;//EDGE_THRESH_2D;
  best_params.sigma = 1.6;//1.6;
  best_params.scale_x = 1.0;
  best_params.scale_y = 1.0;
  best_params.res = FULL;


  params trial_params;
  trial_params.num_features = 4;
  trial_params.num_octaves = 6;
  trial_params.contrast_threshold = .015;
  trial_params.edge_threshold = 10;
  trial_params.sigma = 1.6;
  trial_params.scale_x = 0.3;
  trial_params.scale_y = 0.3;
  trial_params.res = FULL;

  MRParams* best_mr_params = new MRParams();
  best_mr_params->put_int_param("num_features",1);
  best_mr_params->put_int_param("num_octaves",6);
  best_mr_params->put_float_param("contrast_threshold",0.015);
  best_mr_params->put_float_param("edge_threshold",10.0);
  best_mr_params->put_float_param("sigma",1.6);
  best_mr_params->put_float_param("scale",1.0);

  MRParams* trial_mr_params = new MRParams();
  trial_mr_params->put_int_param("num_features",4);
  trial_mr_params->put_int_param("num_octaves",6);
  trial_mr_params->put_float_param("contrast_threshold",0.015);
  trial_mr_params->put_float_param("edge_threshold",10.0);
  trial_mr_params->put_float_param("sigma",1.6);
  trial_mr_params->put_float_param("scale",0.3);


  std::vector<int> random_numbers_1;
  std::vector<int> random_numbers_2;
  std::vector<int> random_numbers_3;

  for (int i = 0; i < trials; i++) {
    random_numbers_1.push_back(rand());
    random_numbers_2.push_back(rand());
    random_numbers_3.push_back(rand());
  }

  int64_t success_count = 0;
  int64_t failure_count = 0;

  int64_t failure_type2_count = 0;

  MLAnn* model = new MLAnn(12-4, "tfk_test_model");
  //model->load("tfk_test_model", true);
  //model->enable_training();
  //model->train(false);

  //return;

  model->enable_training();
   

  cilk_for (int i = 0; i < trials; i++) {
    int section_index = random_numbers_1[i]%this->sections.size();
    int tile_index = random_numbers_2[i]%this->sections[section_index]->tiles.size();

    Tile* tile_a = this->sections[section_index]->tiles[tile_index];
    std::vector<Tile*> close_tiles = this->sections[section_index]->get_all_close_tiles(tile_a);

    if (close_tiles.size() == 0) continue;
    int n_index = random_numbers_3[i]%close_tiles.size();
    Tile* tile_b = close_tiles[n_index];



    std::map<int, TileSiftTask*> dependencies;
    TileSiftTask* sift_task_a = new TileSiftTask(this->paramdbs[MATCH_TILE_PAIR_TASK_ID], tile_a);
    TileSiftTask* sift_task_b = new TileSiftTask(this->paramdbs[MATCH_TILE_PAIR_TASK_ID], tile_b);

    dependencies[tile_a->tile_id] = sift_task_a;
    dependencies[tile_b->tile_id] = sift_task_b;

    dependencies[tile_a->tile_id]->compute(0.9);
    dependencies[tile_b->tile_id]->compute(0.9);
    MatchTilePairTask* task2 = new MatchTilePairTask(tile_a, tile_b, true);
    task2->dependencies = dependencies;
    task2->compute_with_params(trial_mr_params);
    bool res1 = task2->error_check(1.9);
    delete dependencies[tile_a->tile_id];
    delete dependencies[tile_b->tile_id];
    cv::Point2f offset2 = task2->predicted_offset;


    MatchTilePairTask* task1 = new MatchTilePairTask(tile_a, tile_b, true);
    task1->compute_with_params(best_mr_params);
    bool res2 = task1->error_check(1.9);
    cv::Point2f offset1 = task1->predicted_offset;

      tile_a->release_2d_keypoints();
      tile_b->release_2d_keypoints();
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
    if (res1 != res2 || dist > 2.0) {
      if (res1 == res2) {
        local_failure_type2_count = __sync_fetch_and_add(&failure_type2_count,1);
      }
      local_failure_count = __sync_fetch_and_add(&failure_count, 1)+1;
      model->add_training_example(task2->get_feature_vector(), 0, dist);
      printf("failure\n");
    } else {
      printf("success\n");
      local_success_count = __sync_fetch_and_add(&success_count, 1)+1;
      model->add_training_example(task2->get_feature_vector(), 1, dist);
    }
    printf("failures %d, successes %d, fraction %f%%; type2: %f%%\n", local_failure_count, local_success_count, (100.0*local_failure_count)/(local_failure_count+local_success_count), (100.0*local_failure_type2_count) / (local_failure_count + local_success_count));
    delete task1;
    delete task2;
  }

  model->train(false);
  model->save("tfk_test_model");
}


void tfk::Stack::test_learning(int trials, int vector_grid_size, int vector_mode) {

  std::vector<Tile*[2]> tile_pairs(trials);


  // choose all the random pairs of overlapping tiles
  cilk_for (int i = 0; i < trials; i++) {
    // pick a random section
    Section* sec = this->sections[rand()%this->n_sections];
    // pick a random tile in that section
    Tile* tile_a = sec->tiles[rand()%sec->n_tiles];
    std::vector<int> neighbor_ids = sec->get_all_close_tiles(tile_a->tile_id);
    //pick a random neighbor of that tile
    while (neighbor_ids.size() == 0) {
      tile_a = sec->tiles[rand()%sec->n_tiles];
      neighbor_ids = sec->get_all_close_tiles(tile_a->tile_id);
    }
    Tile* tile_b = sec->tiles[neighbor_ids[rand()%neighbor_ids.size()]];
    tile_pairs[i][0] = tile_a;
    tile_pairs[i][1] = tile_b;
  }
  //printf("found pairs for testing\n");

  params best_params;
  best_params.num_features = 1;
  best_params.num_octaves = 6;
  best_params.contrast_threshold = .015;//CONTRAST_THRESH;
  best_params.edge_threshold = 10;//EDGE_THRESH_2D;
  best_params.sigma = 1.6;//1.6;
  best_params.scale_x = 1.0;
  best_params.scale_y = 1.0;
  best_params.res = FULL;

  //random forest test
  if (false) {
    printf("random forest test\n");
    int training_runs = 9*trials/10;

    cv::Mat data;
    if (vector_mode == 2) {
      data = cv::Mat::zeros((training_runs)*2, 1 + (vector_grid_size * vector_grid_size), CV_32F);
    } else {
      printf("not implemented for this vector mode\n");
      return;
    }

    cv::Mat labels = cv::Mat::zeros((training_runs)*2, 1, CV_32F);

    float thresh = .75;
    // find the best movement for each pair using the best params
    cilk_for (int i = 0; i < training_runs; i++) {
      Tile* tile_a = new Tile(tile_pairs[i][0]->tile_data);
      Tile* tile_b = new Tile(tile_pairs[i][1]->tile_data);
      std::vector<cv::KeyPoint> a_tile_keypoints;
      cv::Mat a_tile_desc;
      std::vector<cv::KeyPoint> b_tile_keypoints;
      cv::Mat b_tile_desc;
      tile_a->compute_sift_keypoints2d_params(best_params, a_tile_keypoints, a_tile_desc, tile_b);
      tile_b->compute_sift_keypoints2d_params(best_params, b_tile_keypoints, b_tile_desc, tile_a);
      std::vector< cv::Point2f > filtered_match_points_a(0);
      std::vector< cv::Point2f > filtered_match_points_b(0);

      cv::Point2f relative_offset = this->sections[tile_a->section_id]->compute_tile_matches_pair(tile_a, tile_b,
        a_tile_keypoints, b_tile_keypoints,
        a_tile_desc, b_tile_desc,
        filtered_match_points_a,
        filtered_match_points_b, 5);
      tile_a->offset_x += relative_offset.x + (tile_b->x_start - tile_a->x_start);
      tile_a->offset_y += relative_offset.y + (tile_b->y_start - tile_a->y_start);

      cv::Mat vector = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
      float label = tile_a->error_tile_pair(tile_b) > thresh;

      if (label) {
        // a good example
        for (int j = 0; j < vector.cols; j++) {
          data.at<float>(2*i, j) = vector.at<float>(j);
        }
        labels.at<float>(2*i) = label;
        if (rand()%2) {
          tile_a->offset_x += (i%10)+1;
        } else {
          tile_a->offset_x -= (i%10)+1;
        }
        if (rand()%2) {
          tile_a->offset_y += (i%10)+1;
        } else {
          tile_a->offset_y -= (i%10)+1;
        }
        cv::Mat vector2 = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
        // a bad example
        for (int j = 0; j < vector2.cols; j++) {
          data.at<float>(2*i+1, j) = vector2.at<float>(j);
        }
        labels.at<float>(2*i+1) = (float)0;
      } else {
        // this is stupid but trying to fix an indexing error
        // unkown examples
        for (int j = 0; j < vector.cols; j++) {
          data.at<float>(2*i, j) = vector.at<float>(j);
        }
        labels.at<float>(2*i) = (float).5;
        for (int j = 0; j < vector.cols; j++) {
          data.at<float>(2*i+1, j) = vector.at<float>(j);
        }
        labels.at<float>(2*i+1) = (float).5;
      }
      tile_a->release_2d_keypoints();
      tile_b->release_2d_keypoints();
      tile_a->release_full_image();
      tile_b->release_full_image();
      delete tile_a;
      delete tile_b;
    }
    static cv::Ptr<cv::ml::RTrees> rtree = cv::ml::RTrees::create();
    int correct[10] = {0};
    //printf(" %d, %d, %d, %d\n",data.rows, data.cols, labels.rows, labels.cols );
    printf("starting training\n");
    static cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, labels);
    rtree->train(tdata);
    printf("done training\n");
    //rtree->predict(cv::Mat::zeros(1, 26, CV_32S));
    int total[10] = {0};
    cilk_for (int i = training_runs; i < trials; i++) {
      Tile* tile_a = new Tile(tile_pairs[i][0]->tile_data);
      Tile* tile_b = new Tile(tile_pairs[i][1]->tile_data);
      std::vector<cv::KeyPoint> a_tile_keypoints;
      cv::Mat a_tile_desc;
      std::vector<cv::KeyPoint> b_tile_keypoints;
      cv::Mat b_tile_desc;
      tile_a->compute_sift_keypoints2d_params(best_params, a_tile_keypoints, a_tile_desc, tile_b);
      tile_b->compute_sift_keypoints2d_params(best_params, b_tile_keypoints, b_tile_desc, tile_a);
      std::vector< cv::Point2f > filtered_match_points_a(0);
      std::vector< cv::Point2f > filtered_match_points_b(0);

      cv::Point2f relative_offset = this->sections[tile_a->section_id]->compute_tile_matches_pair(tile_a, tile_b,
        a_tile_keypoints, b_tile_keypoints,
        a_tile_desc, b_tile_desc,
        filtered_match_points_a,
        filtered_match_points_b, 5);
      tile_a->offset_x += relative_offset.x + (tile_b->x_start - tile_a->x_start);
      tile_a->offset_y += relative_offset.y + (tile_b->y_start - tile_a->y_start);

      cv::Mat vector = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
      float label = tile_a->error_tile_pair(tile_b) > thresh;
      if (label) {
        if (rand()%2) {
          tile_a->offset_x += (i%10)+1;
        } else {
          tile_a->offset_x -= (i%10)+1;
        }
        if (rand()%2) {
          tile_a->offset_y += (i%10)+1;
        } else {
          tile_a->offset_y -= (i%10)+1;
        }
        cv::Mat vector2 = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
        float prediction1 = rtree->predict(vector); // a correct one
        float prediction2 = rtree->predict(vector2); // a bad one
        if (prediction1  > .5) {
          __sync_fetch_and_add(&correct[(i%10)], 1);
        }
        if (prediction2 < .5) {
          __sync_fetch_and_add(&correct[(i%10)], 1);
        }
        __sync_fetch_and_add(&total[(i%10)], 2);

      }
 

      tile_a->release_2d_keypoints();
      tile_b->release_2d_keypoints();
      tile_a->release_full_image();
      tile_b->release_full_image();
      delete tile_a;
      delete tile_b;
    }
    for (int i = 0; i < 10; i++) {
      printf("%f, ",(float) correct[i] / total[i]);
    }
    printf("\n");
  }

  // neural net test
  if (true) {
    printf("neural net test\n");


    // setup the ann:
    int nfeatures;
    int batch_size = 100;
    printf("batch size = %d\n", batch_size);
    if (vector_mode == 1) {
      nfeatures = 1 + (4 * vector_grid_size * vector_grid_size);
      printf("using grids mean and stddev for vector\n");
    } else if (vector_mode == 2) {
      nfeatures = 1 + (vector_grid_size * vector_grid_size);
      printf("using grids corralation for vector\n");
    } else {
      printf("not implemented for this vector mode\n");
      return;
    }
    cv::Mat_<int> layers(2,1);
    int nclasses = 3;
    layers(0) = nfeatures;     // input
    //layers(1) = 3;     // hidden
    layers(1) = nclasses;      // positive negative and unknown
    cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
    ann->setLayerSizes(layers);
    ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);



    float thresh = .75;
    // find the best movement for each pair using the best params
    int total_correct[10] = {0};
    int total[10] = {0};
    for (int i = 0; i < trials/batch_size; i++) {
      cv::Mat labels = cv::Mat::zeros(batch_size*2, 3, CV_32F);
      cv::Mat data = cv::Mat::zeros(batch_size*2, nfeatures , CV_32F);
      cilk_for (int j = 0; j < batch_size ; j++) {
        Tile* tile_a = new Tile(tile_pairs[i*batch_size +j][0]->tile_data);
        Tile* tile_b = new Tile(tile_pairs[i*batch_size +j][1]->tile_data);
        std::vector<cv::KeyPoint> a_tile_keypoints;
        cv::Mat a_tile_desc;
        std::vector<cv::KeyPoint> b_tile_keypoints;
        cv::Mat b_tile_desc;
        tile_a->compute_sift_keypoints2d_params(best_params, a_tile_keypoints, a_tile_desc, tile_b);
        tile_b->compute_sift_keypoints2d_params(best_params, b_tile_keypoints, b_tile_desc, tile_a);
        std::vector< cv::Point2f > filtered_match_points_a(0);
        std::vector< cv::Point2f > filtered_match_points_b(0);

        cv::Point2f relative_offset = this->sections[tile_a->section_id]->compute_tile_matches_pair(tile_a, tile_b,
          a_tile_keypoints, b_tile_keypoints,
          a_tile_desc, b_tile_desc,
          filtered_match_points_a,
          filtered_match_points_b, 5);
        tile_a->offset_x += relative_offset.x + (tile_b->x_start - tile_a->x_start);
        tile_a->offset_y += relative_offset.y + (tile_b->y_start - tile_a->y_start);

        cv::Mat vector = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
        //std::cout << vector;
        float label = tile_a->error_tile_pair(tile_b) > thresh;
        if (label) {
          // a good example
          for (int k = 0; k < vector.cols; k++) {
            float val = vector.at<float>(k);
            data.at<float>(2*j, k) = val;
          }
          labels.at<float>(2*j, 0) = (float)1.0;
          if (rand()%2) {
            tile_a->offset_x += (j%10)+1;
          } else {
            tile_a->offset_x -= (j%10)+1;
          }
          if (rand()%2) {
            tile_a->offset_y += (j%10)+1;
          } else {
            tile_a->offset_y -= (j%10)+1;
          }
          cv::Mat vector2 = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
          // a bad example
          for (int k = 0; k < vector2.cols; k++) {
            data.at<float>(2*j+1, k) = vector2.at<float>(k);
          }
          labels.at<float>(2*j+1, 2) = (float)1.0;
        } else {
          // this is stupid but trying to fix an indexing error
          // unkown examples
          for (int k = 0; k < vector.cols; k++) {
            data.at<float>(2*j, k) = vector.at<float>(k);
          }
          labels.at<float>(2*j,1) = 1.0;
          for (int k = 0; k < vector.cols; k++) {
            data.at<float>(2*j+1, k) = vector.at<float>(k);
          }
          labels.at<float>(2*j+1,1) = (float)1.0;
        }
        tile_a->release_2d_keypoints();
        tile_b->release_2d_keypoints();
        tile_a->release_full_image();
        tile_b->release_full_image();
        delete tile_a;
        delete tile_b;
      }
      int correct[10] = {0};

      cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, labels);

      if (i > 0) {
        int total_batch[10] = {0};
        for (int j = 0; j < batch_size*2  ; j++) {
          cv::Mat pred_mat = cv::Mat::zeros(1, 3, CV_32F);
          int pred = ann->predict(data.row(j), pred_mat);
          int truth = 0;
          if (labels.at<float>(j, 0)) {
            truth = 0;
            total_batch[((j/2)%10)]++;
          } else if (labels.at<float>(j, 1)) {
            truth = 1;
            continue;
          } else if (labels.at<float>(j, 2)) {
            truth = 2;
            total_batch[((j/2)%10)]++;
          } else {
            printf("something BW doesn't understand about how anns works\n");
            return;
          }
          //printf("pred = %d, truth = %d\n", pred, truth);
          correct[((j/2)%10)] += (pred == truth);
          //printf("predicted = %d, truth = %d\n", pred, truth);
        }
        for (int in = 0; in < 10; in++) {
          total_correct[in] += correct[in];
          total[in] += total_batch[in];
        }
        // this prints out the accuracies on pixel errors 1 through 10
        printf("on iteration %d out of %d\nThis rounds accuracies are: ", i, trials/batch_size);
        for (int in = 0; in < 10; in++) {
          printf(" %f,",((float)correct[in])/ total_batch[in]);
        }
        printf("\nThe overall accuracies are: ");
        for (int in = 0; in < 10; in++) {
          printf(" %f,", ((float) total_correct[in])/(total[in]));
        }
        printf("\n\n");
        ann->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
      } else { // i == 0
        ann->train(tdata);
      }
    }


  }

   // no learning test test
  if (false) {

    float thresh = .75;

    int total[10] = {0};
    int correct[10] = {0};
    thresh = .7;
    cilk_for (int i = 0; i < trials; i++) {
      Tile* tile_a = new Tile(tile_pairs[i][0]->tile_data);
      Tile* tile_b = new Tile(tile_pairs[i][1]->tile_data);
      std::vector<cv::KeyPoint> a_tile_keypoints;
      cv::Mat a_tile_desc;
      std::vector<cv::KeyPoint> b_tile_keypoints;
      cv::Mat b_tile_desc;
      tile_a->compute_sift_keypoints2d_params(best_params, a_tile_keypoints, a_tile_desc, tile_b);
      tile_b->compute_sift_keypoints2d_params(best_params, b_tile_keypoints, b_tile_desc, tile_a);
      std::vector< cv::Point2f > filtered_match_points_a(0);
      std::vector< cv::Point2f > filtered_match_points_b(0);

      cv::Point2f relative_offset = this->sections[tile_a->section_id]->compute_tile_matches_pair(tile_a, tile_b,
        a_tile_keypoints, b_tile_keypoints,
        a_tile_desc, b_tile_desc,
        filtered_match_points_a,
        filtered_match_points_b, 5);
      tile_a->offset_x += relative_offset.x + (tile_b->x_start - tile_a->x_start);
      tile_a->offset_y += relative_offset.y + (tile_b->y_start - tile_a->y_start);

      cv::Mat vector = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
      float label = tile_a->error_tile_pair(tile_b) > thresh;
      if (label) {
        if (rand()%2) {
          tile_a->offset_x += (i%10)+1;
        } else {
          tile_a->offset_x -= (i%10)+1;
        }
        if (rand()%2) {
          tile_a->offset_y += (i%10)+1;
        } else {
          tile_a->offset_y -= (i%10)+1;
        }
        float label2 = tile_a->error_tile_pair(tile_b) > thresh;
        if (label2 < .5) {
          __sync_fetch_and_add(&correct[(i%10)], 1);
        }
        __sync_fetch_and_add(&total[(i%10)], 1);

      }


      tile_a->release_2d_keypoints();
      tile_b->release_2d_keypoints();
      tile_a->release_full_image();
      tile_b->release_full_image();
      delete tile_a;
      delete tile_b;
    }
    for (int i = 0; i < 10; i++) {
      printf("%f, ",(float) correct[i] / total[i]);
    }
    printf("\n"); 
  }
  
}

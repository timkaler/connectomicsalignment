#include "stack.hpp"
#include "stack_helpers.cpp"
#include "./othersift2.hpp"
#include "./stack_tile.cpp"
#include "./stack_section.cpp"


// Contains the initial code for mr stuff.

// BEGIN init functions
tfk::Stack::Stack(int base_section, int n_sections,
    std::string input_filepath, std::string output_dirpath) {
  this->base_section = base_section;
  this->n_sections = n_sections;
  this->input_filepath = input_filepath;
  this->output_dirpath = output_dirpath;
  this->ml_models.push_back(new MLAnn(10));
  printf("ml models for stack %p\n", &this->ml_models);
}

void tfk::Stack::init() {
  printf("Initializing the stack.\n");
  AlignData align_data;
  // Read the existing address book.
  std::fstream input(this->input_filepath, std::ios::in | std::ios::binary);
  if (!align_data.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse protocal buffer." << std::endl;
    exit(1);
  }
  // first deal with AlignData level
  if (align_data.has_mode()) {
    this->mode = align_data.mode();
  }

  if (align_data.has_output_dirpath()) {
    this->output_dirpath = align_data.output_dirpath();
  }

  if (align_data.has_base_section()) {
    this->base_section = align_data.base_section();
  }

  if (align_data.has_n_sections()) {
    this->n_sections = align_data.n_sections();
  }

  if (align_data.has_do_subvolume()) {
    this->do_subvolume = align_data.do_subvolume();
    this->min_x = align_data.min_x();
    this->min_y = align_data.min_y();
    this->max_x = align_data.max_x();
    this->max_y = align_data.max_y();
  }
  printf("got this far\n");
  // then do the section data
  for (int i = this->base_section; i < (this->base_section + this->n_sections); i++) {
    printf("doing section %d\n", i);
    SectionData section_data = align_data.sec_data(i);
    printf("doing section %d\n", i);
    printf("bounding box is %f %f %f %f\n", _bounding_box.first.x, _bounding_box.first.y, _bounding_box.second.x, _bounding_box.second.y);
    Section* sec = new Section(section_data, _bounding_box);
    printf("doing section %d\n", i);
    sec->section_id = this->sections.size();
    printf("doing section %d\n", i);
    this->sections.push_back(sec);
    // passing down the pointer to ml_models
    sec->ml_models = &this->ml_models;
    //printf("ml models for section %p\n", sec->ml_models);
  }
}
// END init functions

// BEGIN Test functions
void tfk::Stack::test_io() {
  for (int i = 0; i < this->sections.size(); i++) {
    Section* section = this->sections[i];
    cilk_for (int j = 0; j < section->tiles.size(); j++) {
      Tile* tile = section->tiles[j];
      cv::Mat mat = tile->get_tile_data(Resolution::FILEIOTEST);
      mat.release();
      printf("tile %d of section %d\n", j, i);
    }
  }
}

void tfk::Stack::compute_on_tile_neighborhood(tfk::Section* section, tfk::Tile* tile) {
  //int distance = 2;
  std::vector<Tile*> neighbors = section->get_all_close_tiles(tile);
  std::set<Tile*> active_set;
  active_set.insert(tile);
  for (int i = 0; i < neighbors.size(); i++) {
    active_set.insert(neighbors[i]);
  }
  for (int j = 0; j < 5000; j++) {
    tile->local2DAlignUpdateLimited(&active_set);
    for (int i = 0; i < neighbors.size(); i++) {
      neighbors[i]->local2DAlignUpdateLimited(&active_set);
    }
  }
}
// END Test functions

// BEGIN Alignment algorithms.
void tfk::Stack::align_3d() {
  for (int i = 0; i < this->sections.size(); i++) {
    cilk_spawn this->sections[i]->construct_triangles();
  }
  cilk_sync;

  std::vector<Section*> filtered_sections;
  for (int i = 0; i < this->sections.size(); i++) {
    if (this->sections[i]->real_section_id == 28 || 
        this->sections[i]->real_section_id == 29 || 
        this->sections[i]->real_section_id == 31) continue;
    filtered_sections.push_back(this->sections[i]);
  }
  this->sections = filtered_sections;

  cilk_spawn this->sections[0]->align_3d(this->sections[0]);

  for (int i = 1; i < this->sections.size(); i++) {
    cilk_spawn this->sections[i]->align_3d(this->sections[i-1]);
  }
  cilk_sync; 

  // section i is aligned to section i-1;
  for (int i = 1; i < this->sections.size(); i++) {
    Section* sec = this->sections[i];
    int j = i-1;
    for (int k = 0; k < sec->triangle_mesh->mesh->size(); k++) {
      (*sec->triangle_mesh->mesh)[k] = this->sections[j]->elastic_transform((*sec->triangle_mesh->mesh)[k]);
    }
  }
  printf("Done with align 3d\n");
}

void tfk::Stack::align_2d() {
  for (int i = 0; i < this->sections.size(); i++) {
    this->sections[i]->align_2d();
    printf("ML Correct positive = %d, correct negatives = %d, false positives = %d, false negative = %d\n", this->ml_models[0]->ml_correct_pos, this->ml_models[0]->ml_correct_neg, this->ml_models[0]->ml_fp, this->ml_models[0]->ml_fn);
    this->ml_models[0]->ml_correct_pos = 0;
    this->ml_models[0]->ml_correct_neg = 0;
    this->ml_models[0]->ml_fp = 0;
    this->ml_models[0]->ml_fn = 0;
  }
  return;
  //int count = 0;
  int j = 0;
  int i = 0;
  while (j < this->sections.size()) {
    j += 4;
    if (j >= this->sections.size()) j = this->sections.size();

    for (; i < j; i++) {
       cilk_spawn this->sections[i]->align_2d();
    }
    cilk_sync;
  }
}

#include "stack_learning.cpp"

//<<<<<<< HEAD
//=======
//  std::vector<Tile*[2]> tile_pairs(trials);
//
//
//  // choose all the random pairs of overlapping tiles
//  cilk_for (int i = 0; i < trials; i++) {
//    // pick a random section
//    Section* sec = this->sections[rand()%this->n_sections];
//    // pick a random tile in that section
//    Tile* tile_a = sec->tiles[rand()%sec->n_tiles];
//    std::vector<int> neighbor_ids = sec->get_all_close_tiles(tile_a->tile_id);
//    //pick a random neighbor of that tile
//    while (neighbor_ids.size() == 0) {
//      tile_a = sec->tiles[rand()%sec->n_tiles];
//      neighbor_ids = sec->get_all_close_tiles(tile_a->tile_id);
//    }
//    Tile* tile_b = sec->tiles[neighbor_ids[rand()%neighbor_ids.size()]];
//    tile_pairs[i][0] = tile_a;
//    tile_pairs[i][1] = tile_b;
//  }
//  //printf("found pairs for testing\n");
//
//  params best_params;
//  best_params.num_features = 1;
//  best_params.num_octaves = 6;
//  best_params.contrast_threshold = .015;//CONTRAST_THRESH;
//  best_params.edge_threshold = 10;//EDGE_THRESH_2D;
//  best_params.sigma = 1.6;//1.6;
//  best_params.scale_x = 1.0;
//  best_params.scale_y = 1.0;
//  best_params.res = FULL;
//
//  //random forest test
//  if (false) {
//    printf("random forest test\n");
//    int training_runs = 9*trials/10;
//
//    cv::Mat data;
//    if (vector_mode == 2) {
//      data = cv::Mat::zeros((training_runs)*2, 1 + (vector_grid_size * vector_grid_size), CV_32F);
//    } else {
//      printf("not implemented for this vector mode\n");
//      return;
//    }
//
//    cv::Mat labels = cv::Mat::zeros((training_runs)*2, 1, CV_32F);
//
//    float thresh = .75;
//    // find the best movement for each pair using the best params
//    cilk_for (int i = 0; i < training_runs; i++) {
//      Tile* tile_a = new Tile(tile_pairs[i][0]->tile_data);
//      Tile* tile_b = new Tile(tile_pairs[i][1]->tile_data);
//      std::vector<cv::KeyPoint> a_tile_keypoints;
//      cv::Mat a_tile_desc;
//      std::vector<cv::KeyPoint> b_tile_keypoints;
//      cv::Mat b_tile_desc;
//      tile_a->compute_sift_keypoints2d_params(best_params, a_tile_keypoints, a_tile_desc, tile_b);
//      tile_b->compute_sift_keypoints2d_params(best_params, b_tile_keypoints, b_tile_desc, tile_a);
//      std::vector< cv::Point2f > filtered_match_points_a(0);
//      std::vector< cv::Point2f > filtered_match_points_b(0);
//
//      cv::Point2f relative_offset = this->sections[tile_a->section_id]->compute_tile_matches_pair(tile_a, tile_b,
//        a_tile_keypoints, b_tile_keypoints,
//        a_tile_desc, b_tile_desc,
//        filtered_match_points_a,
//        filtered_match_points_b, 5);
//      tile_a->offset_x += relative_offset.x + (tile_b->x_start - tile_a->x_start);
//      tile_a->offset_y += relative_offset.y + (tile_b->y_start - tile_a->y_start);
//
//      cv::Mat vector = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
//      float label = tile_a->error_tile_pair(tile_b) > thresh;
//
//      if (label) {
//        // a good example
//        for (int j = 0; j < vector.cols; j++) {
//          data.at<float>(2*i, j) = vector.at<float>(j);
//        }
//        labels.at<float>(2*i) = label;
//        if (rand()%2) {
//          tile_a->offset_x += (i%10)+1;
//        } else {
//          tile_a->offset_x -= (i%10)+1;
//        }
//        if (rand()%2) {
//          tile_a->offset_y += (i%10)+1;
//        } else {
//          tile_a->offset_y -= (i%10)+1;
//        }
//        cv::Mat vector2 = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
//        // a bad example
//        for (int j = 0; j < vector2.cols; j++) {
//          data.at<float>(2*i+1, j) = vector2.at<float>(j);
//        }
//        labels.at<float>(2*i+1) = (float)0;
//      } else {
//        // this is stupid but trying to fix an indexing error
//        // unkown examples
//        for (int j = 0; j < vector.cols; j++) {
//          data.at<float>(2*i, j) = vector.at<float>(j);
//        }
//        labels.at<float>(2*i) = (float).5;
//        for (int j = 0; j < vector.cols; j++) {
//          data.at<float>(2*i+1, j) = vector.at<float>(j);
//        }
//        labels.at<float>(2*i+1) = (float).5;
//      }
//      tile_a->release_2d_keypoints();
//      tile_b->release_2d_keypoints();
//      tile_a->release_full_image();
//      tile_b->release_full_image();
//      delete tile_a;
//      delete tile_b;
//    }
//    static cv::Ptr<cv::ml::RTrees> rtree = cv::ml::RTrees::create();
//    int correct[10] = {0};
//    //printf(" %d, %d, %d, %d\n",data.rows, data.cols, labels.rows, labels.cols );
//    printf("starting training\n");
//    static cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, labels);
//    rtree->train(tdata);
//    printf("done training\n");
//    //rtree->predict(cv::Mat::zeros(1, 26, CV_32S));
//    int total[10] = {0};
//    cilk_for (int i = training_runs; i < trials; i++) {
//      Tile* tile_a = new Tile(tile_pairs[i][0]->tile_data);
//      Tile* tile_b = new Tile(tile_pairs[i][1]->tile_data);
//      std::vector<cv::KeyPoint> a_tile_keypoints;
//      cv::Mat a_tile_desc;
//      std::vector<cv::KeyPoint> b_tile_keypoints;
//      cv::Mat b_tile_desc;
//      tile_a->compute_sift_keypoints2d_params(best_params, a_tile_keypoints, a_tile_desc, tile_b);
//      tile_b->compute_sift_keypoints2d_params(best_params, b_tile_keypoints, b_tile_desc, tile_a);
//      std::vector< cv::Point2f > filtered_match_points_a(0);
//      std::vector< cv::Point2f > filtered_match_points_b(0);
//
//      cv::Point2f relative_offset = this->sections[tile_a->section_id]->compute_tile_matches_pair(tile_a, tile_b,
//        a_tile_keypoints, b_tile_keypoints,
//        a_tile_desc, b_tile_desc,
//        filtered_match_points_a,
//        filtered_match_points_b, 5);
//      tile_a->offset_x += relative_offset.x + (tile_b->x_start - tile_a->x_start);
//      tile_a->offset_y += relative_offset.y + (tile_b->y_start - tile_a->y_start);
//
//      cv::Mat vector = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
//      float label = tile_a->error_tile_pair(tile_b) > thresh;
//      if (label) {
//        if (rand()%2) {
//          tile_a->offset_x += (i%10)+1;
//        } else {
//          tile_a->offset_x -= (i%10)+1;
//        }
//        if (rand()%2) {
//          tile_a->offset_y += (i%10)+1;
//        } else {
//          tile_a->offset_y -= (i%10)+1;
//        }
//        cv::Mat vector2 = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
//        float prediction1 = rtree->predict(vector); // a correct one
//        float prediction2 = rtree->predict(vector2); // a bad one
//        if (prediction1  > .5) {
//          __sync_fetch_and_add(&correct[(i%10)], 1);
//        }
//        if (prediction2 < .5) {
//          __sync_fetch_and_add(&correct[(i%10)], 1);
//        }
//        __sync_fetch_and_add(&total[(i%10)], 2);
//
//      }
// 
//
//      tile_a->release_2d_keypoints();
//      tile_b->release_2d_keypoints();
//      tile_a->release_full_image();
//      tile_b->release_full_image();
//      delete tile_a;
//      delete tile_b;
//    }
//    for (int i = 0; i < 10; i++) {
//      printf("%f, ",(float) correct[i] / total[i]);
//    }
//    printf("\n");
//  }
//
//  // neural net test
//  if (true) {
//    printf("neural net test\n");
//
//
//    // setup the ann:
//    int nfeatures;
//    int batch_size = 100;
//    printf("batch size = %d\n", batch_size);
//    if (vector_mode == 1) {
//      nfeatures = 1 + (4 * vector_grid_size * vector_grid_size);
//      printf("using grids mean and stddev for vector\n");
//    } else if (vector_mode == 2) {
//      nfeatures = 1 + (vector_grid_size * vector_grid_size);
//      printf("using grids corralation for vector\n");
//    } else {
//      printf("not implemented for this vector mode\n");
//      return;
//    }
//    cv::Mat_<int> layers(2,1);
//    int nclasses = 3;
//    layers(0) = nfeatures;     // input
//    //layers(1) = 3;     // hidden
//    layers(1) = nclasses;      // positive negative and unknown
//    cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
//    ann->setLayerSizes(layers);
//    ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
//
//
//
//    float thresh = .75;
//    // find the best movement for each pair using the best params
//    int total_correct[10] = {0};
//    int total[10] = {0};
//    for (int i = 0; i < trials/batch_size; i++) {
//      cv::Mat labels = cv::Mat::zeros(batch_size*2, 3, CV_32F);
//      cv::Mat data = cv::Mat::zeros(batch_size*2, nfeatures , CV_32F);
//      cilk_for (int j = 0; j < batch_size ; j++) {
//        Tile* tile_a = new Tile(tile_pairs[i*batch_size +j][0]->tile_data);
//        Tile* tile_b = new Tile(tile_pairs[i*batch_size +j][1]->tile_data);
//        std::vector<cv::KeyPoint> a_tile_keypoints;
//        cv::Mat a_tile_desc;
//        std::vector<cv::KeyPoint> b_tile_keypoints;
//        cv::Mat b_tile_desc;
//        tile_a->compute_sift_keypoints2d_params(best_params, a_tile_keypoints, a_tile_desc, tile_b);
//        tile_b->compute_sift_keypoints2d_params(best_params, b_tile_keypoints, b_tile_desc, tile_a);
//        std::vector< cv::Point2f > filtered_match_points_a(0);
//        std::vector< cv::Point2f > filtered_match_points_b(0);
//
//        cv::Point2f relative_offset = this->sections[tile_a->section_id]->compute_tile_matches_pair(tile_a, tile_b,
//          a_tile_keypoints, b_tile_keypoints,
//          a_tile_desc, b_tile_desc,
//          filtered_match_points_a,
//          filtered_match_points_b, 5);
//        tile_a->offset_x += relative_offset.x + (tile_b->x_start - tile_a->x_start);
//        tile_a->offset_y += relative_offset.y + (tile_b->y_start - tile_a->y_start);
//
//        cv::Mat vector = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
//        //std::cout << vector;
//        float label = tile_a->error_tile_pair(tile_b) > thresh;
//        if (label) {
//          // a good example
//          for (int k = 0; k < vector.cols; k++) {
//            float val = vector.at<float>(k);
//            data.at<float>(2*j, k) = val;
//          }
//          labels.at<float>(2*j, 0) = (float)1.0;
//          if (rand()%2) {
//            tile_a->offset_x += (j%10)+1;
//          } else {
//            tile_a->offset_x -= (j%10)+1;
//          }
//          if (rand()%2) {
//            tile_a->offset_y += (j%10)+1;
//          } else {
//            tile_a->offset_y -= (j%10)+1;
//          }
//          cv::Mat vector2 = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
//          // a bad example
//          for (int k = 0; k < vector2.cols; k++) {
//            data.at<float>(2*j+1, k) = vector2.at<float>(k);
//          }
//          labels.at<float>(2*j+1, 2) = (float)1.0;
//        } else {
//          // this is stupid but trying to fix an indexing error
//          // unkown examples
//          for (int k = 0; k < vector.cols; k++) {
//            data.at<float>(2*j, k) = vector.at<float>(k);
//          }
//          labels.at<float>(2*j,1) = 1.0;
//          for (int k = 0; k < vector.cols; k++) {
//            data.at<float>(2*j+1, k) = vector.at<float>(k);
//          }
//          labels.at<float>(2*j+1,1) = (float)1.0;
//        }
//        tile_a->release_2d_keypoints();
//        tile_b->release_2d_keypoints();
//        tile_a->release_full_image();
//        tile_b->release_full_image();
//        delete tile_a;
//        delete tile_b;
//      }
//      int correct[10] = {0};
//
//      cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, labels);
//
//      if (i > 0) {
//        int total_batch[10] = {0};
//        for (int j = 0; j < batch_size*2  ; j++) {
//          cv::Mat pred_mat = cv::Mat::zeros(1, 3, CV_32F);
//          int pred = ann->predict(data.row(j), pred_mat);
//          int truth = 0;
//          if (labels.at<float>(j, 0)) {
//            truth = 0;
//            total_batch[((j/2)%10)]++;
//          } else if (labels.at<float>(j, 1)) {
//            truth = 1;
//            continue;
//          } else if (labels.at<float>(j, 2)) {
//            truth = 2;
//            total_batch[((j/2)%10)]++;
//          } else {
//            printf("something BW doesn't understand about how anns works\n");
//            return;
//          }
//          //printf("pred = %d, truth = %d\n", pred, truth);
//          correct[((j/2)%10)] += (pred == truth);
//          //printf("predicted = %d, truth = %d\n", pred, truth);
//        }
//        for (int in = 0; in < 10; in++) {
//          total_correct[in] += correct[in];
//          total[in] += total_batch[in];
//        }
//        // this prints out the accuracies on pixel errors 1 through 10
//        printf("on iteration %d out of %d\nThis rounds accuracies are: ", i, trials/batch_size);
//        for (int in = 0; in < 10; in++) {
//          printf(" %f,",((float)correct[in])/ total_batch[in]);
//        }
//        printf("\nThe overall accuracies are: ");
//        for (int in = 0; in < 10; in++) {
//          printf(" %f,", ((float) total_correct[in])/(total[in]));
//        }
//        printf("\n\n");
//        ann->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
//      } else { // i == 0
//        ann->train(tdata);
//      }
//    //std::cout << ann->getWeights(0);
//    //std::cout << ann->getWeights(1);
//    //printf("\n");
//    }
//
//
//  }
//
//   // no learning test test
//  if (false) {
//
//    float thresh = .75;
//
//    int total[10] = {0};
//    int correct[10] = {0};
//    thresh = .7;
//    cilk_for (int i = 0; i < trials; i++) {
//      Tile* tile_a = new Tile(tile_pairs[i][0]->tile_data);
//      Tile* tile_b = new Tile(tile_pairs[i][1]->tile_data);
//      std::vector<cv::KeyPoint> a_tile_keypoints;
//      cv::Mat a_tile_desc;
//      std::vector<cv::KeyPoint> b_tile_keypoints;
//      cv::Mat b_tile_desc;
//      tile_a->compute_sift_keypoints2d_params(best_params, a_tile_keypoints, a_tile_desc, tile_b);
//      tile_b->compute_sift_keypoints2d_params(best_params, b_tile_keypoints, b_tile_desc, tile_a);
//      std::vector< cv::Point2f > filtered_match_points_a(0);
//      std::vector< cv::Point2f > filtered_match_points_b(0);
//
//      cv::Point2f relative_offset = this->sections[tile_a->section_id]->compute_tile_matches_pair(tile_a, tile_b,
//        a_tile_keypoints, b_tile_keypoints,
//        a_tile_desc, b_tile_desc,
//        filtered_match_points_a,
//        filtered_match_points_b, 5);
//      tile_a->offset_x += relative_offset.x + (tile_b->x_start - tile_a->x_start);
//      tile_a->offset_y += relative_offset.y + (tile_b->y_start - tile_a->y_start);
//
//      cv::Mat vector = tile_a->get_feature_vector(tile_b, vector_grid_size, vector_mode);
//      float label = tile_a->error_tile_pair(tile_b) > thresh;
//      if (label) {
//        if (rand()%2) {
//          tile_a->offset_x += (i%10)+1;
//        } else {
//          tile_a->offset_x -= (i%10)+1;
//        }
//        if (rand()%2) {
//          tile_a->offset_y += (i%10)+1;
//        } else {
//          tile_a->offset_y -= (i%10)+1;
//        }
//        float label2 = tile_a->error_tile_pair(tile_b) > thresh;
//        if (label2 < .5) {
//          __sync_fetch_and_add(&correct[(i%10)], 1);
//        }
//        __sync_fetch_and_add(&total[(i%10)], 1);
//
//      }
//
//
//      tile_a->release_2d_keypoints();
//      tile_b->release_2d_keypoints();
//      tile_a->release_full_image();
//      tile_b->release_full_image();
//      delete tile_a;
//      delete tile_b;
//    }
//    for (int i = 0; i < 10; i++) {
//      printf("%f, ",(float) correct[i] / total[i]);
//    }
//    printf("\n"); 
//  }
//
//}
//>>>>>>> cdd2df2c332cdad2a05d9e719c4016eda32d01c9

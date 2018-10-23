#include "mlbase.hpp"

namespace tfk {

// general MLBAse functions
  //static float FP_PENALTY = -0.2;
  static float FP_PENALTY = 0.0;
  MLBase::MLBase(int num_features, std::string saved_model) {
    this->mutex = new std::recursive_mutex();
  }

  float MLBase::feature_dist(std::vector<float> a, std::vector<float> b) {
    float dist = 0.0;
    for (int i = 0; i < a.size(); i++) {
      dist += std::sqrt((a[i]-b[i])*(a[i]-b[i]));
    }
    dist = dist/a.size();
    return dist;
  }
  void MLBase::flush_train_buffer() {
    mutex->lock();
    if (new_data.size() == 0) return;
    float sum_error = 0.0;
    for (int i = 0; i < new_data.size(); i++) {
      if (!new_labels[i]) {
        sum_error += new_errors[i];
      }
    }
    float mean_error = sum_error / new_data.size();
    float var_error = 0.0;
    for (int i = 0; i < new_data.size(); i++) {
      if (!new_labels[i]) {
        var_error += std::pow(mean_error-new_errors[i],2);
      }
    }
    var_error = var_error/new_data.size();
    float stdev = std::sqrt(var_error);

    for (int i = 0; i < new_data.size(); i++) {
      if (new_labels[i] == true) {
        old_data.push_back(new_data[i]);
        old_labels.push_back(new_labels[i]);
      } else if (new_errors[i] > mean_error + 2*stdev) {
        old_data.push_back(new_data[i]);
        old_labels.push_back(new_labels[i]);
      }
    }

    new_data.clear();
    new_labels.clear();
    //new_errors.clear();
    mutex->unlock();
  }


  //TODO(wheatman) should keep old data in train buffer until it is used


  // old data, old labels.
  // new data, new labels.
  // move the new data/labels into old labels. Then balance the positive and negative examples.
  void MLBase::balance_and_flush_train_buffer() {
    mutex->lock();
    printf("There are %zu %zu old data/labels and %zu %zu new data /labels\n",
           old_data.size(), old_labels.size(), new_data.size(), new_labels.size());

    if (new_data.size() == 0) {
      mutex->unlock();
      return;
    }
    float sum_error = 0.0;
    for (int i = 0; i < new_data.size(); i++) {
      if (!new_labels[i]) {
        sum_error += new_errors[i];
      }
    }
    float mean_error = sum_error / new_data.size();
    float var_error = 0.0;
    int _negative_count = 0;
    for (int i = 0; i < new_data.size(); i++) {
      if (!new_labels[i]) {
        var_error += std::pow(mean_error-new_errors[i],2);
        _negative_count++;
      }
    }
    if (_negative_count > 0) {
      var_error = var_error/_negative_count;
    }
    float stdev = std::sqrt(var_error);

    std::vector<std::vector<float> > tmp_data;
    std::vector<float> tmp_labels;

    for (int i = 0; i < new_data.size(); i++) {
      if (new_labels[i] == true) {
        tmp_data.push_back(new_data[i]);
        tmp_labels.push_back(new_labels[i]);
      } else if (true || new_errors[i] > mean_error + 2*stdev) {
        tmp_data.push_back(new_data[i]);
        tmp_labels.push_back(new_labels[i]);
      }
    }


    std::vector<std::vector<float> > tmp_data2;
    std::vector<float> tmp_labels2;


    for (int i = 0; i < tmp_data.size(); i++) {
      bool add = true;
      /*
      for (int j = i+1; j < tmp_data.size(); j++) {
        float dist = feature_dist(tmp_data[i], tmp_data[j]);
        if (dist < 0.001) {
          add = false;
          break;
        }
      }
      */
      if (add) {
        tmp_data2.push_back(tmp_data[i]);
        tmp_labels2.push_back(tmp_labels[i]);
      }
    }


    std::vector<std::vector<float> > tmp_data3;
    std::vector<float> tmp_labels3;

    // now deduplicate with old data
    for (int i = 0; i < tmp_data2.size(); i++) {
      //bool add = true;
      /*
      for (int j = 0; j < old_data.size(); j++) {
        float dist = feature_dist(tmp_data2[i], old_data[j]);
        if (dist < 0.001) {
          // change the label of old_data.
          old_labels[j] = tmp_labels2[i];
          add = false;
        }
      }
      */
      tmp_data3.push_back(tmp_data2[i]);
      tmp_labels3.push_back(tmp_labels2[i]);
    }


    new_data = tmp_data3;
    new_labels = tmp_labels3;


    printf("There are %zu %zu old data/labels and %zu %zu new data /labels\n",
           old_data.size(), old_labels.size(), new_data.size(), new_labels.size());
    //new_data.clear();
    //new_labels.clear();
    //new_errors.clear();


 
    //for (int i = 0; i < new_data.size(); i++) {
    //  old_data.push_back(new_data[i]);
    //  old_labels.push_back(new_labels[i]);
    //}

    static float MAX_IMBALANCE_RATIO = 500000.0;

    int positive_count = 0;
    int negative_count = 0;
    for (int i = 0; i < new_labels.size(); i++) {
      float label = new_labels[i];
      bool positive = label > 0.5;
      if (positive) {
        positive_count++;
      } else {
        negative_count++;
      }
    }

    int baseline_size = std::min(positive_count, negative_count);

    bool label_to_remove = false;
    float remove_prob = 0.0;

    if (positive_count > baseline_size*MAX_IMBALANCE_RATIO) {
      //need to remove positive examples to keep balance.
      printf("Balance due to positive count %d, baseline size %d\n", positive_count, baseline_size);
      int num_to_remove = positive_count - baseline_size*MAX_IMBALANCE_RATIO + 1;
      remove_prob = num_to_remove*1.0/positive_count;
      label_to_remove = true;
    } else if (negative_count > baseline_size*MAX_IMBALANCE_RATIO) {
      // need to remove negative examples to keep balance.
      printf("Balance due to negative count %d, baseline size %d\n", negative_count, baseline_size);
      int num_to_remove = negative_count - baseline_size*MAX_IMBALANCE_RATIO + 1;
      label_to_remove = false;
      remove_prob = num_to_remove*1.0/negative_count;
    }

      std::mt19937 gen(42);
      std::uniform_real_distribution<> dis(0.0,1.0);
      for (int i = 0; i < new_labels.size(); i++) {
        bool label = new_labels[i] > 0.5;
        if (label == label_to_remove && dis(gen) < remove_prob) {
        } else {
          old_labels.push_back(new_labels[i]);
          old_data.push_back(new_data[i]);
        }
      }
    new_data.clear();
    new_labels.clear();


    mutex->unlock();
    //int target_positives = fraction_positive_examples * new_labels.size();
    //int target_negatives = new_labels.size() - target_positives;
    //printf("we have %d postive examples and %d negative examples\n",target_positives, target_negatives);
    //std::vector<int> indx;
    //for( int i = 0; i <= new_labels.size(); i++ ) {
    //  indx.push_back( i );
    //}
    //std::random_shuffle(indx.begin(), indx.end());


    //for (int i = 0; i < new_data.size(); i++) {
    //  std::vector<float> data = new_data[indx[i]];
    //  float label = new_labels[indx[i]];
    //  if (label && target_positives > 0) {
    //    old_data.push_back(data);
    //    old_labels.push_back(label);
    //    target_positives--;
    //  } else if (!label && target_negatives > 0) {
    //    old_data.push_back(data);
    //    old_labels.push_back(label);
    //    target_negatives--;
    //  }
    //}
    //new_data.clear();
    //new_labels.clear();
  }

  void MLBase::clear_train_buffer() {
    mutex->lock();
    new_data.clear();
    new_labels.clear();
    mutex->unlock();
  }

  void MLBase::clear_saved_buffer() {
    //old_data.clear();
    //old_labels.clear();
  }

  void MLBase::add_training_example(std::vector<float> new_vector, float new_label, float error) {
    mutex->lock();
    ASSERT(new_vector.size() == size_of_feature_vector);
    new_data.push_back(new_vector);
    new_labels.push_back(new_label);
    new_errors.push_back(error);
    num_positive_examples += new_label > .5;
    num_negative_examples += new_label <= .5;
    mutex->unlock();
  }


  bool MLBase::predict(std::vector<float> vec) {
    //return true;
    //return true;
    //return false;
    //return false;
    //return true;
    //if (!trained) {
    //  return true;
    //}
    //printf("predict was called\n");
    mutex->lock();
    cv:: Mat mat_vec = cv::Mat::zeros(1, size_of_feature_vector,CV_32F);
    for (int i = 0; i < size_of_feature_vector; i++) {
        //printf("inside predict value I'm seeing is %f\n", vec[i]);
        mat_vec.at<float>(0,i) = vec[i];
    }
        cv::Mat results = cv::Mat::zeros(1, 1, CV_32F);
        model->predict(mat_vec, results);
        bool ret = results.at<float>(0) > 0.5 +FP_PENALTY;
    //std::cout << mat_vec <<std::endl; 
    //bool ret = model->predict(mat_vec);
    mutex->unlock();
    //if (!ret) printf("guess ml was false inside predict\n");
    return ret;
  }

  void MLBase::enable_training() {
    mutex->lock();
    training_active = true;
    mutex->unlock();
  }

  void MLBase::disable_training() {
    mutex->lock();
    training_active = false;
    mutex->unlock();
  }


// spcific to ANN
  MLAnn::MLAnn(int num_features, std::string saved_model) : MLBase(num_features, saved_model) {
    cv::Ptr<cv::ml::RTrees> ann_model = cv::ml::RTrees::create();

    //cv::Mat_<int> layers(4,1);
    //layers(0) = num_features;     // input
    //layers(1) = num_features;      // positive negative and unknown
    //layers(2) = num_features/2 + 1;      // positive negative and unknown
    ////layers(2) = num_features/8 + 2;      // positive negative and unknown
    //layers(3) = 2;      // positive negative and unknown
    //ann_model->setLayerSizes(layers);
    ////ann_model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM  , 1, 1);
    //ann_model->setActivationFunction(cv::ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM  , 1, 1);
    model = ann_model;
    this->ann_model = ann_model;
    size_of_feature_vector = num_features;
    if (!saved_model.empty()) {
      training_active = false;
      this->load(saved_model, true);
    }
  }


  // these next two are not very general
  // probably should do something like move the data to old_data and be able to recalcuate from the start with all the data
  // this would be useul for the case where we can't do reinforcement learning
  void MLAnn::train(bool reinforcement) {
    mutex->lock();
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::Type::COUNT, 128, 1e-12);
    if (training_active) {
      clear_saved_buffer();
      balance_and_flush_train_buffer();
      int new_training_examples = old_data.size();
      cv::Mat labels = cv::Mat::zeros(new_training_examples, 1, CV_32F);
      cv::Mat weights = cv::Mat::zeros(new_training_examples,1, CV_32F);
      cv::Mat data = cv::Mat::zeros(new_training_examples, size_of_feature_vector , CV_32F);
      int count_1 = 0;
      int count_2 = 0;



      for (int i = 0; i < new_training_examples; i++) {
          if (old_labels[i]) {
              count_1++;
          } else {
              count_2++;
          }
      }

      float weight_1 = 1.0;//(1.0*count_2/count_2;
      float weight_2 = 0.25*(count_1*1.0)/count_2;


      for (int i = 0; i < new_training_examples; i++) {
          if (old_labels[i]) {
              labels.at<float>(i, 0) = 1;
              weights.at<float>(i,0) = weight_1;
              //count_1++;
          } else {
              labels.at<float>(i, 0) = 0;
              //weights.at<float>(i,0) = 10000.0;//10000.0;
              weights.at<float>(i,0) = /*std::min(10.0f, new_errors[i]) **/ weight_2;//10000.0;
              //count_2++;
          }
          for (int j = 0; j < size_of_feature_vector; j++) {
              data.at<float>(i, j) = old_data[i][j];
          }
      }
      printf("count 1 %d, count 2 %d\n", count_1, count_2);
      cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, labels,
          cv::noArray(), cv::noArray(), weights);
      tdata->setTrainTestSplitRatio(.75, true);
      //if (reinforcement && trained) {
      //    model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS | cv::ml::ANN_MLP::NO_INPUT_SCALE);
      //} else {
      printf("training the model\n");
        //ann_model->setTrainMethod(0, 0.1,0.1);
        ann_model->setTermCriteria(term_crit);
        //ann_model->setMaxDepth(4);
        //ann_model->setMinSampleCount(2);
        ann_model->setCalculateVarImportance(true);
        model->train(tdata);
        printf("after training\n");

        for (int i = 0; i < new_training_examples; i++) {
          bool pred = predict(old_data[i]);
          bool actual = old_labels[i];
          if (pred != actual) {
            if (actual) {
              //printf("false negative dist: %f\n", new_errors[i]);
            } else {
              printf("false positive dist: %f\n", new_errors[i]);
            }
          }
        }

        //model->train(tdata);
          /*
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
          
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);           
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);

           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
           model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
          */
      printf("finished training the model\n");
           
      //}
      clear_saved_buffer();
      trained = true;


      int tp = 0;
      int tn = 0;
      int fp = 0;
      int fn = 0;
      std::vector<std::vector<float> > filtered_data;
      std::vector<float> filtered_labels;
      // check error on training.
      std::vector<float> pos_preds;
      std::vector<float> neg_preds;

      cv::Mat trainSamples = tdata->getTrainSamples();
      cv::Mat trainLabels = tdata->getTrainResponses();
      printf("samples rows %d, samples cols %d, label rows %d, label cols %d\n\n",trainSamples.rows, trainSamples.cols, trainLabels.rows, trainLabels.cols );
      // for training set
      for (int i = 0; i < trainSamples.rows; i++) {
        cv::Mat results = cv::Mat::zeros(1, 1, CV_32F);
        model->predict(trainSamples.row(i), results);
        //bool prediction = results.at<float>(1) > results.at<float>(0)+FP_PENALTY;
        std::vector<float> vec;
        for (int j = 0; j < trainSamples.cols; j++) {
          vec.push_back(trainSamples.at<float>(i,j));
        }
        bool prediction = this->predict(vec);

        //bool prediction = results.at<float>(0) > 0.5 + FP_PENALTY;
        //printf();
        float pred_f = (results.at<float>(0) + 1.716) / (results.at<float>(0)+1.716);
        bool actual = trainLabels.at<float>(i,0) > 0.5 + FP_PENALTY;
        if (prediction == actual) {
          filtered_data.push_back(old_data[i]);
          filtered_labels.push_back(old_labels[i]);
          if (prediction) {
            tp++;
            pos_preds.push_back(pred_f);
          } else {
            tn++;
            neg_preds.push_back(pred_f);
          }
        } else {
          if (prediction) {
            //printf("dist of false positive is %f\n", new_errors[i]);
            fp++;
            neg_preds.push_back(pred_f);
          } else {
            fn++;
            //printf("dist of false negative is %f\n", new_errors[i]);
            pos_preds.push_back(pred_f);
          }
        } 
      }
      FILE * pFile;
      pFile = fopen ("for_hist_training.csv","w");
      for (int i = 0; i < pos_preds.size(); i++) {
        fprintf(pFile, "%.2f, ", pos_preds[i]);
      }
      fprintf(pFile, "\n");
      for (int i = 0; i < neg_preds.size(); i++) {
        fprintf(pFile, "%.2f, ", neg_preds[i]);
      }
      fprintf(pFile, "\n");
      if (old_labels.size() == 0) {
        printf("no training examples\n");
      } else {
        printf("After training with %d data points, accuracy = %f%%, true positives: %d, false positives: %d, true negatives: %d, false negatives: %d\n", trainSamples.rows, 100.0*(tp+tn)/ trainSamples.rows, tp, fp, tn, fn);
      }
      tp = 0;
      tn = 0;
      fp = 0;
      fn = 0;
      // check error on training.
      pos_preds.clear();
      neg_preds.clear();
      cv::Mat testSamples = tdata->getTestSamples();
      cv::Mat testLabels = tdata->getTestResponses();
      FILE * pFile2;
      pFile2 = fopen ("data_ml_both_pins.csv","w");

      // for test set
      for (int i = 0; i < testSamples.rows; i++) {
        cv::Mat results = cv::Mat::zeros(1, 1, CV_32F);
        model->predict(testSamples.row(i), results);

        std::vector<float> vec;
        for (int j = 0; j < testSamples.cols; j++) {
          vec.push_back(testSamples.at<float>(i,j));
        }
        //std::cout << testSamples.row(i) << std::endl;

        bool prediction = this->predict(vec);
        //exit(0);


        //bool prediction = results.at<float>(0) > 0.5 + FP_PENALTY;//results.at<float>(0)+FP_PENALTY;
        float pred_f = (results.at<float>(0) + 1.716) / (results.at<float>(0)+1.716);
        bool actual = testLabels.at<float>(i,0) > 0.5;
        fprintf(pFile2, "%d, %f, %f\n", actual, results.at<float>(0), results.at<float>(0));
        if (prediction == actual) {
          filtered_data.push_back(old_data[i]);
          filtered_labels.push_back(old_labels[i]);
          if (prediction) {
            tp++;
            pos_preds.push_back(pred_f);
          } else {
            tn++;
            neg_preds.push_back(pred_f);
          }
        } else {
          if (prediction) {
            //printf("dist of false positive is %f\n", new_errors[i]);
            fp++;
            neg_preds.push_back(pred_f);
          } else {
            //printf("dist of false negative is %f\n", new_errors[i]);
            fn++;
            pos_preds.push_back(pred_f);
          }
        } 
      }
      pFile = fopen ("for_hist_testing.csv","w");
      for (int i = 0; i < pos_preds.size(); i++) {
        fprintf(pFile, "%.2f, ", pos_preds[i]);
      }
      fprintf(pFile, "\n");
      for (int i = 0; i < neg_preds.size(); i++) {
        fprintf(pFile, "%.2f, ", neg_preds[i]);
      }
      fprintf(pFile, "\n");
      
      
      if (old_labels.size() == 0) {
        printf("no training examples\n");
      } else {
        printf("on test set we got, accuracy = %f%%, true positives: %f%%, false positives: %f%%, true negatives: %f%%, false negatives: %f%%\n",
                     100.0*(tp+tn)/testSamples.rows, 100.0*tp/testSamples.rows, 100.0*fp/testSamples.rows, 100.0*tn/testSamples.rows, 100.0*fn/testSamples.rows);
      }
      //float percent = correct*100.0/(correct+wrong);
      //if (percent < 98.0) {
      //  printf("retraining!\n");
      //  old_data = filtered_data;
      //  old_labels = filtered_labels;
      //  train(true);
      //}

    }
    mutex->unlock();
  }

  void MLAnn::save(std::string filename) {
    mutex->lock();
    printf("saving model\n");
    model->save(filename);
    MLAnnState state;
    for (int i = 0; i < old_data.size(); i++) {
      state.add_match_tile_pair_vector();
      MatchTilePairVector tile_pair_vector;
      tile_pair_vector.set_label(old_labels[i]);
      tile_pair_vector.set_dist(new_errors[i]);
      for (int j = 0; j < old_data[i].size(); j++) {
        tile_pair_vector.add_feature_vector(old_data[i][j]);
        //*(tile_pair_vector.mutable_feature_vector(j)) = old_data[i][j];
      }
      *(state.mutable_match_tile_pair_vector(i)) = tile_pair_vector;
    }
    std::fstream output(filename+".pbuf", std::ios::out | std::ios::trunc | std::ios::binary);
    state.SerializeToOstream(&output);
    output.close();
    mutex->unlock();
  }

  void MLAnn::load(std::string filename, bool data_only) {
    mutex->lock();
    printf("loading model\n");
    
    //model->load(filename);
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (fs.isOpened()) {
      fs.release();
      if (!data_only) {
        ann_model = cv::ml::StatModel::load<cv::ml::RTrees>(filename);
        //model->load(filename);// = cv::ml::RTrees::load(filename);
        model = ann_model;
        trained = true;
        training_active = false;
      }

      MLAnnState state;
      std::fstream input(filename+".pbuf", std::ios::in | std::ios::binary);
      state.ParseFromIstream(&input);
      for (int i = 0; i < state.match_tile_pair_vector_size(); i++) {
        std::vector<float> fvector;
        MatchTilePairVector mvector = state.match_tile_pair_vector(i);
        for (int j = 0; j < mvector.feature_vector_size(); j++) {
          fvector.push_back(mvector.feature_vector(j));
        }
        old_labels.push_back(mvector.label());
        old_data.push_back(fvector);
        new_errors.push_back(mvector.dist());
      }
      input.close();
      
    } else {
      printf("No model found so starting with empty model\n");
      trained = false;
    }
    mutex->unlock();
    //if (fs.isOpened()) {
    //    const cv::FileNode& fn = fs["model"];
    //    model->read(fn);
    //    fs.release();
    //} else {
    //  printf("issue with reading model\n");
    //  assert(false);
    //}
  }
// spicific to Random Forest
//TODO(wheatman) not thread safe
  MLRandomForest::MLRandomForest(int num_features, std::string saved_model) : MLBase(num_features, saved_model)  {
    model = cv::ml::RTrees::create();
    size_of_feature_vector = num_features;
    if (!saved_model.empty()) {
      training_active = false;
      this->load(saved_model);
    }
  }


  void MLRandomForest::train(bool reinforcement) {
    if (training_active) {
      trained = true;
      
      if (!reinforcement) {
        clear_saved_buffer();
      }  
      balance_and_flush_train_buffer();
      int training_examples = old_labels.size();
      cv::Mat labels = cv::Mat::zeros(training_examples, 1, CV_32F);
      cv::Mat data = cv::Mat::zeros(training_examples, size_of_feature_vector , CV_32F);
      for (int i = 0; i < training_examples; i++) {
          labels.at<float>(i) = old_labels[i];
          for (int j = 0; j < size_of_feature_vector; j++) {
              data.at<float>(i, j) = new_data[i][j];
          }
      }
      cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, labels);
      model->train(tdata);

    }
  }
  void MLRandomForest::save(std::string filename) {
    //cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    model->save(filename);
  }

  void MLRandomForest::load(std::string filename, bool data_only) {
    trained = true;
    training_active = true;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (fs.isOpened()) {
      fs.release();
      model = cv::ml::ANN_MLP::load(filename);
    } else {
      printf("No model found so starting with empty model\n");
      trained = false;
    }
  }

}

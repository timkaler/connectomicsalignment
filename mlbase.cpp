#include "mlbase.hpp"

namespace tfk {

// general MLBAse functions
  void MLBase::flush_train_buffer() {
    for (int i = 0; i < new_data.size(); i++) {
      old_data.push_back(new_data[i]);
      old_labels.push_back(old_labels[i]);
    }

    new_data.clear();
    new_labels.clear();
  }

  void MLBase::clear_train_buffer() {
    new_data.clear();
    new_labels.clear();
  }

  void MLBase::clear_saved_buffer() {
    old_data.clear();
    old_labels.clear();
  }

  void MLBase::add_training_example(std::vector<float> new_vector, float new_label) {
      ASSERT(new_vector.size() == size_of_feature_vector);
      new_data.push_back(new_vector);
      new_labels.push_back(new_label);
  }


  bool MLBase::predict(std::vector<float> vec) {
    if (!trained) {
      return 0;
    }
    assert(false);
      cv:: Mat mat_vec = cv::Mat::zeros(1, size_of_feature_vector, CV_32F);
      for (int i = 0; i < size_of_feature_vector; i++) {
          mat_vec.at<float>(i) = vec[i];
      }
      return model->predict(mat_vec);
  }


// spcific to ANN
  MLAnn::MLAnn(int num_features) : MLBase(num_features) {
    cv::Ptr<cv::ml::ANN_MLP> ann_model = cv::ml::ANN_MLP::create();
    cv::Mat_<int> layers(2,1);
    layers(0) = num_features;     // input
    //layers(1) = 3;     // hidden
    layers(1) = 2;      // positive negative and unknown
    ann_model->setLayerSizes(layers);
    ann_model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
    model = ann_model;
    size_of_feature_vector = num_features;
  }

  // these next two are not very general
  // probably should do something like move the data to old_data and be able to recalcuate from the start with all the data
  // this would be useul for the case where we can't do reinforcement learning
  void MLAnn::train(bool reinforcement) {
      int new_training_examples = new_data.size();
      cv::Mat labels = cv::Mat::zeros(new_training_examples, 2, CV_32F);
      cv::Mat data = cv::Mat::zeros(new_training_examples, size_of_feature_vector , CV_32F);
      for (int i = 0; i < new_training_examples; i++) {
          if (new_labels[i]) {
              labels.at<float>(i, 1) = 1;
          } else {
              labels.at<float>(i, 0) = 1;
          }
          for (int j = 0; j < size_of_feature_vector; j++) {
              data.at<float>(i, j) = new_data[i][j];
          }
      }
      cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(data, cv::ml::ROW_SAMPLE, labels);
      if (reinforcement && trained) {
          model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
      } else {
          model->train(tdata);
      }
      clear_train_buffer();
      trained = true;
  }


// spicific to Random Forest
  MLRandomForest::MLRandomForest(int num_features) : MLBase(num_features)  {
    model = cv::ml::RTrees::create();
    size_of_feature_vector = num_features;
  }


  void MLRandomForest::train(bool reinforcement) {
    trained = true;
    
    if (!reinforcement) {
      clear_saved_buffer();
    }  
    flush_train_buffer();
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

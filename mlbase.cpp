#include "mlbase.hpp"

namespace tfk {

  MLBase::MLBase(int num_features) {
    // whatever spicific thing needs to be done to set up the model
    cv::Mat_<int> layers(2,1);
    layers(0) = num_features;     // input
    //layers(1) = 3;     // hidden
    layers(1) = 2;      // positive negative and unknown
    model->setLayerSizes(layers);
    model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
    size_of_feature_vector = num_features;
  }


  void MLBase::add_training_example(std::vector<float> new_vector, float new_label) {
      ASSERT(new_vector.size() == size_of_feature_vector);
      new_data.push_back(new_vector);
      new_labels.push_back(new_label);
  }

  // these next two are not very general
  // probably should do something like move the data to old_data and be able to recalcuate from the start with all the data
  // this would be useul for the case where we can't do reinforcement learning
  void MLBase::train(bool reinforcement) {
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
      if (reinforcement) {
          model->train(tdata, cv::ml::ANN_MLP::UPDATE_WEIGHTS);
      } else {
          model->train(tdata);
      }
      clear_train_buffer();
  }

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

  bool MLBase::predict(std::vector<float> vec) {
      cv:: Mat mat_vec = cv::Mat::zeros(1, size_of_feature_vector, CV_32F);
      for (int i = 0; i < size_of_feature_vector; i++) {
          mat_vec.at<float>(i) = vec[i];
      }
      return model->predict(mat_vec);
  }



}

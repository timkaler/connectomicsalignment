#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <mutex>
#include <thread>
#include <future>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/types_c.h>
#include "cilk_tools/Graph.h"
//#include "cilk_tools/engine.h"
#include "./common.h"
#include "./fasttime.h"
#include "AlignData.pb.h"

#include "./match.h"
#include "./ransac.h"

#include "mrparams.hpp"
#include "paramdb.hpp"
#include "opencv2/ml.hpp"
#include "cilk_tools/engine.h"
//#include "./meshoptimize.h"
#ifndef MLBASE
#define MLBASE

namespace tfk {

class MLBase {
  public:
    //TODO(wheatman) either read write locking or threadsafe structures
    std::recursive_mutex* mutex;

    // not needed for reinforcement learning
    std::vector<std::vector<float>> old_data;
    std::vector<float> old_labels;
    cv::Ptr<cv::ml::RTrees> ann_model;
    std::vector<std::vector<float>> new_data;
    std::vector<float> new_labels;
    std::vector<float> new_errors;
    cv::Ptr<cv::ml::StatModel> model;
    int size_of_feature_vector;
    bool trained = false;
    int ml_correct_pos = 0;
    int ml_correct_neg = 0;
    int ml_fp = 0;
    int ml_fn = 0;
    bool training_active = true;
    int num_positive_examples = 0;
    int num_negative_examples = 0;

    MLBase(int num_features, std::string saved_model = "");
    virtual void train(bool reinforcement) = 0;
    virtual void save(std::string filename) = 0;
    virtual void load(std::string filename, bool data_only = false) = 0;


    void add_training_example(std::vector<float> new_vector, float new_label, float error);
    bool predict(std::vector<float> vec);
    void flush_train_buffer();
    void balance_and_flush_train_buffer();
    void clear_train_buffer();
    void clear_saved_buffer();
    void enable_training();
    void disable_training();
    float feature_dist(std::vector<float> a, std::vector<float> b);
    
};

class MLAnn  : public MLBase{
  public:
    MLAnn(int num_features, std::string saved_model = "");
    void train(bool reinforcement);
    void save(std::string filename);
    void load(std::string filename, bool data_only = false);
};

class MLRandomForest  : public MLBase{
  public:
    MLRandomForest(int num_features, std::string saved_model = "");
    void train(bool reinforcement);
    void save(std::string filename);
    void load(std::string filename, bool data_only = false);
};


} // end namespace tfk.
#endif // MLBASE

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

#include "cilk_tools/engine.h"
//#include "./meshoptimize.h"
#ifndef MLBASE
#define MLBASE

namespace tfk {

class MLBase {
  public:

    // not needed for reinforcement learning
    std::vector<std::vector<float>> old_data;
    std::vector<float> old_labels;

    std::vector<std::vector<float>> new_data;
    std::vector<float> new_labels;
    cv::Ptr<cv::ml::ANN_MLP> model = cv::ml::ANN_MLP::create();
    int size_of_feature_vector;

    MLBase(int num_features);
    void add_training_example(std::vector<float> new_vector, float new_label);
    void train(bool reinforcement);
    bool predict(std::vector<float> vec);
    void flush_train_buffer();
    void clear_train_buffer();
};

} // end namespace tfk.
#endif // MLBASE

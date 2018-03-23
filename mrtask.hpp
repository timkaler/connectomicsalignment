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

#include "cilk_tools/engine.h"
//#include "./meshoptimize.h"
#ifndef MRTASK
#define MRTASK


namespace tfk {

class MRTask {
  public:
    MRTask() {

    }

    //virtual std::vector<std::map<> get_compute_params();
    //virtual std::vector<int> get_error_check_params();

    //virtual void compute(std::vector<int> params);
    //virtual void error_check(std::vector<int> params);

    // compute --- returns false if it cannot satisfy the necessary correctness probability.
    //         --- returns true if it believes it can provide the desired correctness probability. 
    virtual void compute(float probability_correct, std::vector<int>& param_adjustments, std::vector<int>& param_train_deltas) = 0;

    // error_check --- returns true if 
    virtual bool error_check(float false_negative_rate) = 0;
};

} // end namespace tfk.
#endif // FPRTASK

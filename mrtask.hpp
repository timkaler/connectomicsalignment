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
#ifndef MRTASK
#define MRTASK


namespace tfk {

class MRTask {
  public:
    tfk::ParamDB* paramDB;
    tfk::MRParams* mr_params;
    MRTask() {

    }

    //virtual std::vector<std::map<> get_compute_params();
    //virtual std::vector<int> get_error_check_params();

    //virtual void compute(std::vector<int> params);
    //virtual void error_check(std::vector<int> params);

    // compute --- returns false if it cannot satisfy the necessary correctness probability.
    //         --- returns true if it believes it can provide the desired correctness probability. 
    virtual void compute_with_params(tfk::MRParams* mr_params) = 0;
    void compute(float probability_correct) {
        this->mr_params = paramDB->get_params_for_accuracy(probability_correct);
        compute_with_params(this->mr_params);
    }

    // error_check --- returns true if 
    virtual bool error_check(float false_negative_rate) = 0;

    virtual std::vector<tfk::MRParams> get_parameter_options() = 0;

    void setup_param_db(int trials) {
        std::vector<tfk::MRParams> param_options = get_parameter_options();
        for (auto &param : param_options) {
            std::clock_t start;
            double duration;
            start = std::clock();
            compute_with_params(&param);
            duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
            param.set_cost(duration);
            paramDB->import_params(&param);
            bool correct = error_check(0);
            if (correct) {
                paramDB->record_success(&param);
            } else {
                paramDB->record_failure(&param);
            }

            cilk_for (int i = 0; i < trials - 1; i++) {
                std::clock_t start;
                double duration;
                start = std::clock();
                compute_with_params(&param);
                duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
                bool correct = error_check(0);
                if (correct) {
                    paramDB->record_success(&param);
                } else {
                    paramDB->record_failure(&param);
                }
                param.set_cost((param.get_cost()*(param.success_count+param.failure_count-1)+duration)/ (param.success_count+param.failure_count));
            }

        }

    }
};

} // end namespace tfk.
#endif // FPRTASK

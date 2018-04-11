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
            double cost = 0;
            fasttime_t start = gettime();
            compute_with_params(&param);
            cost = tdiff(start, gettime());
            param.set_cost(cost);
            paramDB->import_params(&param);
            bool correct = error_check(0);
            if (correct) {
                paramDB->record_success(&param);
            } else {
                paramDB->record_failure(&param);
            }
            
            
            for (int i = 1; i < trials; i++) {
                //TODO replace with better clock from fasttime
                start = gettime();
                compute_with_params(&param);
                double duration = tdiff(start, gettime());
                bool correct2 = error_check(0);
                if (correct2) {
                    paramDB->record_success(&param);
                } else {
                    paramDB->record_failure(&param);
                }
                cost += duration;
                
            }
            param.set_cost(cost / trials);

        }

    }
    virtual void commit () = 0;
};

} // end namespace tfk.
#endif // FPRTASK

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
#include "mlbase.hpp"

#include "cilk_tools/engine.h"
//#include "./meshoptimize.h"
#ifndef MRTASK
#define MRTASK


namespace tfk {

class MRTask {
  public:
    tfk::ParamDB* paramDB;
    tfk::MRParams* mr_params;
    tfk::MLBase* model;

    // used to keep track of the different task types to have single dbs and single mls
    int task_type_id;

    MRTask() {

    }
    virtual ~MRTask() {}

    //~MRTask() {};

    //virtual std::vector<std::map<> get_compute_params();
    //virtual std::vector<int> get_error_check_params();

    //virtual void compute(std::vector<int> params);
    //virtual void error_check(std::vector<int> params);

    // compute --- returns false if it cannot satisfy the necessary correctness probability.
    //         --- returns true if it believes it can provide the desired correctness probability. 
    virtual void compute_with_params(tfk::MRParams* mr_params) = 0;
    virtual void compute(float probability_correct) {
        this->mr_params = paramDB->get_params_for_accuracy(probability_correct);
        compute_with_params(this->mr_params);
    }

    // error_check --- returns true if 
    virtual bool error_check(float false_negative_rate) = 0;

    virtual void get_parameter_options(std::vector<tfk::MRParams*>* vec) = 0;

    virtual bool compare_results_and_update_model(MRTask* known_good, float accuracy) = 0;

    void setup_param_db_init(std::vector<tfk::MRParams*> *param_options) {
        get_parameter_options(param_options);
        for (int j = 0; j < param_options->size(); j++) {
            tfk::MRParams *param = (*param_options)[j];
            paramDB->import_params(param);
        }
    }
    void setup_param_db(int trials, MRTask* known_good) {
        std::vector<tfk::MRParams*> *param_options = paramDB->get_all_params();
        for (int j = 0; j < param_options->size(); j++) {
            tfk::MRParams *param = (*param_options)[j];
            //printf("param option %d out of %zu\n", j, param_options->size());
            //int count = param->get_count();
            double cost = 0;
            for (int i = 0; i < trials; i++) {
                fasttime_t start = gettime();
                compute_with_params(param);
                error_check(-1);
                double duration = tdiff(start, gettime());
                bool correct = this->compare_results_and_update_model(known_good, 5);
                if (correct) {
                    paramDB->record_success(param);
                } else {
                    paramDB->record_failure(param);
                }
                cost += duration;

            }

            param->set_cost(cost / trials);

        }

    }
    virtual void commit () = 0;
};

} // end namespace tfk.
#endif // FPRTASK

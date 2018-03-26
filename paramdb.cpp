#include "paramdb.hpp"
#include "make_paramsdb_gen.cpp"

namespace tfk {

  ParamDB::ParamDB (MRParams* default_params, MRParams* min_params, MRParams* max_params) {
    this->default_params = default_params;
    this->min_params = min_params;
    this->max_params = max_params;

    possible_params.push_back(default_params);
    possible_params.push_back(min_params);
    possible_params.push_back(max_params);
  }

  MRParams* ParamDB::get_params_for_accuracy(float accuracy) {
    MRParams* ret = this->possible_params[0];
    // Find the lowest cost MRParams with greater than given accuracy.
    for (int i = 0; i < possible_params.size(); i++) {
      MRParams* params = possible_params[i];
      if (params->get_accuracy() >= accuracy) {
        if (params->get_cost() < ret->get_cost() || ret->get_accuracy() < accuracy) {
          ret = params;  
        }
      }
    }
    return ret;
  }

  void ParamDB::import_params(MRParams* params) {
    possible_params.push_back(params);
  }

  MRParams* ParamDB::get_max_params() {return max_params;}
  MRParams* ParamDB::get_min_params() {return min_params;}
  MRParams* ParamDB::get_default_params() {return default_params;}

  void ParamDB::init_ParamsDB() {
    /* thrown in main in run.cpp to test
            tfk::MRParams* mrp0 = new tfk::MRParams;
        mrp0->put_int_param("num_features", 1);
        mrp0->put_int_param("num_octaves", 5);
        mrp0->put_float_param("scale", 0.100000);
        mrp0->set_accuracy(0);
        mrp0->set_cost(5.468917);
        tfk::MRParams* mrp1 = new tfk::MRParams;
        mrp1->put_int_param("num_features", 1);
        mrp1->put_int_param("num_octaves", 5);
        mrp1->put_float_param("scale", 0.15000);
        mrp1->set_accuracy(1);
        mrp1->set_cost(99999);
        tfk::MRParams* mrp2 = new tfk::MRParams;
        mrp2->put_int_param("num_features", 1);
        mrp2->put_int_param("num_octaves", 5);
        mrp2->put_float_param("scale", 0.200000);
        mrp2->set_accuracy(0.5);
        mrp2->set_cost(100);
  tfk::ParamDB pdb = tfk::ParamDB(mrp2, mrp0, mrp1);
  pdb.init_ParamsDB();
  */
    param_db_import(this);
    for (float acc = .01; acc <=1; acc+=.01) {
      printf("acc = %f, cost =  %f\n",acc, this->get_params_for_accuracy(acc)->get_cost());
    }
  }



// end namespace tfk
}

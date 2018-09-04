/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

#include "align.h"
#include "fasttime.h"
#include "./cxxopts.hpp"

#include <string>
#include <vector>

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif


std::string TFK_TMP_DIR;

std::pair<cv::Point2f, cv::Point2f> process_bounding_box_string(std::string bounding_box_string) {

  std::stringstream ss(bounding_box_string);
  std::string token;
  char delim = ',';
  std::vector<float> items;
  while (std::getline(ss,token,delim)) {
    items.push_back(stof(token));
  }

  if (items.size() == 4) {
    //printf("Got a bounding box.\n");
    for (int i =0; i < items.size(); i++) {
      //printf("item %d %f\n", i, items[i]);
    }
  }
  return std::make_pair(cv::Point2f(items[0], items[1]), cv::Point2f(items[2], items[3]));
}

int main(int argc, char **argv) {

  cv::setNumThreads(0);

  align_data_t *p_align_data;
  p_align_data = (align_data_t *)malloc(sizeof(align_data_t));

  //init_align(p_align_data);

  // declare all the options here.
  int zstart, numslices, mode;
  std::string datafile_path, outputdir_path;

  bool use_params,usefsj,skipoctfast,skipoctslow;
  float fast_scale;

  // parse the arguments.
  try {
    const int NUM_REQUIRED = 4;
    const char* required[NUM_REQUIRED] =
        {"zstart",
         "numslices",
         "datafile",
         "outputdir"};

    std::string bounding_box_string = "";
    std::string tmpdir;
    cxxopts::Options options(argv[0], " - testing command line options.");
    options.positional_help("Positional Help Text");
    options.add_options()
      ("z,zstart", "Z Start Index", cxxopts::value<int>(zstart))
      ("n,numslices", "Number of frames to process after zstart", cxxopts::value<int>(numslices))
      ("d,datafile", "The protobuf containing the tilespec information.", cxxopts::value<std::string>(datafile_path))
      ("o,outputdir", "The directory used to output results.", cxxopts::value<std::string>(outputdir_path))
      ("m,mode", "what mode to run in 1 to run the alignment 2 to run the paramter optimization", cxxopts::value<int>(mode))
      ("useparams", "use params for matchtilepairtask and fsj", cxxopts::value<bool>(use_params))
      ("fastscale", "param for fast pass image scale", cxxopts::value<float>(fast_scale))
      ("skipoctfast", "skip first octave for fast", cxxopts::value<bool>(skipoctfast))
      ("skipoctslow", "skip first octave for slow", cxxopts::value<bool>(skipoctslow))
      ("usefsj", "enable fsj, o.w. fast pass always succeeds", cxxopts::value<bool>(usefsj))
      ("tmpdir", "enable fsj, o.w. fast pass always succeeds", cxxopts::value<std::string>(tmpdir))
      ("help", "Print help")
      ("b,bbox", "Bounding box", cxxopts::value<std::string>(bounding_box_string));

    options.parse(argc, argv);

    if (options.count("help")) {
      std::cout << options.help({"", "Group"}) << std::endl;
    }

    if (!options.count("mode")) {
      mode = 1;
    }


    bool missing_required = false;
    for (int i = 0; i < NUM_REQUIRED; i++) {
      if (!options.count(required[i])) {
        std::cout << "Missing required argument '" << required[i] << "'" << std::endl;
        missing_required = true;
      }
    }

    if (!options.count("useparams") ) {
      p_align_data->use_params = false;
    } else {
      p_align_data->use_params = true;
      printf("using params!\n");
      if (!options.count("fastscale") ||
          !options.count("tmpdir")) {
        printf("Missing required options that go withn params\n");
        exit(0);
      }
      p_align_data->scale_fast = fast_scale;
      p_align_data->skip_octave_fast = options.count("skipoctfast");
      p_align_data->skip_octave_slow = options.count("skipoctslow");
      p_align_data->use_fsj = options.count("usefsj");
      p_align_data->TMP_DIR = (char*) tmpdir.c_str();
      TFK_TMP_DIR = tmpdir;
    }


    if (missing_required) {
      std::cout << "Missing at least one required argument, stopping program" << std::endl;
      exit(0);
    }

    p_align_data->mode = mode; // probably unused.
    p_align_data->base_section = zstart;
    p_align_data->n_sections = numslices;
    p_align_data->input_filepath = (char*)datafile_path.c_str();
    p_align_data->work_dirpath = (char*)outputdir_path.c_str(); // use same path for both.
    p_align_data->output_dirpath = (char*)outputdir_path.c_str();
    //TFK_TMP_DIR = std::to_string(output_dir_path);
    p_align_data->do_subvolume = false;
    if (options.count("bbox")) {
      p_align_data->bounding_box = process_bounding_box_string(bounding_box_string);
    }

  } catch (const cxxopts::OptionException& e) {
    std::cout << "Error parsing options: " << e.what() << std::endl;
    exit(0);
  }

  // Execute the actual alignment code.
  #ifdef PROFILE
    ProfilerStart("profile.data");
  #endif
    if (mode == 1) {
      align_execute(p_align_data);
    } else if (mode == 2) {
      //param_optimize(p_align_data);
    } else if (mode == 3) {
      //test_learning(p_align_data);
    } else if (mode == 4) {
      //fill_match_tiles_task_pdb(p_align_data);
    } else if (mode == 5) {
      //testing_corralation_test(p_align_data);
    } else if (mode == 6) {
      train_fsj(p_align_data);
    } else {
      std::cout << "Invalid mode, stopping program" << std::endl;
    }
    
  #ifdef PROFILE
    ProfilerStop();
  #endif
  return 0;
}

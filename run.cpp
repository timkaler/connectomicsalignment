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

std::pair<cv::Point2f, cv::Point2f> process_bounding_box_string(std::string bounding_box_string) {

  std::stringstream ss(bounding_box_string);
  std::string token;
  char delim = ',';
  std::vector<float> items;
  while (std::getline(ss,token,delim)) {
    items.push_back(stof(token));
  }

  if (items.size() == 4) {
    printf("Got a bounding box.\n");
    for (int i =0; i < items.size(); i++) {
      printf("item %d %f\n", i, items[i]);
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

  // parse the arguments.
  try {
    const int NUM_REQUIRED = 4;
    const char* required[NUM_REQUIRED] =
        {"zstart",
         "numslices",
         "datafile",
         "outputdir"};

    std::string bounding_box_string = "";

    cxxopts::Options options(argv[0], " - testing command line options.");
    options.positional_help("Positional Help Text");
    options.add_options()
      ("z,zstart", "Z Start Index", cxxopts::value<int>(zstart))
      ("n,numslices", "Number of frames to process after zstart", cxxopts::value<int>(numslices))
      ("d,datafile", "The protobuf containing the tilespec information.", cxxopts::value<std::string>(datafile_path))
      ("o,outputdir", "The directory used to output results.", cxxopts::value<std::string>(outputdir_path))
      ("m,mode", "what mode to run in 1 to run the alignment 2 to run the paramter optimization", cxxopts::value<int>(mode))
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
    p_align_data->do_subvolume = false;
    p_align_data->bounding_box = process_bounding_box_string(bounding_box_string);

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
      param_optimize(p_align_data);
    } else if (mode == 3) {
      test_learning(p_align_data);
    } else {
      std::cout << "Invalid mode, stopping program" << std::endl;
    }
    
  #ifdef PROFILE
    ProfilerStop();
  #endif
  return 0;
}

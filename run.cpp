/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

#include "align.h"
#include "fasttime.h"
#include "./cxxopts.hpp"

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

int main(int argc, char **argv) {

  cv::setNumThreads(0);

  align_data_t *p_align_data;
  p_align_data = (align_data_t *)malloc(sizeof(align_data_t));

  //init_align(p_align_data);

  // declare all the options here.
  int zstart, numslices;
  std::string datafile_path, outputdir_path;

  // parse the arguments.
  try {
    const int NUM_REQUIRED = 4;
    const char* required[NUM_REQUIRED] =
        {"zstart",
         "numslices",
         "datafile",
         "outputdir"};

    cxxopts::Options options(argv[0], " - testing command line options.");
    options.positional_help("Positional Help Text");
    options.add_options()
      ("z,zstart", "Z Start Index", cxxopts::value<int>(zstart))
      ("n,numslices", "Number of frames to process after zstart", cxxopts::value<int>(numslices))
      ("d,datafile", "The protobuf containing the tilespec information.", cxxopts::value<std::string>(datafile_path))
      ("o,outputdir", "The directory used to output results.", cxxopts::value<std::string>(outputdir_path))
      ("help", "Print help");

    options.parse(argc, argv);
    if (options.count("help")) {
      std::cout << options.help({"", "Group"}) << std::endl;
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

    p_align_data->mode = 1; // probably unused.
    p_align_data->base_section = zstart;
    p_align_data->n_sections = numslices;
    p_align_data->input_filepath = (char*)datafile_path.c_str();
    p_align_data->work_dirpath = (char*)outputdir_path.c_str(); // use same path for both.
    p_align_data->output_dirpath = (char*)outputdir_path.c_str();
    p_align_data->do_subvolume = false;

  } catch (const cxxopts::OptionException& e) {
    std::cout << "Error parsing options: " << e.what() << std::endl;
    exit(0);
  }

  // Execute the actual alignment code.
  #ifdef PROFILE
    ProfilerStart("profile.data");
  #endif
    align_execute(p_align_data);
  #ifdef PROFILE
    ProfilerStop();
  #endif
  return 0;
}

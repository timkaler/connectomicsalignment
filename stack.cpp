#include "stack.hpp"
#include "stack_helpers.cpp"
#include "./othersift2.hpp"
#include "./stack_tile.cpp"
#include "./stack_section.cpp"



// Contains the initial code for mr stuff.
#include "stack_mr.cpp"

//#include "cilk_tools/engine.h"


// BEGIN init functions
tfk::Stack::Stack(int base_section, int n_sections,
    std::string input_filepath, std::string output_dirpath) {
  this->base_section = base_section;
  this->n_sections = n_sections;
  this->input_filepath = input_filepath;
  this->output_dirpath = output_dirpath;
}

void tfk::Stack::init() {
  printf("Initializing the stack.\n");
  AlignData align_data;
  // Read the existing address book.
  std::fstream input(this->input_filepath, std::ios::in | std::ios::binary);
  if (!align_data.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse protocal buffer." << std::endl;
    exit(1);
  }
  // first deal with AlignData level
  if (align_data.has_mode()) {
    this->mode = align_data.mode();
  }

  if (align_data.has_output_dirpath()) {
    this->output_dirpath = align_data.output_dirpath();
  }

  if (align_data.has_base_section()) {
    this->base_section = align_data.base_section();
  }

  if (align_data.has_n_sections()) {
    this->n_sections = align_data.n_sections();
  }

  if (align_data.has_do_subvolume()) {
    this->do_subvolume = align_data.do_subvolume();
    this->min_x = align_data.min_x();
    this->min_y = align_data.min_y();
    this->max_x = align_data.max_x();
    this->max_y = align_data.max_y();
  }

  // then do the section data
  for (int i = this->base_section; i < (this->base_section + this->n_sections); i++) {
    SectionData section_data = align_data.sec_data(i);
    Section* sec = new Section(section_data);
    sec->section_id = this->sections.size();
    this->sections.push_back(sec);
  }
}
// END init functions


// BEGIN Test functions
void tfk::Stack::test_io() {
  for (int i = 0; i < this->sections.size(); i++) {
    Section* section = this->sections[i];
    cilk_for (int j = 0; j < section->tiles.size(); j++) {
      Tile* tile = section->tiles[j];
      cv::Mat mat = tile->get_tile_data(Resolution::FILEIOTEST);
      mat.release();
      printf("tile %d of section %d\n", j, i);
    }
  }
}

void tfk::Stack::compute_on_tile_neighborhood(tfk::Section* section, tfk::Tile* tile) {
  int distance = 2;
  std::vector<Tile*> neighbors = section->get_all_close_tiles(tile);
  std::set<Tile*> active_set;
  active_set.insert(tile);
  for (int i = 0; i < neighbors.size(); i++) {
    active_set.insert(neighbors[i]);
  }
  for (int j = 0; j < 5000; j++) {
    tile->local2DAlignUpdateLimited(&active_set);
    for (int i = 0; i < neighbors.size(); i++) {
      neighbors[i]->local2DAlignUpdateLimited(&active_set);
    }
  }
}
// END Test functions

// Begin rendering functions.
void tfk::Stack::render_error(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix) {
  cilk_for (int i = 1; i < this->sections.size()-2; i++) {
    std::cout << "starting section "  << i << std::endl;
    Section* section = this->sections[i];
    std::pair<std::vector<std::pair<cv::Point2f, cv::Point2f>>, std::vector<std::pair<cv::Point2f, cv::Point2f> > > res = section->render_error(this->sections[i-1], this->sections[i+1], this->sections[i+2], bbox, filename_prefix+std::to_string(i)+".png");
  }
}

void tfk::Stack::render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix,
    Resolution res) {
  cilk_for (int i = 0; i < this->sections.size(); i++) {
    Section* section = this->sections[i];
    section->render(bbox, filename_prefix+std::to_string(section->real_section_id)+".tif", res);
  }
}


// BEGIN Alignment algorithms.
void tfk::Stack::align_3d() {
  for (int i = 0; i < this->sections.size(); i++) {
    cilk_spawn this->sections[i]->construct_triangles();
  }
  cilk_sync;

  std::vector<Section*> filtered_sections;
  for (int i = 0; i < this->sections.size(); i++) {
    if (this->sections[i]->real_section_id == 28 || 
        this->sections[i]->real_section_id == 29 || 
        this->sections[i]->real_section_id == 31) continue;
    filtered_sections.push_back(this->sections[i]);
  }
  this->sections = filtered_sections;

  cilk_spawn this->sections[0]->align_3d(this->sections[0]);

  for (int i = 1; i < this->sections.size(); i++) {
    cilk_spawn this->sections[i]->align_3d(this->sections[i-1]);
  }
  cilk_sync; 

  // section i is aligned to section i-1;
  for (int i = 1; i < this->sections.size(); i++) {
    Section* sec = this->sections[i];
    int j = i-1;
    for (int k = 0; k < sec->mesh->size(); k++) {
      (*sec->mesh)[k] = this->sections[j]->elastic_transform((*sec->mesh)[k]);
    }
  }
}

void tfk::Stack::align_2d() {
  int count = 0;
  int j = 0;
  int i = 0;
  while (j < this->sections.size()) {
    j += 4;
    if (j >= this->sections.size()) j = this->sections.size();

    for (; i < j; i++) {
       cilk_spawn this->sections[i]->align_2d();
    }
    cilk_sync;
  }
}

// END Alignment algorithms.

//}


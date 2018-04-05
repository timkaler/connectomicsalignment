#include "stack.hpp"
#include "stack_helpers.cpp"
#include "./othersift2.hpp"
#include "./stack_tile.cpp"
#include "./stack_section.cpp"


// Contains the initial code for mr stuff.
#include "stack_mr.cpp"

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
  printf("got this far\n");
  // then do the section data
  for (int i = this->base_section; i < (this->base_section + this->n_sections); i++) {
    printf("doing section %d\n", i);
    SectionData section_data = align_data.sec_data(i);
    printf("doing section %d\n", i);
    printf("bounding box is %f %f %f %f\n", _bounding_box.first.x, _bounding_box.first.y, _bounding_box.second.x, _bounding_box.second.y);
    Section* sec = new Section(section_data, _bounding_box);
    printf("doing section %d\n", i);
    sec->section_id = this->sections.size();
    printf("doing section %d\n", i);
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
  //int distance = 2;
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
  printf("Done with align 3d\n");
}

void tfk::Stack::align_2d() {
  //int count = 0;
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

#include "stack_learning.cpp"


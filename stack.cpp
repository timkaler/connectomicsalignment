// Copyright 2016 - Supertech Research Group

#include <utility>
#include <string>
#include <set>

#include "stack.hpp"
#include "./othersift2.hpp"
#include "./stack_tile.cpp"
#include "./stack_section.cpp"

#include "matchtilestask.hpp"


// Contains the initial code for mr stuff.

// BEGIN init functions
tfk::Stack::Stack(int base_section, int n_sections,
    std::string input_filepath, std::string output_dirpath) {
  this->base_section = base_section;
  this->n_sections = n_sections;
  this->input_filepath = input_filepath;
  this->output_dirpath = output_dirpath;
}


// gets bbox for entire stack.
std::pair<cv::Point2f, cv::Point2f> tfk::Stack::get_bbox() {
  float min_x = 0;
  float max_x = 0;
  float min_y = 0;
  float max_y = 0;

  for (int i = 0; i < this->sections.size(); i++) {
    Section* section = this->sections[i];
    std::pair<cv::Point2f, cv::Point2f> bbox = section->elastic_transform_bbox(section->get_bbox());
    if (i == 0) {
      min_x = bbox.first.x;
      min_y = bbox.first.y;
      max_x = bbox.second.x;
      max_y = bbox.second.y;
    } else {
      if (bbox.first.x < min_x) min_x = bbox.first.x;
      if (bbox.first.y < min_y) min_y = bbox.first.y;
      if (bbox.second.x > max_x) max_x = bbox.second.x;
      if (bbox.second.y > max_y) max_y = bbox.second.y;
    }
  }
  return std::make_pair(cv::Point2f(min_x, min_y), cv::Point2f(max_x, max_y));
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

  if (!this->align_data->use_params) {
    this->align_data->TMP_DIR = (char*)ALIGN_CACHE_FILE_DIRECTORY.c_str();
  }

  // first deal with AlignData level
  if (align_data.has_mode()) {
    this->mode = align_data.mode();
  }

  if (align_data.has_output_dirpath()) {
    this->output_dirpath = align_data.output_dirpath();
  }

  //if (align_data.has_base_section()) {
  //  this->base_section = align_data.base_section();
  //}

  //if (align_data.has_n_sections()) {
  //  this->n_sections = align_data.n_sections();
  //}

  if (align_data.has_do_subvolume()) {
    this->do_subvolume = align_data.do_subvolume();
    this->min_x = align_data.min_x();
    this->min_y = align_data.min_y();
    this->max_x = align_data.max_x();
    this->max_y = align_data.max_y();
  }
  printf("got this far\n");

  printf("setting up the ml models and the paramsdb\n");

  this->ml_models[MATCH_TILE_PAIR_TASK_ID] = new MLAnn(12-4 +6+4);
  this->ml_models[MATCH_TILES_TASK_ID] = new MLAnn(4);

  std::string ml_model_location = "tfk_test_model";
  printf("the ml model for task MATCH_TILE_PAIR is at %s\n", ml_model_location.c_str());
  try {
    this->ml_models[MATCH_TILE_PAIR_TASK_ID]->load(ml_model_location, true);
    this->ml_models[MATCH_TILE_PAIR_TASK_ID]->enable_training();
    this->ml_models[MATCH_TILE_PAIR_TASK_ID]->train(false);
    this->ml_models[MATCH_TILE_PAIR_TASK_ID]->disable_training();

    this->ml_models[MATCH_TILE_PAIR_TASK_ID]->save("test_tfk_model.ml");
  } catch(const std::exception& e) {
    printf("There was an exception!\n");
  }
  ParamsDatabase pdb;

  this->paramdbs[MATCH_TILE_PAIR_TASK_ID] = new tfk::ParamDB(pdb);
  ParamsDatabase pdb2;
  this->paramdbs[MATCH_TILES_TASK_ID] = new tfk::ParamDB(pdb2);

  // then do the section data
  for (int i = this->base_section; i < (this->base_section + this->n_sections); i++) {
    SectionData section_data = align_data.sec_data(i);
    Section* sec = new Section(section_data, _bounding_box, use_bbox_prefilter);
    sec->align_data = this->align_data;
    sec->section_id = this->sections.size();
    this->sections.push_back(sec);
    sec->ml_models = this->ml_models;
    sec->paramdbs = this->paramdbs;
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
// END Test functions

// BEGIN Alignment algorithms.
void tfk::Stack::align_3d() {
  cilk_for (int i = 0; i < this->sections.size(); i++) {
    this->sections[i]->construct_triangles();
  }

  std::set<int> sections_with_3d_keypoints;
  cilk_for (int i = 0; i < this->sections.size(); i++) {
    this->sections[i]->read_3d_keypoints("");
  }


  for (int i = 0; i < this->sections.size(); i++) {
    if (i == 0) {
      this->sections[0]->align_3d(this->sections[0]);
    } else {
      this->sections[i]->align_3d(this->sections[i-1]);
    }
  }

  // calculate bounding points for bounding box.
  //std::vector<cv::Point2f> edges; 
  //for (int i = this->sections.size()-1; i>=0; i--){
  // auto sec_bbox = this->sections[i]->triangle_mesh->index->bbox;
  // edges.push_back(sec_bbox.first);
  // edges.push_back(sec_bbox.second);
  // edges.push_back(cv::Point2f(sec_bbox.first.x, sec_bbox.second.y));
  // edges.push_back(cv::Point2f(sec_bbox.second.x, sec_bbox.first.y));
  // double min_x = sec_bbox.second.x;
  // double min_y = sec_bbox.second.y;
  // double max_x = sec_bbox.first.x;
  // double max_y = sec_bbox.first.y;
  // for (int j=0; j<edges.size(); j++){
  //   double x = edges[j].x;
  //   double y = edges[j].y;
  //   if (x < min_x) min_x = x;
  //   if (y < min_y) min_y = y;
  //   if (x > max_x) max_x = x;
  //   if (y > max_y) max_y = y;
  // }
  // this->sections[i]->estimate_bbox = std::make_pair(cv::Point2f(min_x,min_y), cv::Point2f(max_x, max_y));
  // cilk_for(int j=0; j<edges.size(); j++){
  //   edges[j] = this->sections[i]->affine_transform_point(edges[j]);
  // }
  //}
  //
  //// Expand and align the borders of the mesh to accomodate the borders of the section above
  //cilk_for(int i = 0; i<this->sections.size()-1;  i++){
  // this->sections[i]->expand_mesh();
  //}

  // section i is aligned to section i-1;
  for (int i = 1; i < this->sections.size(); i++) {
    Section* sec = this->sections[i];
    int j = i-1;
    for (int k = 0; k < sec->triangle_mesh->mesh->size(); k++) {
      (*sec->triangle_mesh->mesh)[k] =
          this->sections[j]->elastic_transform((*sec->triangle_mesh->mesh)[k]);
    }
  }
}

void tfk::Stack::align_2d() {
  int j = 0;
  int i = 0;
  while (j < this->sections.size()) {
    j += 1;
    if (j >= this->sections.size()) j = this->sections.size();
    if (j-i > 1) {
      for (; i < j; i++) {
         cilk_spawn this->sections[i]->align_2d();
      }
      cilk_sync;
    } else {
      for (; i < j; i++) {
         this->sections[i]->align_2d();
      }
    }
  }
}

#include "stack_learning.cpp"


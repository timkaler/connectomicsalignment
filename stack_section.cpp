// Copyright 2016 - Supertech Research Group

#include <mutex>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include <set>

#include "./fasttime.h"

#include "./render.hpp"
#include "./matchtilestask.hpp"
#include "./elastic_optimization.h"
#include <cilk/reducer_opadd.h>

extern fasttime_t global_start;
static int64_t ALL_2D_ERRORS_1 = 0;
static int64_t ALL_2D_ERRORS_2 = 0;
static int64_t ALL_2D_ERRORS_4 = 0;
static int64_t ALL_2D_ERRORS_8 = 0;
static int64_t ALL_2D_ERRORS_16 = 0;

#define CORR_THRESH 0.8


void tfk::Section::print_2d_error_info(
    Tile* t, Tile* n, float val) {
  printf("===BEGIN===");
  printf("error info %d %d %f\n", t->tile_id, n->tile_id, val);
  for (int i = 0; i < t->edges.size(); i++) {
    Tile* neighbor = this->tiles[t->edges[i].neighbor_id];
    if (neighbor->bad_2d_alignment) {
      printf("Skip neighbor %d (bad alignment)\n", neighbor->tile_id);
      continue;
    }

    if (t->ideal_offsets.find(neighbor->tile_id) == t->ideal_offsets.end() &&
        n->ideal_offsets.find(t->tile_id) == neighbor->ideal_offsets.end()) {
      printf("Skip neighbor %d (no ideal offset)\n", neighbor->tile_id);
      continue;
    }

      cv::Point2f a_point = cv::Point2f(t->x_start+t->offset_x,
                                        t->y_start+t->offset_y);
      cv::Point2f b_point = cv::Point2f(neighbor->x_start+neighbor->offset_x,
                                        neighbor->y_start+neighbor->offset_y);
    cv::Point2f delta = a_point-b_point;
      cv::Point2f deviation;
      if (t->ideal_offsets.find(neighbor->tile_id) != t->ideal_offsets.end()) {
        deviation = t->ideal_offsets[neighbor->tile_id] - delta;
      } else {
        deviation = -1*neighbor->ideal_offsets[t->tile_id] - delta;
      }

    printf("neigh %d a_pt %f,%f; b_pt %f,%f; delta %f,%f; deviation %f,%f\n",
           neighbor->tile_id, a_point.x, a_point.y, b_point.x, b_point.y, delta.x,
           delta.y, deviation.x, deviation.y);
  }
  printf("===END===");
}

void tfk::Section::erase_3d_keypoints() {
  for (int i = 0; i < this->tiles.size(); i++) {
    this->tiles[i]->release_3d_keypoints();
  }
}


std::vector<tfk::Tile*> tfk::Section::get_all_neighbor_tiles(tfk::Tile* tile) {
  std::vector<Tile*> neighbors;
  for (int i = 0; i < tile->edges.size(); i++) {
    int id = tile->edges[i].neighbor_id;
    neighbors.push_back(this->tiles[id]);
  }
  return neighbors;
}

// Init functions
tfk::Section::Section(int section_id) {
  this->section_id = section_id;
  this->num_tiles_replaced = 0;
}

// BEGIN transformation functions
cv::Point2f tfk::Section::affine_transform(cv::Point2f pt) {
  float new_x = pt.x*this->a00 + pt.y * this->a01 + this->offset_x;
  float new_y = pt.x*this->a10 + pt.y * this->a11 + this->offset_y;
  return cv::Point2f(new_x, new_y);
}

cv::Point2f tfk::Section::affine_transform_plusA(cv::Point2f pt, cv::Mat A) {
  float pre_new_x = pt.x*this->a00 + pt.y * this->a01 + this->offset_x;
  float pre_new_y = pt.x*this->a10 + pt.y * this->a11 + this->offset_y;

  double ta00 = A.at<double>(0, 0);
  double ta01 = A.at<double>(0, 1);
  double toffset_x = A.at<double>(0, 2);
  double ta10 = A.at<double>(1, 0);
  double ta11 = A.at<double>(1, 1);
  double toffset_y = A.at<double>(1, 2);

  float new_x = pre_new_x*ta00 + pre_new_y * ta01 + toffset_x;
  float new_y = pre_new_x*ta10 + pre_new_y * ta11 + toffset_y;

  return cv::Point2f(new_x, new_y);
}

// assumes point p is post section-global affine.
cv::Point2f tfk::Section::elastic_transform(cv::Point2f p) {
  std::tuple<bool, float, float, float, int> info = this->get_triangle_for_point(p);
  if (!std::get<0>(info)) {
    //off_grid->push_back(p);
    return p;
  }

  int triangle_index = std::get<4>(info);
  if (triangle_index == -1) {
    printf("triangle not found for some reason! %f %f\n", p.x, p.y);
    return p;
  }
  renderTriangle tri = getRenderTriangle((*this->triangle_mesh->triangles)[triangle_index]);

  float u = std::get<1>(info);
  float v = std::get<2>(info);
  float w = std::get<3>(info);
  float new_x = u*tri.q[0].x + v*tri.q[1].x + w*tri.q[2].x;
  float new_y = u*tri.q[0].y + v*tri.q[1].y + w*tri.q[2].y;
  return cv::Point2f(new_x, new_y);
}

// assumes point p is post section-global affine.
cv::Point2f tfk::Section::elastic_transform(cv::Point2f p, Triangle _tri) {
  std::tuple<bool, float, float, float, int> info = this->get_triangle_for_point(p, _tri);
  if (!std::get<0>(info)) return p;


  int triangle_index = std::get<4>(info);
  renderTriangle tri = getRenderTriangle((*this->triangle_mesh->triangles)[triangle_index]);

  float u = std::get<1>(info);
  float v = std::get<2>(info);
  float w = std::get<3>(info);
  float new_x = u*tri.q[0].x + v*tri.q[1].x + w*tri.q[2].x;
  float new_y = u*tri.q[0].y + v*tri.q[1].y + w*tri.q[2].y;
  return cv::Point2f(new_x, new_y);
}
// END transformation functions


// BEGIN alignment functions
void tfk::Section::optimize_tile_grid() {
  double lr = 0.1;
  cilk_for (int _i = 0; _i < this->tiles.size(); _i++) {
    int i = _i;
    this->tiles[i]->grad_error_x = 0.0;
    this->tiles[i]->grad_error_y = 0.0;
    //this->tiles[i]->angle = 0.0;
    //this->tiles[i]->grad_error_angle = 0.0;
  }

  printf("starting run\n");
  double* tile_momentum_x = (double*) malloc(this->tiles.size()*sizeof(double));
  double* tile_momentum_y = (double*) malloc(this->tiles.size()*sizeof(double));
  cilk_for (int i = 0; i < this->tiles.size(); i++) {
    tile_momentum_x[i] = 0.0;
    tile_momentum_y[i] = 0.0;
  }

  bool keep_going = false;
  double last_energy = 0.0;
  //int num_angles = 1;

      TFK_TIMER_VAR(tfktimer); 
      TFK_START_TIMER(&tfktimer); 
      TFK_STOP_TIMER(&tfktimer, "begin"); 

  cv::Point2f* previous_offsets = (cv::Point2f*) calloc(this->tiles.size(), sizeof(cv::Point2f));


  cilk_for (int x = 0; x < this->tiles.size(); x++) {
    this->tiles[x]->local2DAlignUpdate_filter_edges();
  }

  for (int i = 0; i < 200000; i++) {
    double MOMENTUM = 0.1;
    // first iteration compute last energy directly.
    if (i == 0) {
      cilk::reducer_opadd<double> last_energy_reducer(0.0);
      cilk_for (int x = 0; x < this->tiles.size(); x++) {
        *last_energy_reducer += this->tiles[x]->local2DAlignUpdateEnergy();
      }
      last_energy = last_energy_reducer.get_value();
    }

    cilk_for (int x = 0; x < this->tiles.size(); x++) {
      this->tiles[x]->local2DAlignUpdateFaster(lr);
    }


    //double* previous_angles = (double*) calloc(this->tiles.size(), sizeof(double));
    cilk_for (int x = 0; x < this->tiles.size(); x++) {
      this->tiles[x]->grad_error_x += MOMENTUM*tile_momentum_x[x];
      this->tiles[x]->grad_error_y += MOMENTUM*tile_momentum_y[x];

      tile_momentum_x[x] = this->tiles[x]->grad_error_x;
      tile_momentum_y[x] = this->tiles[x]->grad_error_y;
      previous_offsets[x] = cv::Point2f(this->tiles[x]->offset_x, this->tiles[x]->offset_y);
      //previous_angles[x] = this->tiles[x]->angle;
      this->tiles[x]->offset_x += (this->tiles[x]->grad_error_x)*lr;
      this->tiles[x]->offset_y += (this->tiles[x]->grad_error_y)*lr;
      //if (num_angles > 1) {
      //  this->tiles[x]->angle += this->tiles[x]->grad_error_angle*lr;
      //}
    }

    double energy = 0.0;
    cilk::reducer_opadd<double> energy_reducer(0.0);
    cilk_for (int x = 0; x < this->tiles.size(); x++) {
      //int start = _x;
      //int end = _x+100;
      //if (end > this->tiles.size()) end = this->tiles.size();
      //for (int x = 0; x < this->tiles.size(); x++) {
      //double energy_sum = 0.0;
      *energy_reducer += this->tiles[x]->local2DAlignUpdateEnergyFaster();
      //for (int x = start; x < end; x++) {
      //}
      //*energy_reducer += energy_sum;
    }
    energy = energy_reducer.get_value();

    if (energy < last_energy) {
      // Decreased the energy. Increase the learning rate a little bit.
      lr = lr + 0.001;
    } else {
      // Increased the energy. Reset the step and decrease the learning rate.
      cilk_for (int j = 0; j < this->tiles.size(); j++) {
        // reset the momentum.
        tile_momentum_x[j] = 0.0;
        tile_momentum_y[j] = 0.0;
        // reset the previous offsets.
        this->tiles[j]->offset_x = previous_offsets[j].x;
        this->tiles[j]->offset_y = previous_offsets[j].y;
        //this->tiles[j]->angle = previous_angles[j];
      }
      lr = lr * 0.99;
      energy = last_energy;
    }
    if (energy + 5 < last_energy) keep_going = true;

    //free(previous_offsets);
    //free(previous_angles);
    if (i%1000 == 0) { 
      printf("intermediate energy for section %d with lr %f is %.10f\n", this->real_section_id, lr, last_energy);
    }
    // after 1000 iterations stop early if no past
    //   iteration has improved the energy.
    if (i >= 1000 && i%1000 == 0) {
      if (!keep_going) break; 
      keep_going = false;
    }

    last_energy = energy; // store last energy for next iter.
    //printf("intermediate energy for section %d with lr %f is %.10f\n", this->real_section_id, lr, last_energy);
  }

  free(previous_offsets);
  cilk_for (int x = 0; x < this->tiles.size(); x++) {
    this->tiles[x]->local2DAlignUpdate_filter_edges_cleanup();
  }

  TFK_STOP_TIMER(&tfktimer, "done with gradient descent\n"); 
  double energy_sum = 0.0;
  cilk_for (int i = 0; i < this->tiles.size(); i++) {
    double energy = this->tiles[i]->local2DAlignUpdateEnergy();
    energy_sum += energy;
    // Following two lines are used to generate grid-energy heatmaps.
    this->tiles[i]->energy = energy;
    this->tiles[i]->highlight = true;
  }
  printf("total energy for section %d is %f\n", this->real_section_id, energy_sum);
}

void tfk::Section::mark_bad_2d_alignment() {
    //MLBase *match_tile_task_model = this->ml_models[MATCH_TILE_PAIR_TASK_ID];

      MLBase *match_tile_pair_task_model = this->ml_models[MATCH_TILE_PAIR_TASK_ID];
      int bas_correct_pos = 0;
      int bas_correct_neg = 0;
      int bas_fp = 0;
      int bas_fn = 0;

      // init tmp bad 2d alignment.
      for (int i = 0; i < this->tiles.size(); i++) {
        this->tiles[i]->tmp_bad_2d_alignment = this->tiles[i]->bad_2d_alignment;
        this->tiles[i]->incident_bad_edges = 0;
      }

      for (int i = 0; i < this->tiles.size(); i++) {
        Tile* t = this->tiles[i];

        if (t->bad_2d_alignment) {
          continue;
        }


        for (int k = 0; k < t->edges.size(); k++) {
          Tile* neighbor = this->tiles[t->edges[k].neighbor_id];
          bool guess_ml = t->ml_preds[neighbor];
          MatchTilesTask *task = t->match_tiles_task;
          bool guess_basic = task->neighbor_to_success[neighbor];
          if (neighbor->bad_2d_alignment) continue;
          if (t->ideal_offsets.find(neighbor->tile_id) == t->ideal_offsets.end() ||
              neighbor->ideal_offsets.find(t->tile_id) == neighbor->ideal_offsets.end()) continue;

          float val = t->compute_deviation(neighbor);

          if (val > 10.0) {
              t->incident_bad_edges += 1;
              neighbor->incident_bad_edges += 1;
            if (guess_ml) {
              match_tile_pair_task_model->ml_fp++;
            } else {
              match_tile_pair_task_model->ml_correct_neg++;
            }
            if (guess_basic) {
              bas_fp++;
            } else {
              bas_correct_neg++;
            }
          } else {
            if (guess_ml) {
              match_tile_pair_task_model->ml_correct_pos++;
            } else {
              match_tile_pair_task_model->ml_fn++;
            }
            if (guess_basic) {
              bas_correct_pos++;
            } else {
              bas_fn++;
            }
          }
        }
      }

      for (int i = 0; i < this->tiles.size(); i++) {
        Tile* t = this->tiles[i];

        if (t->bad_2d_alignment) {
          continue;
        }


        for (int k = 0; k < t->edges.size(); k++) {
          Tile* neighbor = this->tiles[t->edges[k].neighbor_id];
          bool guess_ml = t->ml_preds[neighbor];
          MatchTilesTask *task = t->match_tiles_task;
          bool guess_basic = task->neighbor_to_success[neighbor];
          if (neighbor->bad_2d_alignment) continue;
          if (t->ideal_offsets.find(neighbor->tile_id) == t->ideal_offsets.end()) continue;

          float val = t->compute_deviation(neighbor);

          if (val > 10.0) {
                t->tmp_bad_2d_alignment = true;
                neighbor->tmp_bad_2d_alignment = true;
            if (guess_ml) {
              match_tile_pair_task_model->ml_fp++;
            } else {
              match_tile_pair_task_model->ml_correct_neg++;
            }
            if (guess_basic) {
              bas_fp++;
            } else {
              bas_correct_neg++;
            }
          } else {
            if (guess_ml) {
              match_tile_pair_task_model->ml_correct_pos++;
            } else {
              match_tile_pair_task_model->ml_fn++;
            }
            if (guess_basic) {
              bas_correct_pos++;
            } else {
              bas_fn++;
            }
          }
        }
      }

      // set the bad alignment based on results.
      for (int i = 0; i < this->tiles.size(); i++) {
        Tile* t = this->tiles[i];
        this->tiles[i]->bad_2d_alignment = this->tiles[i]->tmp_bad_2d_alignment;
        if (t->bad_2d_alignment) continue;
        bool broke = true;
        for (int k = 0; k < t->edges.size(); k++) {
          Tile* neighbor = this->tiles[t->edges[k].neighbor_id];
          if (!neighbor->bad_2d_alignment) {
            broke = false;
          }
        }
        t->bad_2d_alignment = broke;
      }

      printf("Basic Correct positive = %d, correct negatives = %d, false positives = %d, false negative = %d\n",
             bas_correct_pos, bas_correct_neg, bas_fp, bas_fn);
}

void tfk::Section::align_2d() {
    if (this->alignment2d_exists()) {
      std::string filename =
      std::string(std::string(TFK_TMP_DIR) + "/prefix_"+std::to_string(this->real_section_id));

      this->load_2d_alignment();
      //compare_2d_alignment();
      //this->load_2d_alignment();
      this->read_3d_keypoints(filename);
      return;
    }

    // Begin computing relative tile alignments.
    for (int i = 0; i < tiles.size(); i++) {
      tiles[i]->ml_models = this->ml_models;
      tiles[i]->paramdbs = this->paramdbs;
      std::vector<Tile*> close_tiles = get_all_close_tiles_with_min_overlap(tiles[i], 50);
      tiles[i]->match_tiles_task = new MatchTilesTask(tiles[i], close_tiles);
    }
    this->compute_keypoints_and_matches();

    for (int i = 0; i < this->tiles.size(); i++) {
      if (!tiles[i]->match_tiles_task->error_check(3.0)) {
        tiles[i]->bad_2d_alignment = true;
      }
    }
    // End computing relative tile alignments.

    // Optimize grid and then mark high-energy tiles as bad.
    this->optimize_tile_grid();
    this->mark_bad_2d_alignment();

    // save 2d alignment
    this->save_2d_alignment();
}



// END alignment functions


// BEGIN utility functions

bool tfk::Section::section_data_exists() {
  std::string filename =
      std::string(std::string(TFK_TMP_DIR)+"/prefix_"+std::to_string(this->real_section_id));

  printf("the filename is %s\n", filename.c_str());

  cv::FileStorage fs(filename+std::string("_3d_keypoints.yml.gz"), cv::FileStorage::READ);
  cv::FileStorage fs2(filename+std::string("_2d_matches.yml.gz"), cv::FileStorage::READ);
  if (!fs.isOpened() || !fs2.isOpened()) {
    return false;
  } else {
    return true;
  }
}

bool tfk::Section::transformed_tile_overlaps_with(Tile* tile,
    std::pair<cv::Point2f, cv::Point2f> bbox, bool use_elastic = true) {
  auto tile_bbox = tile->get_bbox();
  tile_bbox = this->affine_transform_bbox(tile_bbox);

  if (use_elastic) {
    tile_bbox = this->elastic_transform_bbox(tile_bbox);
  }

  int x1_start = tile_bbox.first.x;
  int x1_finish = tile_bbox.second.x;
  int y1_start = tile_bbox.first.y;
  int y1_finish = tile_bbox.second.y;

  int x2_start = bbox.first.x;
  int x2_finish = bbox.second.x;
  int y2_start = bbox.first.y;
  int y2_finish = bbox.second.y;

  bool res = false;
  if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
      (y1_start < y2_finish) && (y1_finish > y2_start)) {
      res = true;
  }
  return res;
}

void tfk::Section::save_elastic_mesh(Section* neighbor) {
  TriangleMeshProto triangleMesh;

  for (int i = 0; i < triangle_mesh->mesh->size(); i++) {
    cv::Point2f pt = (*(triangle_mesh->mesh))[i];
    PairDouble pair;
    pair.set_x(pt.x);
    pair.set_y(pt.y);
    triangleMesh.add_mesh();
    *(triangleMesh.mutable_mesh(i)) = pair;
  }

  for (int i = 0; i < triangle_mesh->mesh_orig->size(); i++) {
    cv::Point2f pt = (*(triangle_mesh->mesh_orig))[i];
    PairDouble pair;
    pair.set_x(pt.x);
    pair.set_y(pt.y);
    triangleMesh.add_mesh_orig();
    *(triangleMesh.mutable_mesh_orig(i)) = pair;
  }

  for (int i = 0; i < triangle_mesh->triangle_edges->size(); i++) {
    std::pair<int, int> edge = (*(triangle_mesh->triangle_edges))[i];
    PairInt pair;
    pair.set_x(edge.first);
    pair.set_y(edge.second);
    triangleMesh.add_triangle_edges();
    *(triangleMesh.mutable_triangle_edges(i)) = pair;
  }

  for (int i = 0; i < triangle_mesh->triangles->size(); i++) {
    tfkTriangle tri = (*(triangle_mesh->triangles))[i];
    TriangleReference ref;
    ref.set_index1(tri.index1);
    ref.set_index2(tri.index2);
    ref.set_index3(tri.index3);
    triangleMesh.add_triangles();
    *(triangleMesh.mutable_triangles(i)) = ref;
  }

  std::fstream output(std::string(TFK_TMP_DIR)+"/emesh_"+std::to_string(this->real_section_id)+"_"+
                      std::to_string(neighbor->real_section_id)+".pbuf",
                      std::ios::out | std::ios::trunc | std::ios::binary);
  triangleMesh.SerializeToOstream(&output);
  output.close();
}

// END utility functions




// BEGIN rendering functions
/*
cv::Point2f tfk::Section::get_render_scale(Resolution resolution) {
  if (resolution == THUMBNAIL || resolution == THUMBNAIL2) {
    Tile* first_tile = this->tiles[0];

    cv::Mat thumbnail_img = first_tile->get_tile_data(THUMBNAIL);
    cv::Mat img = first_tile->get_tile_data(FULL);

    float scale_x = (float)(img.size().width)/thumbnail_img.size().width;
    float scale_y = (float)(img.size().height)/thumbnail_img.size().height;
    thumbnail_img.release();
    img.release();
    return cv::Point2f(scale_x, scale_y);
  }

  if (resolution == FULL) {
    return cv::Point2f(1.0, 1.0);
  }

  if (resolution == PERCENT30) {
    return cv::Point2f(10.0/3, 10.0/3);
  }

  return cv::Point2f(1.0, 1.0);
}
*/

std::pair<cv::Point2f, cv::Point2f> tfk::Section::elastic_transform_bbox(
    std::pair<cv::Point2f, cv::Point2f> bbox) {
  cv::Point2f corners[4];
  corners[0] = cv::Point2f(bbox.first.x, bbox.first.y);
  corners[1] = cv::Point2f(bbox.second.x, bbox.first.y);
  corners[2] = cv::Point2f(bbox.first.x, bbox.second.y);
  corners[3] = cv::Point2f(bbox.second.x, bbox.second.y);

  for (int i = 0; i < 4; i++) {
    corners[i] = this->elastic_transform(corners[i]);
  }

  float min_x = corners[0].x;
  float max_x = corners[0].x;
  float min_y = corners[0].y;
  float max_y = corners[0].y;
  for (int i = 1; i < 4; i++) {
    if (corners[i].x < min_x) min_x = corners[i].x;
    if (corners[i].x > max_x) max_x = corners[i].x;

    if (corners[i].y < min_y) min_y = corners[i].y;
    if (corners[i].y > max_y) max_y = corners[i].y;
  }
  return std::make_pair(cv::Point2f(min_x, min_y), cv::Point2f(max_x, max_y));
}



// END rendering functions





std::pair<cv::Point2f, cv::Point2f> tfk::Section::scale_bbox(
    std::pair<cv::Point2f, cv::Point2f> bbox, cv::Point2f scale) {
  int lower_x = (int) (bbox.first.x/scale.x + 0.5);
  int lower_y = (int) (bbox.first.y/scale.y + 0.5);
  int upper_x = (int) (bbox.second.x/scale.x + 0.5);
  int upper_y = (int) (bbox.second.y/scale.y + 0.5);
  return std::make_pair(cv::Point2f(1.0*lower_x, 1.0*lower_y),
                        cv::Point2f(1.0*upper_x, 1.0*upper_y));
}

bool tfk::Section::tile_in_render_box_affine(cv::Mat A, Tile* tile,
                                             std::pair<cv::Point2f, cv::Point2f> bbox) {
  std::pair<cv::Point2f, cv::Point2f> tile_bbox = tile->get_bbox();

  cv::Point2f corners[4];
  corners[0] = cv::Point2f(tile_bbox.first.x, tile_bbox.first.y);
  corners[1] = cv::Point2f(tile_bbox.second.x, tile_bbox.first.y);
  corners[2] = cv::Point2f(tile_bbox.first.x, tile_bbox.second.y);
  corners[3] = cv::Point2f(tile_bbox.second.x, tile_bbox.second.y);

  for (int i = 0; i < 4; i++) {
    corners[i] = this->affine_transform_plusA(corners[i], A);
  }

  float tile_min_x = corners[0].x;
  float tile_max_x = corners[0].x;
  float tile_min_y = corners[0].y;
  float tile_max_y = corners[0].y;
  for (int i = 1; i < 4; i++) {
    if (corners[i].x < tile_min_x) tile_min_x = corners[i].x;
    if (corners[i].y < tile_min_y) tile_min_y = corners[i].y;

    if (corners[i].x > tile_max_x) tile_max_x = corners[i].x;
    if (corners[i].y > tile_max_y) tile_max_y = corners[i].y;
  }

  if (tile_max_x < bbox.first.x) return false;
  if (tile_max_y < bbox.first.y) return false;
  if (tile_min_x > bbox.second.x) return false;
  if (tile_min_y > bbox.second.y) return false;

  return true;
}



bool tfk::Section::tile_in_render_box(Tile* tile, std::pair<cv::Point2f, cv::Point2f> bbox) {
  std::pair<cv::Point2f, cv::Point2f> tile_bbox = tile->get_bbox();

  cv::Point2f corners[4];
  corners[0] = cv::Point2f(tile_bbox.first.x, tile_bbox.first.y);
  corners[1] = cv::Point2f(tile_bbox.second.x, tile_bbox.first.y);
  corners[2] = cv::Point2f(tile_bbox.first.x, tile_bbox.second.y);
  corners[3] = cv::Point2f(tile_bbox.second.x, tile_bbox.second.y);

  for (int i = 0; i < 4; i++) {
    corners[i] = this->affine_transform(corners[i]);
    corners[i] = this->elastic_transform(corners[i]);
  }

  float tile_min_x = corners[0].x;
  float tile_max_x = corners[0].x;
  float tile_min_y = corners[0].y;
  float tile_max_y = corners[0].y;
  for (int i = 1; i < 4; i++) {
    if (corners[i].x < tile_min_x) tile_min_x = corners[i].x;
    if (corners[i].y < tile_min_y) tile_min_y = corners[i].y;

    if (corners[i].x > tile_max_x) tile_max_x = corners[i].x;
    if (corners[i].y > tile_max_y) tile_max_y = corners[i].y;
  }

  if (tile_max_x < bbox.first.x) return false;
  if (tile_max_y < bbox.first.y) return false;
  if (tile_min_x > bbox.second.x) return false;
  if (tile_min_y > bbox.second.y) return false;

  return true;
}

renderTriangle tfk::Section::getRenderTriangle(tfkTriangle tri) {
  renderTriangle rTri;
  rTri.p[0] = (*(this->triangle_mesh->mesh_orig))[tri.index1];
  rTri.p[1] = (*(this->triangle_mesh->mesh_orig))[tri.index2];
  rTri.p[2] = (*(this->triangle_mesh->mesh_orig))[tri.index3];

  rTri.q[0] = (*(this->triangle_mesh->mesh))[tri.index1];
  rTri.q[1] = (*(this->triangle_mesh->mesh))[tri.index2];
  rTri.q[2] = (*(this->triangle_mesh->mesh))[tri.index3];
  return rTri;
}

std::tuple<bool, float, float, float, int> tfk::Section::get_triangle_for_point(cv::Point2f pt) {
  Triangle tri = this->triangle_mesh->find_triangle(pt);

  if (tri.index == -1) {
    return std::make_tuple(false, -1, -1, -1, -1);
  }

  renderTriangle rTri = this->getRenderTriangle((*this->triangle_mesh->triangles)[tri.index]);
  float u, v, w;
  cv::Point2f a, b, c;
  a = rTri.p[0];
  b = rTri.p[1];
  c = rTri.p[2];

  Barycentric(pt, a, b, c, u, v, w);
  if (u >= 0 && v >= 0 && w >= 0) {
    return std::make_tuple(true, u, v, w, tri.index);
  }
  return std::make_tuple(false, -1, -1, -1, -1);
}


std::tuple<bool, float, float, float, int> tfk::Section::get_triangle_for_point(cv::Point2f pt,
    Triangle tri) {
  renderTriangle rTri = this->getRenderTriangle((*this->triangle_mesh->triangles)[tri.index]);
  float u, v, w;
  cv::Point2f a, b, c;
  a = rTri.p[0];
  b = rTri.p[1];
  c = rTri.p[2];

  Barycentric(pt, a, b, c, u, v, w);
  if (u >= 0 && v >= 0 && w >= 0) {
    return std::make_tuple(true, u, v, w, tri.index);
  }
  return std::make_tuple(false, -1, -1, -1, -1);
}



void tfk::Section::get_3d_keypoints_for_box(std::pair<cv::Point2f, cv::Point2f> bbox,
    std::vector<cv::KeyPoint>& kps_in_box, cv::Mat& kps_desc_in_box,
    bool use_cached, tfk::params sift_parameters, std::vector<Tile*>& tiles_loaded,
    std::mutex& tiles_loaded_mutex,
    bool apply_transform) {
  std::vector <cv::KeyPoint > atile_all_kps;
  std::vector <cv::Mat > atile_all_kps_desc;
  if (use_cached) {
    for (int i = 0; i < this->tiles.size(); i++) {
      if (this->tiles[i]->bad_2d_alignment) continue;
      if (apply_transform) {
        if (this->transformed_tile_overlaps_with(this->tiles[i], bbox)) {
          this->tiles[i]->get_3d_keypoints(atile_all_kps, atile_all_kps_desc);
        }
      } else {
        if (this->transformed_tile_overlaps_with(this->tiles[i], bbox, false)) {
          this->tiles[i]->get_3d_keypoints(atile_all_kps, atile_all_kps_desc);
        }
      }
    }
    if (apply_transform) {
      this->affine_transform_keypoints(atile_all_kps);
    }
  } else {
    cv::Mat A(3, 3, cv::DataType<double>::type);
    A.at<double>(0, 0) = 1.0;
    A.at<double>(0, 1) = 0.0;
    A.at<double>(0, 2) = 0.0;
    A.at<double>(1, 0) = 0.0;
    A.at<double>(1, 1) = 1.0;
    A.at<double>(1, 2) = 0.0;
    A.at<double>(2, 0) = 0.0;
    A.at<double>(2, 1) = 0.0;
    A.at<double>(2, 2) = 1.0;

    cv::Mat tmp_image;
    Render* render = new Render();
    if (apply_transform) {
     tmp_image = render->render(this, bbox, Resolution::PERCENT30);
    } else {
     tmp_image = render->render(this, bbox, Resolution::PERCENT30, true);
    }

    int black_pixels = 0;
    for (int r = 0; r < tmp_image.rows; r++) {
      for (int c = 0; c < tmp_image.cols; c++) {
        if (tmp_image.at<unsigned char>(r, c) == 0) black_pixels++;
      }
    }

    cv::Mat local_p_image;
    float scale_x = sift_parameters.scale_x;
    float scale_y = sift_parameters.scale_y;
    cv::resize(tmp_image, local_p_image, cv::Size(), scale_x, scale_y, CV_INTER_AREA);

    int rows = local_p_image.rows;
    int cols = local_p_image.cols;

    cv::Ptr<cv::Feature2D> p_sift = new cv::xfeatures2d::SIFT_Impl(
              sift_parameters.num_features,  // num_features --- unsupported.
              sift_parameters.num_octaves,  // number of octaves
              sift_parameters.contrast_threshold,  // contrast threshold.
              sift_parameters.edge_threshold,  // edge threshold.
              sift_parameters.sigma);  // sigma.

    cv::Mat sub_im = (local_p_image)(cv::Rect(0, 0, cols, rows));
    cv::Mat sub_im_mask = cv::Mat::ones(rows, cols, CV_8UC1);

    // lets try to mask out any background.
    for (int r = 0; r < local_p_image.rows; r++) {
      for (int c = 0; c < local_p_image.cols; c++) {
        if (local_p_image.at<unsigned char>(r, c) == 0) {
          for (int dx = -1; dx < 2; dx++) {
            for (int dy = -1; dy < 2; dy++) {
              int nc = r+dx;
              int nr = c+dy;
              if (nc < 0 || nc >= local_p_image.cols) continue;
              if (nr < 0 || nr >= local_p_image.rows) continue;
              sub_im_mask.at<unsigned char>(nr, nc) = 0;
            }
          }
        }
      }
    }


    std::vector<cv::KeyPoint> v_kps;
    cv::Mat m_kps_desc;

    p_sift->detectAndCompute(sub_im, sub_im_mask, v_kps, m_kps_desc);

    for (int j = 0; j < v_kps.size(); j++) {
      v_kps[j].pt.x /= (scale_x*0.3);
      v_kps[j].pt.y /= (scale_y*0.3);
      v_kps[j].pt.x += bbox.first.x;
      v_kps[j].pt.y += bbox.first.y;
      atile_all_kps.push_back(v_kps[j]);
      atile_all_kps_desc.push_back(m_kps_desc.row(j).clone());
    }
  }

  //std::vector<cv::Point2f> match_points_a, match_points_b;
  //double box_min_x = bbox.first.x;
  //double box_max_x = bbox.second.x;
  //double box_min_y = bbox.first.y;
  //double box_max_y = bbox.second.y;

  std::vector <cv::Mat > kps_desc_in_box_list;

  for (int i = 0; i < atile_all_kps.size(); i++) {
    kps_in_box.push_back(atile_all_kps[i]);
    kps_desc_in_box_list.push_back(atile_all_kps_desc[i]);
  }

  if (kps_in_box.size() < 4) {
    kps_in_box.clear();
    return;  // no points.
  }

  cv::vconcat(kps_desc_in_box_list, kps_desc_in_box);
}

void tfk::Section::find_3d_matches_in_box_cache(Section* neighbor,
    std::pair<cv::Point2f, cv::Point2f> sliding_bbox,
    std::vector<cv::Point2f>& test_filtered_match_points_a,
    std::vector<cv::Point2f>& test_filtered_match_points_b,
    bool use_cached, tfk::params sift_parameters, std::vector<Tile*>& tiles_loaded,
    std::mutex& tiles_loaded_mutex, std::vector<cv::KeyPoint>& prev_keypoints, cv::Mat& prev_desc,
    std::vector<cv::KeyPoint>& my_keypoints, cv::Mat& my_desc) {
  double ransac_thresh = 64.0;
  int num_filtered = 0;


  cv::Mat A(3, 3, cv::DataType<double>::type);
  A.at<double>(0, 0) = 1.0;
  A.at<double>(0, 1) = 0.0;
  A.at<double>(0, 2) = 0.0;
  A.at<double>(1, 0) = 0.0;
  A.at<double>(1, 1) = 1.0;
  A.at<double>(1, 2) = 0.0;
  A.at<double>(2, 0) = 0.0;
  A.at<double>(2, 1) = 0.0;
  A.at<double>(2, 2) = 1.0;


  this->get_3d_keypoints_for_box(sliding_bbox, my_keypoints,
      my_desc, use_cached, sift_parameters, tiles_loaded, tiles_loaded_mutex, true);
  std::vector<cv::KeyPoint>& atile_kps_in_overlap = my_keypoints;
  cv::Mat& atile_kps_desc_in_overlap = my_desc;

  if (atile_kps_in_overlap.size() < 12) return;

  if (prev_keypoints.size() == 0) {
    neighbor->get_3d_keypoints_for_box(sliding_bbox, prev_keypoints,
        prev_desc, use_cached, sift_parameters, tiles_loaded, tiles_loaded_mutex, false);
  }

  std::vector<cv::KeyPoint>& btile_kps_in_overlap = prev_keypoints;
  cv::Mat& btile_kps_desc_in_overlap = prev_desc;

  if (btile_kps_in_overlap.size() < 12) return;

  if (atile_kps_in_overlap.size() < 4 || btile_kps_in_overlap.size() < 4) return;

  std::vector< cv::DMatch > matches;
  match_features(matches,
                 atile_kps_desc_in_overlap,
                 btile_kps_desc_in_overlap,
                 0.65, false);

  //printf("Num matches is %zu\n", matches.size());

  // Bad don't add filtered matches.
  if (matches.size() < 12) return;

  std::vector<cv::Point2f> match_points_a, match_points_b;

  for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
    match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
    match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
  }

  bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
  tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, ransac_thresh, mask);


  for (int c = 0; c < match_points_a.size(); c++) {
    if (mask[c]) {
      num_filtered++;
      test_filtered_match_points_a.push_back(
          match_points_a[c]);
      test_filtered_match_points_b.push_back(
          match_points_b[c]);
    }
  }

  // Bad clear filtered matches.
  if (num_filtered < 0.05*match_points_a.size() || num_filtered < 12) {
     test_filtered_match_points_a.clear();
     test_filtered_match_points_b.clear();
  }

  free(mask);
}

void tfk::Section::find_3d_matches_in_box(Section* neighbor,
    std::pair<cv::Point2f, cv::Point2f> sliding_bbox,
    std::vector<cv::Point2f>& test_filtered_match_points_a,
    std::vector<cv::Point2f>& test_filtered_match_points_b,
    bool use_cached, tfk::params sift_parameters, std::vector<Tile*>& tiles_loaded,
    std::mutex& tiles_loaded_mutex) {

  double ransac_thresh = 64.0;
  int num_filtered = 0;

  std::vector<cv::KeyPoint> atile_kps_in_overlap;
  std::vector<cv::KeyPoint> btile_kps_in_overlap;
  cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;


  cv::Mat A(3, 3, cv::DataType<double>::type);
  A.at<double>(0, 0) = 1.0;
  A.at<double>(0, 1) = 0.0;
  A.at<double>(0, 2) = 0.0;
  A.at<double>(1, 0) = 0.0;
  A.at<double>(1, 1) = 1.0;
  A.at<double>(1, 2) = 0.0;
  A.at<double>(2, 0) = 0.0;
  A.at<double>(2, 1) = 0.0;
  A.at<double>(2, 2) = 1.0;



  this->get_3d_keypoints_for_box(sliding_bbox, atile_kps_in_overlap,
                                 atile_kps_desc_in_overlap, use_cached, sift_parameters,
                                 tiles_loaded, tiles_loaded_mutex, true);

  if (atile_kps_in_overlap.size() < 12) return;

  neighbor->get_3d_keypoints_for_box(sliding_bbox, btile_kps_in_overlap,
                                     btile_kps_desc_in_overlap, use_cached, sift_parameters,
                                     tiles_loaded, tiles_loaded_mutex, false);

  if (btile_kps_in_overlap.size() < 12) return;

  if (atile_kps_in_overlap.size() < 4 || btile_kps_in_overlap.size() < 4) return;

  std::vector< cv::DMatch > matches;
  match_features(matches,
                 atile_kps_desc_in_overlap,
                 btile_kps_desc_in_overlap,
                 0.92, true/*false*/);

  //printf("Num matches is %zu\n", matches.size());

  // Bad don't add filtered matches.
  if (matches.size() < 120) return;

  std::vector<cv::Point2f> match_points_a, match_points_b;

  for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
    match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
    match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
  }

  bool* mask = (bool*)calloc(match_points_a.size()+1, 1);
  tfk_simple_ransac_strict_ret_affine(match_points_a,
                                      match_points_b, ransac_thresh, mask);


  for (int c = 0; c < match_points_a.size(); c++) {
    if (mask[c]) {
      num_filtered++;
      test_filtered_match_points_a.push_back(
          match_points_a[c]);
      test_filtered_match_points_b.push_back(
          match_points_b[c]);
    }
  }

  // Bad clear filtered matches.
  if (num_filtered < /*0.05**/0.1*match_points_a.size() || num_filtered < 12) {
     test_filtered_match_points_a.clear();
     test_filtered_match_points_b.clear();
  }

  free(mask);
}

std::pair<cv::Point2f, cv::Point2f> tfk::Section::affine_transform_bbox(
    std::pair<cv::Point2f, cv::Point2f> bbox) {
  cv::Point2f corners[4];
  corners[0] = cv::Point2f(bbox.first.x, bbox.first.y);
  corners[1] = cv::Point2f(bbox.second.x, bbox.first.y);
  corners[2] = cv::Point2f(bbox.first.x, bbox.second.y);
  corners[3] = cv::Point2f(bbox.second.x, bbox.second.y);

  for (int i = 0; i < 4; i++) {
    corners[i] = this->affine_transform(corners[i]);
  }

  float min_x = corners[0].x;
  float max_x = corners[0].x;
  float min_y = corners[0].y;
  float max_y = corners[0].y;
  for (int i = 1; i < 4; i++) {
    if (corners[i].x < min_x) min_x = corners[i].x;
    if (corners[i].x > max_x) max_x = corners[i].x;

    if (corners[i].y < min_y) min_y = corners[i].y;
    if (corners[i].y > max_y) max_y = corners[i].y;
  }
  return std::make_pair(cv::Point2f(min_x, min_y), cv::Point2f(max_x, max_y));
}


void tfk::Section::affine_transform_keypoints(std::vector<cv::KeyPoint>& keypoints) {
  for (int i = 0; i < keypoints.size(); i++) {
    keypoints[i].pt = this->elastic_transform(keypoints[i].pt);
  }
}

void tfk::Section::get_elastic_matches_relative(Section* neighbor) {
  if (neighbor->real_section_id == this->real_section_id) return;

  auto bbox = this->get_bbox();

  // transforms section to align to neighbor.
  bbox = this->elastic_transform_bbox(bbox);

  double min_x = bbox.first.x;
  double min_y = bbox.first.y;
  double max_x = bbox.second.x;
  double max_y = bbox.second.y;
  std::vector<std::pair<double, double> > valid_boxes;
  printf("elastic matches relative genreating bounding boxes\n");
  for (double box_iter_x = min_x; box_iter_x < max_x + 24000; box_iter_x += 12000) {
    for (double box_iter_y = min_y; box_iter_y < max_y + 24000; box_iter_y += 12000) {
      valid_boxes.push_back(std::make_pair(box_iter_x, box_iter_y));
    }
  }

  printf("elastic matches relative genreating bounding boxes DONE\n");


  this->section_mesh_matches.clear();
  //std::mutex lock;
  cilk_for (int bbox_iter = 0; bbox_iter < valid_boxes.size(); bbox_iter++) {
    std::vector< cv::Point2f > filtered_match_points_a(0);
    std::vector< cv::Point2f > filtered_match_points_b(0);
    auto bbox = valid_boxes[bbox_iter];
    double box_iter_x = bbox.first;
    double box_iter_y = bbox.second;

    std::vector<Tile*> tiles_loaded;
    std::mutex tiles_loaded_mutex;
    std::pair<cv::Point2f, cv::Point2f> sliding_bbox =
        std::make_pair(cv::Point2f(box_iter_x, box_iter_y),
                       cv::Point2f(box_iter_x+12000, box_iter_y+12000));


    std::vector< cv::Point2f > test_filtered_match_points_a(0);
    std::vector< cv::Point2f > test_filtered_match_points_b(0);
    double bad_fraction = 2.0;


    for (int trial = 0; trial < 1; trial++) {
      // need to clear these to avoid pollution in event of multiple trials.
      test_filtered_match_points_a.clear();
      test_filtered_match_points_b.clear();
      double _bad_fraction = 2.0;
      // no need to init sift_parameters if we are passing 'true' for use_cached.
      if (trial == 0) {
        tfk::params sift_parameters;
        sift_parameters.num_features = 1;
        sift_parameters.num_octaves = 12;
        sift_parameters.contrast_threshold = 0.02;
        sift_parameters.edge_threshold = 5.0;
        sift_parameters.sigma = 1.1;
        sift_parameters.scale_x = 0.1;
        sift_parameters.scale_y = 0.1;

        this->find_3d_matches_in_box(neighbor, sliding_bbox, test_filtered_match_points_a,
            test_filtered_match_points_b, true, sift_parameters, tiles_loaded, tiles_loaded_mutex);
        //printf("test filtered match points size is %zu\n", test_filtered_match_points_a.size());
        if (test_filtered_match_points_a.size() > 12) _bad_fraction = 0.0;

      } else {
        tfk::params sift_parameters;
        sift_parameters.num_features = 1;
        sift_parameters.num_octaves = 12;
        sift_parameters.contrast_threshold = 0.015 + 0.005*(trial-1);
        sift_parameters.edge_threshold = 5 + (trial-1)*2;
        sift_parameters.sigma = 1.05 + (trial-1)*0.05;
        sift_parameters.scale_x = 0.25;
        sift_parameters.scale_y = 0.25;


        this->find_3d_matches_in_box(neighbor, sliding_bbox, test_filtered_match_points_a,
            test_filtered_match_points_b, false, sift_parameters, tiles_loaded, tiles_loaded_mutex);
        if (test_filtered_match_points_a.size() >= 1) _bad_fraction = 0.0;
      }
      if (trial > 0 && _bad_fraction < 0.2) {
        printf("Hurray recomputation helped us and got us error fraction from %f to %f trial %d\n",
               bad_fraction, _bad_fraction, trial);
      } else if (trial > 0 && _bad_fraction >= 0.2) {
        //  printf("Recomputation Failed and got us error fraction from %f to %f trial %d\n",
        //       bad_fraction, _bad_fraction, trial);
      }

      if (_bad_fraction < bad_fraction) {
        bad_fraction = _bad_fraction;
      }

      if (bad_fraction <= 0.2) {
        break;
      }
    }


    for (int i = 0; i < tiles_loaded.size(); i++) {
      tiles_loaded[i]->release_full_image();
    }

    if (bad_fraction <= 0.2) {
      //printf("bad fraction was small enough\n");
      for (int c = 0; c < test_filtered_match_points_a.size(); c++) {
        filtered_match_points_a.push_back(test_filtered_match_points_a[c]);
        filtered_match_points_b.push_back(test_filtered_match_points_b[c]);
      }
    }


  //  // divide all the points into triangles.
  //  std::map<int, std::vector<cv::Point2f> > parted_points_a;
  //  std::map<int, std::vector<cv::Point2f> > parted_points_b;

  //  for (int m = 0; m < filtered_match_points_a.size(); m++) {
  //    cv::Point2f my_pt = filtered_match_points_a[m];
  //    cv::Point2f n_pt = filtered_match_points_b[m];
  //    Triangle my_tri = this->triangle_mesh->find_triangle_post(my_pt);
  //    if (my_tri.index == -1) continue;
  //    parted_points_a[my_tri.index].push_back(my_pt);
  //    parted_points_b[my_tri.index].push_back(n_pt);
  //  }

  //// after dividing all the points filter out any triangles that don't have at least 3 points.
  //for (auto iter = parted_points_a.begin(); iter != parted_points_a.end(); ++iter) {
  //  std::vector<tfkTriangle>* triangles = this->triangle_mesh->triangles;
  //  std::vector<cv::Point2f>* mesh = this->triangle_mesh->mesh;

  //  int s = iter->first;
  //  cv::Point2f opt1 = (*mesh)[(*triangles)[s].index1];
  //  cv::Point2f opt2 = (*mesh)[(*triangles)[s].index2];
  //  cv::Point2f opt3 = (*mesh)[(*triangles)[s].index3];

  //  cv::Point2f pt1 = (*mesh)[(*triangles)[s].index1];
  //  cv::Point2f pt2 = (*mesh)[(*triangles)[s].index2];
  //  cv::Point2f pt3 = (*mesh)[(*triangles)[s].index3];
  //  if (parted_points_a.size() < 3) continue;

  //  for (int gs = 0; gs < 1000; gs++) {
  //     cv::Point2f d_pt1 = cv::Point2f(0.0, 0.0);
  //     cv::Point2f d_pt2 = cv::Point2f(0.0, 0.0);
  //     cv::Point2f d_pt3 = cv::Point2f(0.0, 0.0);

  //     for (int k = 0; k < parted_points_a[s].size(); k++) {
  //       cv::Point2f my_pt = parted_points_a[s][k];
  //       cv::Point2f n_pt = parted_points_b[s][k];
  //       float u, v, w;
  //       Barycentric(my_pt, opt1, opt2, opt3, u, v, w);
  //       if (std::isnan(u) || std::isnan(v) || std::isnan(w)) continue;
  //       cv::Point2f test_pt = pt1*u + pt2*v + pt3*w;
  //       d_pt1 += u*(n_pt - test_pt);
  //       d_pt2 += v*(n_pt - test_pt);
  //       d_pt3 += w*(n_pt - test_pt);
  //     }

  //     d_pt1 = d_pt1/(1.0*parted_points_a.size());
  //     d_pt2 = d_pt2/(1.0*parted_points_a.size());
  //     d_pt3 = d_pt3/(1.0*parted_points_a.size());

  //     pt1 += 0.4*d_pt1;
  //     pt2 += 0.4*d_pt2;
  //     pt3 += 0.4*d_pt3;
  //  }

  //  tfkMatch match1;
  //  match1.my_tri = (*triangles)[s];
  //  match1.my_barys[0] = (double)1.0;
  //  match1.my_barys[1] = 0.0;
  //  match1.my_barys[2] = 0.0;
  //  match1.dest_p = pt1;

  //  tfkMatch match2;
  //  match2.my_tri = (*triangles)[s];
  //  match2.my_barys[0] = 0.0;
  //  match2.my_barys[1] = (double)1.0;
  //  match2.my_barys[2] = 0.0;
  //  match2.dest_p = pt2;

  //  tfkMatch match3;
  //  match3.my_tri = (*triangles)[s];
  //  match3.my_barys[0] = 0.0;
  //  match3.my_barys[1] = 0.0;
  //  match3.my_barys[2] = (double)1.0;
  //  match3.dest_p = pt3;


  //  match1.my_section = (void*) this;
  //  match1.n_section = (void*) neighbor;
  //  match2.my_section = (void*) this;
  //  match2.n_section = (void*) neighbor;
  //  match3.my_section = (void*) this;
  //  match3.n_section = (void*) neighbor;
  //}

  //if (filtered_match_points_a.size() < 12 || filtered_match_points_b.size() < 12) continue;
 
  printf("before loop over filtered match points of size %zu\n", filtered_match_points_a.size());
  for (int m = 0; m < filtered_match_points_a.size(); m++) {
    cv::Point2f my_pt = filtered_match_points_a[m];
    cv::Point2f n_pt = filtered_match_points_b[m];

    Triangle my_tri = this->triangle_mesh->find_triangle_post(my_pt);
    Triangle n_tri = neighbor->triangle_mesh->find_triangle(my_pt);
    if (my_tri.index == -1 || n_tri.index == -1) continue;
    //printf("triangle index found\n");
    tfkMatch match;

    // find the triangle...
    std::vector<tfkTriangle>* triangles = this->triangle_mesh->triangles;
    std::vector<cv::Point2f>* mesh = this->triangle_mesh->mesh;

    std::vector<tfkTriangle>* n_triangles = neighbor->triangle_mesh->triangles;
    std::vector<cv::Point2f>* n_mesh = neighbor->triangle_mesh->mesh_orig;

    {
      int s = my_tri.index;
      float u, v, w;
      cv::Point2f pt1 = (*mesh)[(*triangles)[s].index1];
      cv::Point2f pt2 = (*mesh)[(*triangles)[s].index2];
      cv::Point2f pt3 = (*mesh)[(*triangles)[s].index3];
      Barycentric(my_pt, pt1, pt2, pt3, u, v, w);
      if (u <= 0 || v <= 0 || w <= 0) continue;
      int my_triangle_index = s;
      match.my_tri = (*triangles)[my_triangle_index];
      match.my_barys[0] = (double)1.0*u;
      match.my_barys[1] = (double)1.0*v;
      match.my_barys[2] = (double)1.0*w;
    }

    {
      int s = n_tri.index;
      float u, v, w;
      cv::Point2f pt1 = (*n_mesh)[(*n_triangles)[s].index1];
      cv::Point2f pt2 = (*n_mesh)[(*n_triangles)[s].index2];
      cv::Point2f pt3 = (*n_mesh)[(*n_triangles)[s].index3];
      Barycentric(n_pt, pt1, pt2, pt3, u, v, w);
      if (u <= 0 || v <= 0 || w <= 0) continue;
      int n_triangle_index = s;
      match.n_tri = (*n_triangles)[n_triangle_index];
      match.n_barys[0] = (double)1.0*u;
      match.n_barys[1] = (double)1.0*v;
      match.n_barys[2] = (double)1.0*w;
    }

    match.my_section = (void*) this;
    match.n_section = (void*) neighbor;
    match.dest_p = n_pt;
    section_mesh_matches_mutex->lock();
    this->section_mesh_matches.push_back(match);
    //printf("inserted a match\n");
    section_mesh_matches_mutex->unlock();
  }
  }
}

cv::Point2f tfk::Section::affine_transform_point(cv::Point2f pt){
  double a00 = this->total_affine_transform.at<double>(0, 0);
  double a01 = this->total_affine_transform.at<double>(0, 1);
  double a10 = this->total_affine_transform.at<double>(1, 0);
  double a11 = this->total_affine_transform.at<double>(1, 1);
  double offset_x = this->total_affine_transform.at<double>(0, 2);
  double offset_y = this->total_affine_transform.at<double>(1, 2);

  float new_x = pt.x * a00 + pt.y * a01 + offset_x;
  float new_y = pt.x * a10 + pt.y * a11 + offset_y;
  return cv::Point2f(new_x, new_y);
}

void tfk::Section::affine_transform_mesh() {
  for (int mesh_index = 0; mesh_index < this->triangle_mesh->mesh->size(); mesh_index++) {
        cv::Point2f pt = (*this->triangle_mesh->mesh)[mesh_index];

        double a00 = this->coarse_transform.at<double>(0, 0);
        double a01 = this->coarse_transform.at<double>(0, 1);
        double a10 = this->coarse_transform.at<double>(1, 0);
        double a11 = this->coarse_transform.at<double>(1, 1);
        double offset_x = this->coarse_transform.at<double>(0, 2);
        double offset_y = this->coarse_transform.at<double>(1, 2);

        float new_x = pt.x*a00 + pt.y * a01 + offset_x;
        float new_y = pt.x*a10 + pt.y * a11 + offset_y;
        (*this->triangle_mesh->mesh)[mesh_index] = cv::Point2f(new_x, new_y);
  }
  this->triangle_mesh->build_index_post();
}

void tfk::Section::affine_transform_mesh(cv::Mat transform) {
  std::cout << "tfk debug" << std::endl;
  std::cout << transform << std::endl;
  for (int mesh_index = 0; mesh_index < this->triangle_mesh->mesh->size(); mesh_index++) {
        cv::Point2f pt = (*this->triangle_mesh->mesh)[mesh_index];

        double a00 = transform.at<double>(0, 0);
        double a01 = transform.at<double>(0, 1);
        double a10 = transform.at<double>(1, 0);
        double a11 = transform.at<double>(1, 1);
        double offset_x = transform.at<double>(0, 2);
        double offset_y = transform.at<double>(1, 2);

        float new_x = pt.x*a00 + pt.y * a01 + offset_x;
        float new_y = pt.x*a10 + pt.y * a11 + offset_y;
        (*this->triangle_mesh->mesh)[mesh_index] = cv::Point2f(new_x, new_y);
  }
  this->triangle_mesh->build_index_post();
}

void tfk::Section::expand_mesh(){
  float hex_spacing = 3000.0;
  int mesh_size = (int)this->triangle_mesh->mesh->size();
  int triangle_size = (int)this->triangle_mesh->triangles->size();
  int edge_size = (int)this->triangle_mesh->triangle_edges->size();
  this->added_triangles = triangle_size;
  this->added_points= mesh_size;
  this->triangle_mesh->expand_bbox(hex_spacing, this->estimate_bbox);
  for (int i = mesh_size; i<this->triangle_mesh->mesh->size(); i++){
    (*this->triangle_mesh->mesh)[i] = this->affine_transform_point((*this->triangle_mesh->mesh)[i]);
    // this->unaligned->push_back(i);
  }
  this->elastic_align_unaligned(mesh_size, triangle_size, edge_size);
  printf("%d \n FIXING MESH\n ___________\n",(int)this->triangle_mesh->mesh->size()-mesh_size);
  this->triangle_mesh->build_index_post();
}

void tfk::Section::construct_triangles() {
  printf("called construct triangles\n");
  float hex_spacing = 6000.0;
  std::pair<cv::Point2f, cv::Point2f> bbox = this->get_bbox();
  triangle_mesh = new TriangleMesh(hex_spacing, bbox);
}

std::pair<cv::Point2f, cv::Point2f> tfk::Section::get_bbox() {
  float min_x = 0;
  float max_x = 0;
  float min_y = 0;
  float max_y = 0;

  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    if (tile->bad_2d_alignment) continue;
    std::pair<cv::Point2f, cv::Point2f> bbox = tile->get_bbox();
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


void tfk::Section::apply_affine_transforms(cv::Mat A) {
  // pass in affine transform matrix
  for (int i = 0; i < this->affine_transforms.size(); i++) {
    A = A*this->affine_transforms[i];
  }

  this->a00 = A.at<double>(0, 0);
  this->a01 = A.at<double>(0, 1);
  this->offset_x = A.at<double>(0, 2);
  this->a10 = A.at<double>(1, 0);
  this->a11 = A.at<double>(1, 1);
  this->offset_y = A.at<double>(1, 2);
}

void tfk::Section::apply_affine_transforms() {
  // init identity matrix.

  cv::Mat A(3, 3, cv::DataType<double>::type);
  A.at<double>(0, 0) = 1.0;
  A.at<double>(0, 1) = 0.0;
  A.at<double>(0, 2) = 0.0;
  A.at<double>(1, 0) = 0.0;
  A.at<double>(1, 1) = 1.0;
  A.at<double>(1, 2) = 0.0;
  A.at<double>(2, 0) = 0.0;
  A.at<double>(2, 1) = 0.0;
  A.at<double>(2, 2) = 1.0;

  this->a00 = A.at<double>(0, 0);
  this->a01 = A.at<double>(0, 1);
  this->offset_x = A.at<double>(0, 2);
  this->a10 = A.at<double>(1, 0);
  this->a11 = A.at<double>(1, 1);
  this->offset_y = A.at<double>(1, 2);
}

bool tfk::Section::load_elastic_mesh(Section* neighbor) {
  printf("Loaded elastic mesh for %d\n", this->real_section_id);

  TriangleMeshProto triangleMesh;


  std::fstream input(std::string(TFK_TMP_DIR)+"/emesh_"+std::to_string(this->real_section_id)+"_"+
                      std::to_string(neighbor->real_section_id)+".pbuf",
                      std::ios::in | std::ios::binary);

  if (!input.good()) return false;
  printf("right before parsing from input stream.\n");
  triangleMesh.ParseFromIstream(&input);
  printf("right before constructing triangle mesh in load\n");
  triangle_mesh = new TriangleMesh(triangleMesh);
  printf("section %d constructed triangle mesh\n", this->real_section_id);

  return true;
}



void tfk::Section::align_3d(Section* neighbor) {
  // check to see if the elastic transforms exist.

  if (!load_elastic_mesh(neighbor)) {
    // do the affine align with the neighbor.


    int trials = 10;
    
    for (int i = 0; i < trials; i++) {
      bool success = this->coarse_affine_align(neighbor);
      if (success) break;
      printf("trial %d failed, retrying.\n", i);
    }

    // affine transform the mesh.
    this->affine_transform_mesh(this->coarse_transform);

    // do the fine affine align with the neighbor.
    //this->fine_affine_align(neighbor);

    //// fine affine transform the mesh.
    ////this->affine_transform_mesh(this->fine_transform);
    //this->total_affine_transform = this->fine_transform * this->coarse_transform;

    //this->affine_transform_mesh(this->fine_transform);

    // do the elastic alignment.

    //this->get_elastic_matches_relative(neighbor);
    //elastic_gradient_descent_section(this, neighbor);
    //this->triangle_mesh->build_index_post();

    this->get_elastic_matches_relative(neighbor);
    elastic_gradient_descent_section(this, neighbor);
    this->triangle_mesh->build_index_post();

    this->get_elastic_matches_relative(neighbor);
    elastic_gradient_descent_section(this, neighbor);
    this->triangle_mesh->build_index_post();

    //this->get_elastic_matches_relative(neighbor);
    //elastic_gradient_descent_section(this, neighbor);
    //this->triangle_mesh->build_index_post();

    //this->get_elastic_matches_relative(neighbor);
    //elastic_gradient_descent_section(this, neighbor);
    //this->triangle_mesh->build_index_post();

    //this->get_elastic_matches_relative(neighbor);
    //elastic_gradient_descent_section(this, neighbor);
    //this->triangle_mesh->build_index_post();

    //this->get_elastic_matches_relative(neighbor);
    //elastic_gradient_descent_section(this, neighbor);
    //this->triangle_mesh->build_index_post();

    this->save_elastic_mesh(neighbor);

  } else {
    //this->load_coarse_transform(neighbor);
    //this->load_fine_transform(neighbor);
    //this->total_affine_transform = this->fine_transform * this->coarse_transform;
    //this->apply_affine_transforms();
  }
}

bool tfk::Section::load_coarse_transform(Section* neighbor) {
  printf("Loaded coarse affine transform for %d\n", this->real_section_id);

  std::string path = std::string(TFK_TMP_DIR) + "/coarse_transform_" +
      std::to_string(this->real_section_id) + "_" + std::to_string(neighbor->real_section_id);

  cv::FileStorage fs(path, cv::FileStorage::READ);
  fs["transform"] >> this->coarse_transform;
  fs.release();
  if (this->coarse_transform.rows == 0){
    cv::Mat A(3, 3, cv::DataType<double>::type);
    A.at<double>(0,0) = 1.0;
    A.at<double>(0,1) = 0.0;
    A.at<double>(0,2) = 0.0;
    A.at<double>(1,0) = 0.0;
    A.at<double>(1,1) = 1.0;
    A.at<double>(1,2) = 0.0;
    A.at<double>(2,0) = 0.0;
    A.at<double>(2,1) = 0.0;
    A.at<double>(2,2) = 1.0;
    this->coarse_transform = A.clone();
  }

  return true;
}

bool tfk::Section::load_fine_transform(Section* neighbor) {
  printf("Loaded coarse affine transform for %d\n", this->real_section_id);

 std::string path = std::string(TFK_TMP_DIR) + "/fine_transform_" +
                    std::to_string(this->real_section_id) + "_" + std::to_string(neighbor->real_section_id);

  cv::FileStorage fs(path, cv::FileStorage::READ);
  fs["transform"] >> this->fine_transform;
  fs.release();
  if (this->fine_transform.rows == 0){
    cv::Mat A(3, 3, cv::DataType<double>::type);
    A.at<double>(0,0) = 1.0;
    A.at<double>(0,1) = 0.0;
    A.at<double>(0,2) = 0.0;
    A.at<double>(1,0) = 0.0;
    A.at<double>(1,1) = 1.0;
    A.at<double>(1,2) = 0.0;
    A.at<double>(2,0) = 0.0;
    A.at<double>(2,1) = 0.0;
    A.at<double>(2,2) = 1.0;
    this->fine_transform= A.clone();
  }

  return true;
}

void tfk::Section::elastic_align_unaligned(int mesh_start_index, int triangle_start_index, int edge_start_index){
  double intra_slice_weight = 1.0;
  double intra_slice_winsor = 200.0;
  int max_iterations = 10000; //ORIGINALL 5000
  double stepsize = 0.1;
  double momentum = 0.9;

  if((int)this->triangle_mesh->mesh->size() == mesh_start_index){
    printf("No alignment issues in section %d\n", this->real_section_id);
    return;
  }

  // INIT:
  std::map<int, double> gradient_momentum;
  std::vector<tfkTriangle>* relevant_triangles = new std::vector<tfkTriangle>();
  std::vector<std::pair<int,int>>* relevant_edges = new std::vector<std::pair<int,int>>();
  this->mesh_old = new std::vector<cv::Point2f>();

  for (int j = 0; j < this->triangle_mesh->mesh_orig->size(); j++){
    this->mesh_old->push_back((*(this->triangle_mesh->mesh))[j]);
  }

  // Find all edges that connect to not matched points
  for (int i = edge_start_index; i < this->triangle_mesh->triangle_edges->size(); i++){
    int p1 = (*(this->triangle_mesh->triangle_edges))[i].first;
    int p2 = (*(this->triangle_mesh->triangle_edges))[i].second;
    relevant_edges->push_back(std::make_pair(p1,p2));
  }

  for (int i = triangle_start_index; i < this->triangle_mesh->triangles->size(); i++) {
    relevant_triangles->push_back((*(this->triangle_mesh->triangles))[i]);
  }

  this->gradients = new cv::Point2f[this->triangle_mesh->mesh_orig->size()];
  this->gradients_with_momentum = new cv::Point2f[this->triangle_mesh->mesh_orig->size()];

  for (int j = 0; j < this->triangle_mesh->mesh_orig->size(); j++) {
    this->gradients[j] = cv::Point2f(0.0,0.0);
    this->gradients_with_momentum[j] = cv::Point2f(0.0,0.0);
  }

  // Optimize Variables
  double* rest_lengths = new double[relevant_edges->size()];
  double* rest_areas = new double[relevant_triangles->size()];

  for (int j = 0; j < relevant_edges->size(); j++) {
    cv::Point2f p1 = this->affine_transform_point((*(this->triangle_mesh->mesh_orig))[(*relevant_edges)[j].first]);
    cv::Point2f p2 = this->affine_transform_point((*(this->triangle_mesh->mesh_orig))[(*relevant_edges)[j].second]);
    double dx = p1.x-p2.x;
    double dy = p1.y-p2.y;
    double len = std::sqrt(dx*dx+dy*dy);
    rest_lengths[j] = len;
  }
  for (int j = 0; j < relevant_triangles->size(); j++) {
    tfkTriangle tri = (*relevant_triangles)[j];
    cv::Point2f p1 = this->affine_transform_point((*(this->triangle_mesh->mesh_orig))[tri.index1]);
    cv::Point2f p2 = this->affine_transform_point((*(this->triangle_mesh->mesh_orig))[tri.index2]);
    cv::Point2f p3 = this->affine_transform_point((*(this->triangle_mesh->mesh_orig))[tri.index3]);
    rest_areas[j] = computeTriangleArea(p1,p2,p3);
  }

  // Do the alignment
  double prev_cost = 0.0;
  for (int iter = 0; iter < max_iterations; iter++) {
    double cost = 0.0;
    // reset old gradients
    for (int j = mesh_start_index; j < this->triangle_mesh->mesh->size(); j++) {
      (this->gradients)[j] = cv::Point2f(0.0,0.0);
    }
    {
      Section* section = this;
      // internal_mesh_derivs
      double all_weight = intra_slice_weight;
      double sigma = intra_slice_winsor;
      std::vector<cv::Point2f>* mesh = section->triangle_mesh->mesh;

      std::vector<std::pair<int, int> >* triangle_edges = relevant_edges;
      std::vector<tfkTriangle >* triangles =  relevant_triangles;

      //// update all edges
      for (int j = 0; j < triangle_edges->size(); j++) {
        cost += internal_mesh_derivs(mesh, gradients, (*triangle_edges)[j], rest_lengths[j],
            all_weight/section->triangle_mesh->triangle_edges->size(), sigma);
      }

      //// update all triangles
      for (int j = 0; j < triangles->size(); j++) {
        int triangle_indices[3] = {(*triangles)[j].index1,
          (*triangles)[j].index2,
          (*triangles)[j].index3};
        cost += area_mesh_derivs(mesh, gradients, triangle_indices, rest_areas[j],
            all_weight/section->triangle_mesh->triangles->size());
      }
    }

    if (iter == 0) prev_cost = cost+10.0;

    if (cost <= prev_cost) {
      stepsize *= 1.1;
      if (stepsize > 10.0) {
        stepsize = 10.0;
      }
      // TODO(TFK): momentum.

      //for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
      //     it != section_data_map.end(); ++it) {
      {
        Section* section = this;
        std::vector<cv::Point2f>* mesh = section->triangle_mesh->mesh; 
        std::vector<cv::Point2f>* mesh_old = section->mesh_old;
        cv::Point2f* gradients = section->gradients;
        cv::Point2f* gradients_with_momentum = section->gradients_with_momentum;
        for (int j = mesh_start_index; j < mesh->size(); j++) {
          gradients_with_momentum[j] = gradients[j] + momentum*gradients_with_momentum[j];
        }

        for (int j = mesh_start_index; j < mesh->size(); j++) {
          (*mesh_old)[j] = (*mesh)[j];
        }
        for (int j = mesh_start_index; j < mesh->size(); j++) {
          (*mesh)[j].x -= (float)(stepsize * (gradients_with_momentum)[j].x);
          (*mesh)[j].y -= (float)(stepsize * (gradients_with_momentum)[j].y);
        }
      }

      if (max_iterations - iter < 1000) {
        if (prev_cost - cost > 1.0/100) {
          max_iterations += 1000;
        }
      }

      if (iter%100 == 0) {

        printf("Good step old cost %f, new cost %f, iteration %d, max %d\n", prev_cost, cost, iter, max_iterations);
      }
      prev_cost = cost;
    } else {
      stepsize *= 0.5;
      // bad step undo.
      //for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
      //     it != section_data_map.end(); ++it) {
      {
        Section* section = this;
        std::vector<cv::Point2f>* mesh = section->triangle_mesh->mesh;
        std::vector<cv::Point2f>* mesh_old = section->mesh_old;
        cv::Point2f* gradients_with_momentum = section->gradients_with_momentum;
        for (int j = mesh_start_index; j < mesh->size(); j++) {
          gradients_with_momentum[j] = cv::Point2f(0.0,0.0);
        }

        //if (mesh_old->size() != mesh->size()) continue;
        for (int j = mesh_start_index; j < mesh->size(); j++) {
          (*mesh)[j] = (*mesh_old)[j];
        }
      }
      if (iter%1000 == 0) {
        printf("Bad step old cost %f, new cost %f, iteration %d\n", prev_cost, cost, iter);
      }
    }
    
  }

}

void tfk::Section::fine_affine_align(Section* neighbor){
  printf("STARTING THE FINE ALIGNMENT IT SO FINE (I hope)\n");
  if (neighbor == NULL || neighbor->real_section_id == this->real_section_id) {
    cv::Mat A(3, 3, cv::DataType<double>::type);
    A.at<double>(0,0) = 1.0;
    A.at<double>(0,1) = 0.0;
    A.at<double>(0,2) = 0.0;
    A.at<double>(1,0) = 0.0;
    A.at<double>(1,1) = 1.0;
    A.at<double>(1,2) = 0.0;
    A.at<double>(2,0) = 0.0;
    A.at<double>(2,1) = 0.0;
    A.at<double>(2,2) = 1.0;

    //printf("Printing out A\n");
    //std::cout << A << std::endl;

    this->fine_transform= A.clone(); 
    return;
  }

  int box_size = 48000;
  //int tile_size = 12000;
 
  auto bbox = this->get_bbox();
  // transforms section to align to neighbor.
  bbox = this->elastic_transform_bbox(bbox);

  double min_x = bbox.first.x; 
  double min_y = bbox.first.y; 
  double max_x = bbox.second.x; 
  double max_y = bbox.second.y;
  std::vector<std::pair<double, double> > valid_boxes;
  for (double box_iter_x = min_x; box_iter_x < max_x + box_size; box_iter_x += box_size/2) {
    for (double box_iter_y = min_y; box_iter_y < max_y + box_size; box_iter_y += box_size/2) {
      valid_boxes.push_back(std::make_pair(box_iter_x, box_iter_y));
    }
  }
  std::vector<cv::Point2f> filtered_match_points_a(0);
  std::vector<cv::Point2f> filtered_match_points_b(0);

  std::mutex lock;
  cilk_for (int bbox_iter = 0; bbox_iter<valid_boxes.size(); bbox_iter++){
    auto bbox = valid_boxes[bbox_iter];
    double box_iter_x = bbox.first;
    double box_iter_y = bbox.second;
    std::vector<Tile*> tiles_loaded;
    std::mutex tiles_loaded_mutex;
    std::pair<cv::Point2f, cv::Point2f> sliding_bbox =
        std::make_pair(cv::Point2f(box_iter_x, box_iter_y),
                       cv::Point2f(box_iter_x + box_size, box_iter_y + box_size));

    std::vector<cv::Point2f> test_filtered_match_points_a(0);
    std::vector<cv::Point2f> test_filtered_match_points_b(0);

    tfk::params sift_parameters; //We don't actually need these...
    this->find_3d_matches_in_box(neighbor, sliding_bbox, test_filtered_match_points_a,
                                 test_filtered_match_points_b, true, sift_parameters, tiles_loaded, tiles_loaded_mutex);
    for (int i = 0; i < tiles_loaded.size(); i++){
      tiles_loaded[i]->release_full_image();
    }
    //for (int c = 0; c < test_filtered_match_points_a.size(); c++){
    //  filtered_match_points_a.push_back(test_filtered_match_points_a[c]);
    //  filtered_match_points_b.push_back(test_filtered_match_points_b[c]);
    //}

    if (test_filtered_match_points_a.size() == 0) {
      printf("No filtered matches, unfiltered size is %zu\n", test_filtered_match_points_a.size());
      continue;
    }

    int num_filtered = 0;
    bool *mask = (bool *)calloc(test_filtered_match_points_a.size() + 1, 1);
    tfk_simple_ransac_strict_ret_affine(test_filtered_match_points_a, test_filtered_match_points_b, 64.0, mask);
    for (int c = 0; c < test_filtered_match_points_a.size(); c++){
      if (mask[c]){
        num_filtered++;
      }
    }

    
    printf("unfiltered size %zu filtered size is %d\n", test_filtered_match_points_a.size(), num_filtered);

    if (num_filtered < 12) {
      free(mask);
      continue;
    }
    lock.lock();
    for (int c = 0; c < test_filtered_match_points_a.size(); c++){
      if (mask[c]){
        filtered_match_points_a.push_back(
            test_filtered_match_points_a[c]);
        filtered_match_points_b.push_back(
            test_filtered_match_points_b[c]);
      }
    }
    lock.unlock();
    free(mask);
  }
  cv::Mat section_transform;
  //printf("%d points for section %d\n", filtered_match_points_a.size(), this->real_section_id);
  bool res = cv::computeAffineTFK(filtered_match_points_a, filtered_match_points_b, section_transform);

  if (!res) {
    printf("The result of the affine transform was false. size of filtered matches %zu %zu\n", filtered_match_points_a.size(), filtered_match_points_b.size());
  }

  cv::Mat A(3, 3, cv::DataType<double>::type);

  A.at<double>(0, 0) = section_transform.at<double>(0, 0);
  A.at<double>(0, 1) = section_transform.at<double>(0, 1);
  A.at<double>(0, 2) = section_transform.at<double>(0, 2);
  A.at<double>(1, 0) = section_transform.at<double>(1, 0);
  A.at<double>(1, 1) = section_transform.at<double>(1, 1);
  A.at<double>(1, 2) = section_transform.at<double>(1, 2);
  A.at<double>(2, 0) = 0.0;
  A.at<double>(2, 1) = 0.0;
  A.at<double>(2, 2) = 1.0;

 this->fine_transform = A.clone();
 std::string path = std::string(TFK_TMP_DIR) + "/fine_transform_" +
                    std::to_string(this->real_section_id) + "_" + std::to_string(neighbor->real_section_id);
 cv::FileStorage fs(path, cv::FileStorage::WRITE);
 cv::write(fs, "transform", this->fine_transform);
 fs.release();
}


// Find affine transform for this section that aligns it to neighbor.
bool tfk::Section::coarse_affine_align(Section* neighbor) {
  if (neighbor == NULL || neighbor->real_section_id == this->real_section_id) {
    cv::Mat A(3, 3, cv::DataType<double>::type);
    A.at<double>(0, 0) = 1.0;
    A.at<double>(0, 1) = 0.0;
    A.at<double>(0, 2) = 0.0;
    A.at<double>(1, 0) = 0.0;
    A.at<double>(1, 1) = 1.0;
    A.at<double>(1, 2) = 0.0;
    A.at<double>(2, 0) = 0.0;
    A.at<double>(2, 1) = 0.0;
    A.at<double>(2, 2) = 1.0;

    printf("Printing out A\n");
    std::cout << A << std::endl;

    this->coarse_transform = A.clone();
    return true;
  }

  // a = neighbor.
  std::vector <cv::KeyPoint > atile_kps_in_overlap;
  std::vector <cv::Mat > atile_kps_desc_in_overlap_list;

  // b = this
  std::vector <cv::KeyPoint > btile_kps_in_overlap;
  std::vector <cv::Mat > btile_kps_desc_in_overlap_list;

  for (int i = 0; i < this->tiles.size(); i++) {
    if (this->tiles[i]->bad_2d_alignment) continue;
    this->tiles[i]->get_3d_keypoints_limit(atile_kps_in_overlap, atile_kps_desc_in_overlap_list, 1);
  }

  for (int i = 0; i < neighbor->tiles.size(); i++) {
    if (neighbor->tiles[i]->bad_2d_alignment) continue;
    neighbor->tiles[i]->get_3d_keypoints_limit(btile_kps_in_overlap, btile_kps_desc_in_overlap_list, 1);
  }

  cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
  cv::vconcat(atile_kps_desc_in_overlap_list, (atile_kps_desc_in_overlap));
  cv::vconcat(btile_kps_desc_in_overlap_list, (btile_kps_desc_in_overlap));

  printf("before match features in coarse\n");

  //std::vector< std::vector<cv::DMatch> > matches_multiple(100);
  std::vector< cv::DMatch > matches;

  //std::vector<std::vector<cv::Point2f> > match_points_a_list(100);
  //std::vector<std::vector<cv::Point2f> > match_points_b_list(100);
  //cilk_for (int i = 0; i < this->tiles.size(); i += 100) {
  //  std::vector <cv::KeyPoint > atile_kps_in_overlap;
  //  std::vector <cv::Mat > atile_kps_desc_in_overlap_list;

  //  int end = i+100;
  //  if (end > this->tiles.size()) end = this->tiles.size(); 
  //  int start = i;

  //  for (int k = start; k < end;/*this->tiles.size();*/ k++) {
  //    if (this->tiles[k]->bad_2d_alignment) continue;
  //    this->tiles[k]->get_3d_keypoints(atile_kps_in_overlap, atile_kps_desc_in_overlap_list);
  //  }

  //  cv::Mat atile_kps_desc_in_overlap;

  //  // Grab the matches.
  //  for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
  //    match_points_a_list[i].push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
  //    match_points_b_list[i].push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
  //  }

  //}

    match_features_brute_parallel(matches,
                   atile_kps_desc_in_overlap,
                   btile_kps_desc_in_overlap,
                   0.92);

    //match_features(matches,
    //               atile_kps_desc_in_overlap,
    //               btile_kps_desc_in_overlap,
    //               0.92, false);
  //for (int i = 0; i < matches_multiple.size(); i++) {
  //  for (int j = 0; j < matches_multiple[i].size(); j++) {
  //    matches.push_back(matches_multiple[i][j]);
  //  }
  //}

  printf("after match features in coarse\n");

  // Filter the matches with RANSAC
  std::vector<cv::Point2f> match_points_a, match_points_b;

  //for (int i = 0; i < 100; i++) {
  //  for (int j = 0; j < match_points_a_list[i].size(); j++) {
  //    match_points_a.push_back(match_points_a_list[i][j]);
  //    match_points_b.push_back(match_points_b_list[i][j]);
  //  }
  //}

  // Grab the matches.
  for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
    match_points_a.push_back(atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
    match_points_b.push_back(btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
  }

  bool* mask = (bool*)calloc(match_points_a.size()+1, 1);

  // pre-filter matches with very forgiving ransac threshold.
  tfk_simple_ransac_strict_ret_affine(match_points_a, match_points_b, 64.0, mask);
  std::vector< cv::Point2f > filtered_match_points_a_pre(0);
  std::vector< cv::Point2f > filtered_match_points_b_pre(0);
  int num_filtered = 0;
  for (int c = 0; c < match_points_a.size(); c++) {
    if (mask[c]) {
      num_filtered++;
      filtered_match_points_a_pre.push_back(
          match_points_a[c]);
      filtered_match_points_b_pre.push_back(
          match_points_b[c]);
    }
  }
  free(mask);

  mask = (bool*)calloc(match_points_a.size()+1, 1);

  if (num_filtered < 32) {
    return false;
  }

  tfk_simple_ransac_strict_ret_affine(filtered_match_points_a_pre,
                                      filtered_match_points_b_pre, 64.0, mask);

  std::vector< cv::Point2f > filtered_match_points_a(0);
  std::vector< cv::Point2f > filtered_match_points_b(0);

  num_filtered = 0;
  for (int c = 0; c < filtered_match_points_a_pre.size(); c++) {
    if (mask[c]) {
      num_filtered++;
      filtered_match_points_a.push_back(
          filtered_match_points_a_pre[c]);
      filtered_match_points_b.push_back(
          filtered_match_points_b_pre[c]);
    }
  }

  if (num_filtered < 12) {
    // printf("Not enough matches %d for section %d with thresh\n", num_filtered, this->section_id);
    return false;
  } else {
    // printf("Got enough matches %d for section %d with thresh\n", num_filtered, this->section_id);
  }

  cv::Mat section_transform;

  cv::computeAffineTFK(filtered_match_points_a, filtered_match_points_b, section_transform);

  cv::Mat A(3, 3, cv::DataType<double>::type);

  A.at<double>(0, 0) = section_transform.at<double>(0, 0);
  A.at<double>(0, 1) = section_transform.at<double>(0, 1);
  A.at<double>(0, 2) = section_transform.at<double>(0, 2);
  A.at<double>(1, 0) = section_transform.at<double>(1, 0);
  A.at<double>(1, 1) = section_transform.at<double>(1, 1);
  A.at<double>(1, 2) = section_transform.at<double>(1, 2);
  A.at<double>(2, 0) = 0.0;
  A.at<double>(2, 1) = 0.0;
  A.at<double>(2, 2) = 1.0;

  printf("Printing out A\n");
  std::cout << A << std::endl;

  this->coarse_transform = A.clone();

  std::string path = std::string(TFK_TMP_DIR) + "/coarse_transform_" +
      std::to_string(this->real_section_id) + "_" + std::to_string(neighbor->real_section_id);
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  cv::write(fs, "transform", this->coarse_transform);
  fs.release();
  return true;
}

// returns the offset vector between the images in the scale of the images
cv::Point2f tfk::Section::compute_tile_matches_pair(Tile* a_tile, Tile* b_tile,
    std::vector< cv::KeyPoint >& a_tile_keypoints, std::vector< cv::KeyPoint >& b_tile_keypoints,
    cv::Mat& a_tile_desc, cv::Mat& b_tile_desc,
    std::vector< cv::Point2f > &filtered_match_points_a,
    std::vector< cv::Point2f > &filtered_match_points_b, float ransac_thresh) {
  std::vector<int> neighbors = get_all_close_tiles(a_tile->tile_id);

  cv::Point2f ZERO = cv::Point2f(0.0, 0.0);

  if (a_tile_keypoints.size() < MIN_FEATURES_NUM) return ZERO;
  if (b_tile_keypoints.size() < MIN_FEATURES_NUM) return ZERO;

  // Filter the features, so that only features that are in the
  //   overlapping tile will be matches.
  std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;

  atile_kps_in_overlap.reserve(a_tile_keypoints.size());
  btile_kps_in_overlap.reserve(b_tile_keypoints.size());

  // atile_kps_in_overlap.clear(); btile_kps_in_overlap.clear();
  cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;

  { // Begin scoped block A.
    // Compute bounding box of overlap
    int overlap_x_start = a_tile->x_start > b_tile->x_start ?
                              a_tile->x_start : b_tile->x_start;
    int overlap_x_finish = a_tile->x_finish < b_tile->x_finish ?
                              a_tile->x_finish : b_tile->x_finish;
    int overlap_y_start = a_tile->y_start > b_tile->y_start ?
                              a_tile->y_start : b_tile->y_start;
    int overlap_y_finish = a_tile->y_finish < b_tile->y_finish ?
                              a_tile->y_finish : b_tile->y_finish;
    // Add 50-pixel offset
    const int OFFSET = 50;  // CHANGED FROM 50.
    overlap_x_start -= OFFSET;
    overlap_x_finish += OFFSET;
    overlap_y_start -= OFFSET;
    overlap_y_finish += OFFSET;

    std::vector< cv::Mat > atile_kps_desc_in_overlap_list;
    atile_kps_desc_in_overlap_list.reserve(a_tile_keypoints.size());
    std::vector< cv::Mat > btile_kps_desc_in_overlap_list;
    btile_kps_desc_in_overlap_list.reserve(b_tile_keypoints.size());

    // Filter the points in a_tile.
    for (size_t pt_idx = 0; pt_idx < a_tile_keypoints.size(); ++pt_idx) {
      cv::Point2f pt = a_tile_keypoints[pt_idx].pt;
      if (bbox_contains(pt.x + a_tile->x_start,
                        pt.y + a_tile->y_start,  // transformed_pt[0],
                        overlap_x_start, overlap_x_finish,
                        overlap_y_start, overlap_y_finish)) {
        atile_kps_in_overlap.push_back(a_tile_keypoints[pt_idx]);
        atile_kps_desc_in_overlap_list.push_back(
            a_tile_desc.row(pt_idx).clone());
      }
    }
    cv::vconcat(atile_kps_desc_in_overlap_list,
        (atile_kps_desc_in_overlap));

    // Filter the points in b_tile.
    for (size_t pt_idx = 0; pt_idx < b_tile_keypoints.size(); ++pt_idx) {
      cv::Point2f pt = b_tile_keypoints[pt_idx].pt;
      if (bbox_contains(pt.x + b_tile->x_start,
                        pt.y + b_tile->y_start,  // transformed_pt[0],
                        overlap_x_start, overlap_x_finish,
                        overlap_y_start, overlap_y_finish)) {
        btile_kps_in_overlap.push_back(b_tile_keypoints[pt_idx]);
        btile_kps_desc_in_overlap_list.push_back(b_tile_desc.row(pt_idx).clone());
      }
    }
    cv::vconcat(btile_kps_desc_in_overlap_list,
        (btile_kps_desc_in_overlap));
  }  // End scoped block A

  printf("keypoints a: %zu, keypoints b: %zu\n", atile_kps_in_overlap.size(),
                                               btile_kps_in_overlap.size());

  if (atile_kps_in_overlap.size() < MIN_FEATURES_NUM) return ZERO;
  if (btile_kps_in_overlap.size() < MIN_FEATURES_NUM) return ZERO;

  float trial_rod;
  for (int trial = 0; trial < 4; trial++) {
    if (trial == 0) trial_rod = 0.7;
    if (trial == 1) trial_rod = 0.8;
    if (trial == 2) trial_rod = 0.92;
    if (trial == 3) trial_rod = 0.96;
    // Match the features
    std::vector< cv::DMatch > matches;
    match_features(matches,
                   atile_kps_desc_in_overlap,
                   btile_kps_desc_in_overlap,
                   trial_rod, false);

    // Filter the matches with RANSAC
    std::vector<cv::Point2f> match_points_a, match_points_b;
    for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
      match_points_a.push_back(
          atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
      match_points_b.push_back(
          btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
    }

    if (matches.size() < MIN_FEATURES_NUM) {
      continue;
    }

    bool* mask = (bool*) calloc(match_points_a.size(), 1);
    double thresh = ransac_thresh;  // 5.0;
    cv::Point2f relative_offset = tfk_simple_ransac(match_points_a, match_points_b, thresh, mask);


    filtered_match_points_a.clear();
    filtered_match_points_b.clear();

    int num_matches_filtered = 0;
    // Use the output mask to filter the matches
    for (size_t i = 0; i < matches.size(); ++i) {
      if (mask[i]) {
        num_matches_filtered++;
        filtered_match_points_a.push_back(
            atile_kps_in_overlap[matches[i].queryIdx].pt);
        filtered_match_points_b.push_back(
            btile_kps_in_overlap[matches[i].trainIdx].pt);
      }
    }
    free(mask);
    if (num_matches_filtered >= MIN_FEATURES_NUM &&
        num_matches_filtered > 0.1*match_points_a.size()) {
      return relative_offset;
    } else {
      filtered_match_points_a.clear();
      filtered_match_points_b.clear();
    }
  }

  return ZERO;
}

void tfk::Section::compute_tile_matches2(Tile* a_tile) {
  std::vector<Tile*> _neighbors = get_all_close_tiles(a_tile);
  std::vector<int> neighbors;
  for (int i = 0; i < _neighbors.size(); i++) {
    neighbors.push_back(_neighbors[i]->tile_id);
  }

  std::set<int> good_neighbors;
  for (int i = 0; i < a_tile->edges.size(); i++) {
    good_neighbors.insert(a_tile->edges[i].neighbor_id);
  }

  for (int i = 0; i < _neighbors.size(); i++) {
    // check number that I have.
    for (int j = 0; j < _neighbors[i]->edges.size(); j++) {
      if (_neighbors[i]->edges[j].neighbor_id == a_tile->tile_id) {
        good_neighbors.insert(_neighbors[i]->tile_id);
      }
    }
  }

  int neighbor_success_count = good_neighbors.size();
  std::vector<int> bad_indices(0);

  for (int i = 0; i < _neighbors.size(); i++) {
    if (good_neighbors.find(_neighbors[i]->tile_id) == good_neighbors.end()) {
      bad_indices.push_back(i);
    }
  }

  if (neighbor_success_count < neighbors.size()*2.0/4.0 || neighbor_success_count == 0) {
    std::vector<cv::KeyPoint> a_tile_keypoints;
    cv::Mat a_tile_desc;
    std::mutex lock;
    tfk::params new_params;
    new_params.scale_x = 1.0;
    new_params.scale_y = 1.0;
    new_params.num_features = 1;
    new_params.num_octaves = 6;
    new_params.contrast_threshold = 0.01;
    new_params.edge_threshold = 20;
    new_params.sigma = 1.2;
    printf("invoked slow path\n");
    a_tile->compute_sift_keypoints2d_params(new_params, a_tile_keypoints, a_tile_desc, a_tile);

    cilk_for (int k = 0; k < bad_indices.size(); k++) {
      int i = bad_indices[k];
      int btile_id = neighbors[i];
      Tile* b_tile = this->tiles[btile_id];

      // need to compute keypoints for both of these tiles.
      std::vector<cv::KeyPoint> b_tile_keypoints;
      cv::Mat b_tile_desc;

      b_tile->compute_sift_keypoints2d_params(new_params, b_tile_keypoints, b_tile_desc, a_tile);

      if (a_tile_keypoints.size() < MIN_FEATURES_NUM) continue;
      if (b_tile_keypoints.size() < MIN_FEATURES_NUM) continue;
      std::vector< cv::Point2f > filtered_match_points_a(0);
      std::vector< cv::Point2f > filtered_match_points_b(0);

      this->compute_tile_matches_pair(a_tile, b_tile,
        a_tile_keypoints, b_tile_keypoints,
        a_tile_desc, b_tile_desc,
        filtered_match_points_a,
        filtered_match_points_b, 5.0);

      Tile tmp_a_tile = *a_tile;

      // put b at 0,0
      if (filtered_match_points_a.size() >= MIN_FEATURES_NUM) {
        for (int _i = 0; _i < 1000; _i++) {
          float dx = 0.0;
          float dy = 0.0;
          for (int j = 0; j < filtered_match_points_a.size(); j++) {
            cv::Point2f dp =
                b_tile->rigid_transform(filtered_match_points_b[j]) -
                                        tmp_a_tile.rigid_transform(filtered_match_points_a[j]);
            dx += 2*dp.x * 1.0 / (filtered_match_points_a.size());
            dy += 2*dp.y * 1.0 / (filtered_match_points_a.size());
          }
          tmp_a_tile.offset_x += 0.48*dx;
          tmp_a_tile.offset_y += 0.48*dy;
        }
      }

      float val = tmp_a_tile.error_tile_pair(b_tile);
      lock.lock();
      if (val >= CORR_THRESH && filtered_match_points_a.size() >= MIN_FEATURES_NUM) {
        neighbor_success_count++;
        a_tile->insert_matches(b_tile, filtered_match_points_a, filtered_match_points_b);

      cv::Point2f a_point = cv::Point2f(tmp_a_tile.x_start+tmp_a_tile.offset_x,
                                        tmp_a_tile.y_start+tmp_a_tile.offset_y);
      cv::Point2f b_point = cv::Point2f(b_tile->x_start+b_tile->offset_x,
                                        b_tile->y_start+b_tile->offset_y);

      cv::Point2f delta = a_point-b_point;

      a_tile->ideal_offsets[b_tile->tile_id] = delta;
      a_tile->neighbor_correlations[b_tile->tile_id] = val;
      }
      lock.unlock();
    }
  }
  if (neighbor_success_count < neighbors.size()*2.0/4.0 || neighbor_success_count == 0) {
    int res = __sync_fetch_and_add(&num_bad_2d_matches, 1);
    printf("Bad 2D match! %d\n", res);
    a_tile->bad_2d_alignment = true;
    a_tile->edges.clear();
  } else {
    a_tile->bad_2d_alignment = false;
  }
}


void tfk::Section::compute_tile_matches(Tile* a_tile) {
  std::vector<Tile*> _neighbors = get_all_close_tiles(a_tile);

  std::vector<int> neighbors;
  for (int i = 0; i < _neighbors.size(); i++) {
    neighbors.push_back(_neighbors[i]->tile_id);
  }

  int neighbor_success_count = 0;
  std::vector<int> bad_indices(0);
  for (int i = 0; i < neighbors.size(); i++) {
    int btile_id = neighbors[i];
    Tile* b_tile = this->tiles[btile_id];

    if (a_tile->p_kps->size() < MIN_FEATURES_NUM) continue;
    if (b_tile->p_kps->size() < MIN_FEATURES_NUM) continue;
    std::vector< cv::Point2f > filtered_match_points_a(0);
    std::vector< cv::Point2f > filtered_match_points_b(0);

    this->compute_tile_matches_pair(a_tile, b_tile,
        *(a_tile->p_kps), *(b_tile->p_kps),
        *(a_tile->p_kps_desc), *(b_tile->p_kps_desc),
        filtered_match_points_a,
        filtered_match_points_b, 16.0);


    Tile tmp_a_tile = *a_tile;

    // put b at 0,0
    if (filtered_match_points_a.size() >= MIN_FEATURES_NUM) {
      for (int _i = 0; _i < 1000; _i++) {
        float dx = 0.0;
        float dy = 0.0;
        for (int j = 0; j < filtered_match_points_a.size(); j++) {
          cv::Point2f dp =
              b_tile->rigid_transform(filtered_match_points_b[j]) -
                                      tmp_a_tile.rigid_transform(filtered_match_points_a[j]);
          dx += 2*dp.x * 1.0 / (filtered_match_points_a.size());
          dy += 2*dp.y * 1.0 / (filtered_match_points_a.size());
        }
        tmp_a_tile.offset_x += 0.48*dx;
        tmp_a_tile.offset_y += 0.48*dy;
      }
    }

    float val = tmp_a_tile.error_tile_pair(b_tile);

    if (val >= CORR_THRESH && filtered_match_points_a.size() >= MIN_FEATURES_NUM) {
      neighbor_success_count++;
      a_tile->insert_matches(b_tile, filtered_match_points_a, filtered_match_points_b);

      cv::Point2f a_point = cv::Point2f(tmp_a_tile.x_start+tmp_a_tile.offset_x,
                                        tmp_a_tile.y_start+tmp_a_tile.offset_y);
      cv::Point2f b_point = cv::Point2f(b_tile->x_start+b_tile->offset_x,
                                        b_tile->y_start+b_tile->offset_y);

      cv::Point2f delta = a_point-b_point;

      a_tile->ideal_offsets[b_tile->tile_id] = delta;
      a_tile->neighbor_correlations[b_tile->tile_id] = val;
    } else {
      bad_indices.push_back(i);
    }
  }
}



bool tfk::Section::alignment2d_exists() {
  printf("ALIGNMENT2D EXISTS PATH: %s\n", TFK_TMP_DIR.c_str());

  std::ifstream f(std::string(TFK_TMP_DIR) + "/2d_alignment_" +
                      std::to_string(this->real_section_id) + ".pbuf");
  return f.good();
}

void tfk::Section::read_3d_keypoints(std::string filename) {
  printf("right before read 3d keypoints\n");
  Saved3DAlignment alignment3d;
  std::fstream input(std::string(TFK_TMP_DIR) + "/3d_keypoints_" +
                         std::to_string(this->real_section_id)+".pbuf",
                     std::ios::in | std::ios::binary);
  alignment3d.ParseFromIstream(&input);
  printf("right after parsing 3d keypoints\n");
  input.close();

  printf("right before loop for putting data into tiles\n");
  cilk_for (int i = 0; i < this->tiles.size(); i++) {
    Saved3DAlignmentTile tiledata = alignment3d.tiles(i);
    Tile* tile = this->tiles[i];
    tile->p_kps_3d = new std::vector<cv::KeyPoint>();
    tile->p_kps_desc_3d = new cv::Mat();
    for (int j = 0; j < tiledata.keypoint_size(); j++) {
      KeyPointProto kproto = tiledata.keypoint(j);
      std::pair<cv::KeyPoint, cv::Mat> ret = proto_to_keypoint(kproto);
      tile->p_kps_3d->push_back(ret.first);
      tile->p_kps_desc_3d->push_back(ret.second);
    }
  }
  printf("right after loop for putting data into tiles\n");
}


void tfk::Section::compare_2d_alignment() {
  printf("Comparing 2d alignment\n");

  Saved2DAlignmentSection sectiondata;
  std::fstream input(std::string(TFK_TMP_DIR) + "_compare/2d_alignment_" +
                         std::to_string(this->real_section_id)+".pbuf",
                     std::ios::in | std::ios::binary);

  sectiondata.ParseFromIstream(&input);


  for (int i = 0; i < sectiondata.tiles_size(); i++) {
    Tile* tile = this->tiles[i];
    Saved2DAlignmentTile tiledata = sectiondata.tiles(i);
    tile->bad_2d_alignment = tiledata.bad_2d_alignment();
    tile->x_start = tiledata.x_start();
    tile->x_finish = tiledata.x_finish();
    tile->y_start = tiledata.y_start();
    tile->y_finish = tiledata.y_finish();
    tile->offset_x = tiledata.offset_x();
    tile->offset_y = tiledata.offset_y();
  }
  input.close();

  Saved2DAlignmentSection sectiondata2;
  std::fstream input2(std::string(TFK_TMP_DIR) + "/2d_alignment_" +
                          std::to_string(this->real_section_id)+".pbuf",
                      std::ios::in | std::ios::binary);

  sectiondata2.ParseFromIstream(&input2);

  float min_x, min_y;
  float min_x2, min_y2;
  for (int i = 0; i < sectiondata2.tiles_size(); i++) {
    Saved2DAlignmentTile tiledata = sectiondata2.tiles(i);
    Tile* tile = this->tiles[i];
    tile->highlight = false;
    if (i == 0) {
      min_x = tile->x_start + tile->offset_x;
      min_y = tile->y_start + tile->offset_y;
      min_x2 = tiledata.x_start() + tiledata.offset_x();
      min_y2 = tiledata.y_start() + tiledata.offset_y();
    } else {
      float x = tile->x_start + tile->offset_x;
      float y = tile->y_start + tile->offset_y;
      float x2 = tiledata.x_start() + tiledata.offset_x();
      float y2 = tiledata.y_start() + tiledata.offset_y();
      if (x < min_x) min_x = x;
      if (y < min_y) min_y = y;
      if (x2 < min_x2) min_x2 = x2;
      if (y2 < min_y2) min_y2 = y2;
    }
  }

  int count_errors = 0;

  int count_errors_1 = 0;
  int count_errors_2 = 0;
  int count_errors_4 = 0;
  int count_errors_8 = 0;
  int count_errors_16 = 0;

  for (int i = 0; i < sectiondata2.tiles_size(); i++) {
    Tile* tile = this->tiles[i];
    float max_error = 0.0;
    Saved2DAlignmentTile tiledata = sectiondata2.tiles(i);

    cv::Point2f corner = cv::Point2f(tile->x_start + tile->offset_x - min_x,
                                     tile->y_start + tile->offset_y - min_y);

    cv::Point2f corner2 = cv::Point2f(tiledata.x_start() + tiledata.offset_x() - min_x2,
                                      tiledata.y_start() + tiledata.offset_y() - min_y2);
    cv::Point2f delta_corners = corner - corner2;


    std::vector<Tile*> neighbors = this->get_all_close_tiles(tile);
    for (int n = 0; n < neighbors.size(); n++) {
      double dx = neighbors[n]->x_start+neighbors[n]->offset_x - tile->x_start - tile->offset_x;
      double dy = neighbors[n]->y_start+neighbors[n]->offset_y - tile->y_start - tile->offset_y;

      Saved2DAlignmentTile tiledata2 = sectiondata2.tiles(neighbors[n]->tile_id);

      double dx2 = tiledata2.x_start() + tiledata2.offset_x() -
                   tiledata.x_start() - tiledata.offset_x();
      double dy2 = tiledata2.y_start() + tiledata2.offset_y() -
                   tiledata.y_start() - tiledata.offset_y();

      float diff = std::sqrt((dx-dx2)*(dx-dx2) + (dy-dy2)*(dy-dy2));
      //bool res1 = tiledata2.bad_2d_alignment() || tiledata.bad_2d_alignment();
      bool res2 = tile->bad_2d_alignment || neighbors[n]->bad_2d_alignment;
      if (diff > 1.0 && !res2) {
        max_error = std::max(diff, max_error);
      }
    }

    float diff = max_error;
      if (diff > 1.0) {
        max_error = std::max(diff, max_error);
        count_errors_1++;
        if (diff > 2.0) count_errors_2++;
        if (diff > 4.0) count_errors_4++;
        if (diff > 8.0) count_errors_8++;
        if (diff > 16.0) count_errors_16++;
        count_errors++;
        if (diff > 4.0) {
          tile->highlight = true;
          printf("Tile id is %d\n", tile->tile_id);
          printf("Diff is %f\n", diff);
        }
        }

    float delta = std::sqrt(delta_corners.x*delta_corners.x + delta_corners.y*delta_corners.y);
    if (delta > 15.0 || true) {
    }
  }
  printf("total errors is %d section %d\n", count_errors, this->real_section_id);
  input2.close();
  __sync_fetch_and_add(&ALL_2D_ERRORS_1, count_errors_1);
  __sync_fetch_and_add(&ALL_2D_ERRORS_2, count_errors_2);
  __sync_fetch_and_add(&ALL_2D_ERRORS_4, count_errors_4);
  __sync_fetch_and_add(&ALL_2D_ERRORS_8, count_errors_8);
  __sync_fetch_and_add(&ALL_2D_ERRORS_16, count_errors_16);
  printf("Running error total %ld %ld %ld %ld %ld\n", ALL_2D_ERRORS_1, ALL_2D_ERRORS_2,
                                                 ALL_2D_ERRORS_4, ALL_2D_ERRORS_8,
                                                 ALL_2D_ERRORS_16);
}

void tfk::Section::load_2d_alignment() {
  printf("Loading 2d alignment\n");

  Saved2DAlignmentSection sectiondata;
  std::fstream input(std::string(TFK_TMP_DIR) + "/2d_alignment_" +
                         std::to_string(this->real_section_id)+".pbuf",
                     std::ios::in | std::ios::binary);

  sectiondata.ParseFromIstream(&input);

  for (int i = 0; i < sectiondata.tiles_size(); i++) {
    Tile* tile = this->tiles[i];
    Saved2DAlignmentTile tiledata = sectiondata.tiles(i);
    tile->bad_2d_alignment = tiledata.bad_2d_alignment();
    tile->x_start = tiledata.x_start();
    tile->x_finish = tiledata.x_finish();
    tile->y_start = tiledata.y_start();
    tile->y_finish = tiledata.y_finish();
    tile->offset_x = tiledata.offset_x();
    tile->offset_y = tiledata.offset_y();
  }
  input.close();
}


void tfk::Section::save_2d_alignment() {
  printf("saving 2d alignment\n");
  Saved2DAlignmentSection sectiondata;
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    Saved2DAlignmentTile tiledata;
    tiledata.set_tile_id(i);
    tiledata.set_bad_2d_alignment(tile->bad_2d_alignment);
    tiledata.set_x_start(1.0*tile->x_start);
    tiledata.set_x_finish(1.0*tile->x_finish);
    tiledata.set_y_start(1.0*tile->y_start);
    tiledata.set_y_finish(1.0*tile->y_finish);
    tiledata.set_offset_x(1.0*tile->offset_x);
    tiledata.set_offset_y(1.0*tile->offset_y);
    sectiondata.add_tiles();
    *(sectiondata.mutable_tiles(i)) = tiledata;
  }
  std::fstream output(std::string(TFK_TMP_DIR) +
                          "/2d_alignment_"+std::to_string(this->real_section_id)+".pbuf",
                      std::ios::out | std::ios::trunc | std::ios::binary);
  sectiondata.SerializeToOstream(&output);
  output.close();
}



KeyPointProto tfk::Section::keypoint_to_proto(cv::KeyPoint pt, cv::Mat desc) {
  KeyPointProto kpt;
  kpt.set_x(pt.pt.x);
  kpt.set_y(pt.pt.y);
  kpt.set_size(pt.size);
  kpt.set_angle(pt.angle);
  kpt.set_response(pt.response);
  kpt.set_octave(pt.octave);
  kpt.set_class_id(pt.class_id);

  KeyPointDesc kpdesc;
  kpdesc.set_rows(desc.rows);
  kpdesc.set_cols(desc.cols);
  for (int r = 0; r < desc.rows; r++) {
    for (int c = 0; c < desc.cols; c++) {
      kpdesc.add_data(desc.at<float>(r, c));
    }
  }
  *(kpt.mutable_desc()) = kpdesc;
  return kpt;
}

std::pair<cv::KeyPoint, cv::Mat> tfk::Section::proto_to_keypoint(KeyPointProto kptproto) {
  cv::KeyPoint kpt;
  kpt.pt = cv::Point2f(kptproto.x(), kptproto.y());
  kpt.size = kptproto.size();
  kpt.angle = kptproto.angle();
  kpt.response = kptproto.response();
  kpt.octave = kptproto.octave();
  kpt.class_id = kptproto.class_id();

  KeyPointDesc kpdesc = kptproto.desc();
  cv::Mat desc = cv::Mat(kpdesc.rows(), kpdesc.cols(), CV_32F);
  int j = 0;
  for (int r = 0; r < desc.rows; r++) {
    for (int c = 0; c < desc.cols; c++) {
      desc.at<float>(r, c) = 1.0*kpdesc.data(j);
      j++;
    }
  }
  return std::make_pair(kpt, desc);
}


void tfk::Section::save_3d_keypoints(std::string filename) {
  Saved3DAlignment alignment3d;
  alignment3d.set_section_id(this->real_section_id);
  for (int i = 0; i < this->tiles.size(); i++) {
    Saved3DAlignmentTile tiledata;
    tiledata.set_tile_id(i);
    Tile* t = this->tiles[i];
    for (int j = 0; j < t->p_kps_3d->size(); j++) {
      tiledata.add_keypoint();
      *(tiledata.mutable_keypoint(j)) =
          keypoint_to_proto((*(t->p_kps_3d))[j], (*(t->p_kps_desc_3d)).row(j));
    }
    alignment3d.add_tiles();
    *(alignment3d.mutable_tiles(i)) = tiledata;
  }

  std::fstream output(std::string(TFK_TMP_DIR) + "/3d_keypoints_" +
                          std::to_string(this->real_section_id)+".pbuf",
                      std::ios::out | std::ios::trunc | std::ios::binary);
  alignment3d.SerializeToOstream(&output);
  output.close();
}

void tfk::Section::save_2d_graph(std::string filename) {
  cv::FileStorage fs(filename+std::string("_2d_matches.yml.gz"), cv::FileStorage::WRITE);
  int count = 0;
  for (int i = 0; i < this->tiles.size(); i++) {
      Tile* tile = this->tiles[i];
      cv::write(fs, "num_edges_"+std::to_string(i), (int) tile->edges.size());
      cv::write(fs, "bad_2d_alignment_"+std::to_string(i), (bool) this->tiles[i]->bad_2d_alignment);
    for (int j = 0; j < tile->edges.size(); j++) {
      cv::write(fs, "neighbor_id_"+std::to_string(i) + "_" + std::to_string(j),
          tile->edges[j].neighbor_id);
      cv::write(fs, "weight_"+std::to_string(i) + "_" + std::to_string(j),
          1.0);
      cv::write(fs, "v_points_"+std::to_string(i)+"_"+std::to_string(j),
          *(tile->edges[j].v_points));
      cv::write(fs, "n_points_"+std::to_string(i)+"_"+std::to_string(j),
          *(tile->edges[j].n_points));
      count++;
    }
  }
  printf("wrote %d edges\n", count);
  fs.release();
}
void tfk::Section::read_2d_graph(std::string filename) {
  cv::FileStorage fs(filename+std::string("_2d_matches.yml.gz"), cv::FileStorage::READ);
  int count = 0;
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    int edge_size;
    fs["num_edges_"+std::to_string(i)] >> edge_size;
    fs["bad_2d_alignment_"+std::to_string(i)] >> this->tiles[i]->bad_2d_alignment;
    std::vector<edata> edge_data;
    std::set<int> n_ids_seen;
    tile->edges.clear();
    n_ids_seen.clear();
    for (int j = 0; j < edge_size; j++) {
      edata edge;
      fs["neighbor_id_"+std::to_string(i) + "_" + std::to_string(j)] >> edge.neighbor_id;
      std::vector<cv::Point2f>* v_points = new std::vector<cv::Point2f>();
      std::vector<cv::Point2f>* n_points = new std::vector<cv::Point2f>();
      fs["weight_"+std::to_string(i) + "_" + std::to_string(j)] >> edge.weight;
      fs["v_points_"+std::to_string(i) + "_" + std::to_string(j)] >> *v_points;
      fs["n_points_"+std::to_string(i) + "_" + std::to_string(j)] >> *n_points;
      edge.v_points = v_points;
      edge.n_points = n_points;
      edge.neighbor_tile = this->tiles[edge.neighbor_id];
      tile->edges.push_back(edge);
      count++;
    }
  }

  printf("read %d edges\n", count);
  fs.release();
}



void tfk::Section::read_tile_matches() {
  std::string filename =
      std::string(std::string(TFK_TMP_DIR) + "/prefix_"+std::to_string(this->real_section_id));

  this->read_3d_keypoints(filename);
  this->read_2d_graph(filename);
}

void tfk::Section::save_tile_matches() {
  std::string filename =
      std::string(std::string(TFK_TMP_DIR)+"/prefix_"+std::to_string(this->real_section_id));

  this->save_3d_keypoints(filename);
  this->save_2d_graph(filename);
}

void tfk::Section::recompute_keypoints() {
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* tile = this->tiles[i];
    if (tile->image_data_replaced) {
      tile->image_data_replaced = false;
    }
  }
}

void tfk::Section::compute_keypoints_and_matches() {
  // assume that section data doesn't exist.
  printf("this is compute_keypoints_and_matches and it's being called!\n");
  int64_t tiles_in_error = 0;
  if (!this->section_data_exists()) {
    std::vector<std::pair<float, Tile*> > sorted_tiles;
    for (int i = 0; i < this->tiles.size(); i++) {
      Tile* t = this->tiles[i];
      sorted_tiles.push_back(std::make_pair(t->x_start, t));
    }
    std::sort(sorted_tiles.begin(), sorted_tiles.end());
    // sorted tiles is now sorted in increasing order based on x_start;

    std::set<Tile*> active_set;
    std::set<Tile*> neighbor_set;

    std::set<Tile*> opened_set;
    std::set<Tile*> closed_set;

    Tile* pivot = sorted_tiles[0].second;

    bool pivot_good = false;
    int pivot_search_start = 0;
    for (int i = pivot_search_start; i < sorted_tiles.size(); i++) {
      if (sorted_tiles[i].second->x_start > pivot->x_finish + 12000) {
        pivot = sorted_tiles[i].second;
        pivot_search_start = i;
        pivot_good = true;
        break;
      } else {
        active_set.insert(sorted_tiles[i].second);
      }
    }
    if (!pivot_good) {
      pivot_search_start = sorted_tiles.size();
    }
    printf("Num tiles in sweep 0 is %lu\n", active_set.size());

    std::map<int, TileSiftTask*> dependencies;
    while (active_set.size() > 0) {
      printf("Current active set size is %lu percent done %f%%\n", active_set.size(), (pivot_search_start*100.0)/sorted_tiles.size());



      printf(" begin find all the neighbors \n");
      TFK_TIMER_VAR(active_set_timer); 
      TFK_START_TIMER(&active_set_timer); 
      TFK_STOP_TIMER(&active_set_timer, "begin"); 
      // find all the neighbors.
      for (auto it = active_set.begin(); it != active_set.end(); ++it) {
        Tile* tile = *it;
        std::vector<Tile*> overlapping = this->get_all_close_tiles(tile);
        for (int j = 0; j < overlapping.size(); j++) {
          neighbor_set.insert(overlapping[j]);
        }
      }
      TFK_STOP_TIMER(&active_set_timer, "find all neighors");
      // close open tiles that aren't in active or neighbor set.
      for (auto it = opened_set.begin(); it != opened_set.end(); ++it) {
        Tile* tile = *it;
        if (active_set.find(tile) == active_set.end() &&
            neighbor_set.find(tile) == neighbor_set.end()) {
          closed_set.insert(tile);
          tile->release_2d_keypoints();
          delete dependencies[tile->tile_id];
          dependencies.erase(tile->tile_id);
          tile->release_full_image();
        }
      }
      TFK_STOP_TIMER(&active_set_timer, "close open tiles");

      std::vector<Tile*> tiles_to_process_keypoints, tiles_to_process_matches;

      for (auto it = active_set.begin(); it != active_set.end(); ++it) {
        Tile* tile = *it;
        tiles_to_process_matches.push_back(tile);
        if (opened_set.find(tile) == opened_set.end()) {
          opened_set.insert(tile);
          tiles_to_process_keypoints.push_back(tile);
        }
      }

      TFK_STOP_TIMER(&active_set_timer, "open active set tiles1");
      for (auto it = neighbor_set.begin(); it != neighbor_set.end(); ++it) {
        Tile* tile = *it;
        if (opened_set.find(tile) == opened_set.end()) {
          opened_set.insert(tile);
          tiles_to_process_keypoints.push_back(tile);
        }
      }
      TFK_STOP_TIMER(&active_set_timer, "open active set tiles2");

      for (int i = 0; i < tiles_to_process_keypoints.size(); i++) {
        Tile* tile = tiles_to_process_keypoints[i];
        TileSiftTask* sift_task = new TileSiftTask(this->paramdbs[MATCH_TILE_PAIR_TASK_ID], tile);
        dependencies[tile->tile_id] = sift_task;
      }
      TFK_STOP_TIMER(&active_set_timer, "open active set tiles3");

      #pragma cilk grainsize 1
      cilk_for (int i = 0; i < tiles_to_process_keypoints.size(); i++) {
         Tile* tile = tiles_to_process_keypoints[i];
         dependencies[tile->tile_id]->compute(0.9);
         tile->compute_sift_keypoints3d();
         tile->match_tiles_task->dependencies = dependencies;
      }
      TFK_STOP_TIMER(&active_set_timer, "keypoints 3D");

      std::vector<std::pair<float, Tile*> > sorted_y_tiles;
      for (int i = 0; i < tiles_to_process_matches.size(); i++) {
        sorted_y_tiles.push_back(std::make_pair(tiles_to_process_matches[i]->y_start,
                                                tiles_to_process_matches[i]));
      }
      std::sort(sorted_y_tiles.begin(), sorted_y_tiles.end());

      TFK_STOP_TIMER(&active_set_timer, "sort tiles by y");

      #pragma cilk grainsize 1
      cilk_for (int i = 0; i < tiles_to_process_matches.size(); i++) {
        Tile* t = sorted_y_tiles[i].second;
        t->match_tiles_task->dependencies = dependencies;
        t->match_tiles_task->compute(0.9);
        //tiles_to_process_matches[i]->match_tiles_task->dependencies = dependencies;
        //tiles_to_process_matches[i]->match_tiles_task->compute(0.9);
      }
      TFK_STOP_TIMER(&active_set_timer, "compute stage 1");

      #pragma cilk grainsize 1
      cilk_for (int i = 0; i < tiles_to_process_matches.size(); i++) {
        Tile* t = sorted_y_tiles[i].second;
        if (!t->match_tiles_task->error_check(0.4)) {
          std::map<int, TileSiftTask*> empty_map;
          t->match_tiles_task->dependencies = empty_map;
          t->match_tiles_task->compute(1.0);
        }
        //if (!tiles_to_process_matches[i]->match_tiles_task->error_check(0.4)) {
        //  std::map<int, TileSiftTask*> empty_map;
        //  tiles_to_process_matches[i]->match_tiles_task->dependencies = empty_map;
        //  tiles_to_process_matches[i]->match_tiles_task->compute(1.0);
        //}
      }
      TFK_STOP_TIMER(&active_set_timer, "compute stage 2");

      #pragma cilk grainsize 1
      cilk_for (int i = 0; i < tiles_to_process_matches.size(); i++) {
        Tile* t = sorted_y_tiles[i].second;
        if (!t->match_tiles_task->error_check(1.9)) {
          t->match_tiles_task->commit();
        } else {
          t->match_tiles_task->commit();
        }

        //if (!tiles_to_process_matches[i]->match_tiles_task->error_check(1.9)) {
        //  tiles_to_process_matches[i]->match_tiles_task->commit();
        //} else {
        //  tiles_to_process_matches[i]->match_tiles_task->commit();
        //}
      }
      TFK_STOP_TIMER(&active_set_timer, "commit");
      opened_set.clear();
      for (auto it = active_set.begin(); it != active_set.end(); ++it) {
        opened_set.insert(*it);
      }

      for (auto it = neighbor_set.begin(); it != neighbor_set.end(); ++it) {
        opened_set.insert(*it);
      }

      // clear the active and neighbor set.
      active_set.clear();
      neighbor_set.clear();

      //float duration = tdiff(global_start, gettime());
      pivot_good = false;
      for (int i = pivot_search_start; i < sorted_tiles.size(); i++) {
        if (sorted_tiles[i].second->x_start > pivot->x_finish + 12000) {
          pivot = sorted_tiles[i].second;
          pivot_search_start = i;
          pivot_good = true;
          TFK_STOP_TIMER(&active_set_timer, "cleanup, break");
          break;
        } else {
          active_set.insert(sorted_tiles[i].second);
        }
      }

      if (!pivot_good) {
        pivot_search_start = sorted_tiles.size();
      }
          TFK_STOP_TIMER(&active_set_timer, "cleanup, no break");
    }

      // close open tiles that aren't in active or neighbor set.
      for (auto it = opened_set.begin(); it != opened_set.end(); ++it) {
        Tile* tile = *it;
        if (active_set.find(tile) == active_set.end() &&
            neighbor_set.find(tile) == neighbor_set.end()) {
          closed_set.insert(tile);
          tile->release_2d_keypoints();

          delete dependencies[tile->tile_id];
          dependencies.erase(tile->tile_id);
          tile->release_full_image();
        }
      }
    for (int i = 0; i < this->tiles.size(); i++) {
      this->tiles[i]->release_full_image();
      this->tiles[i]->release_2d_keypoints();
    }
    this->save_tile_matches();
  } else {
    this->read_tile_matches();
  }
  printf("Total tiles falling onto second pass %ld\n", tiles_in_error);
    this->graph = new Graph();
    graph->resize(this->tiles.size());
  // phase 0 of make_symmetric --- find edges to add.
  cilk_for (int i = 0; i < this->tiles.size(); i++) {
    this->tiles[i]->make_symmetric(0, this->tiles);
  }

  // phase 1 of make_symmetric --- insert found edges.
  cilk_for (int i = 0; i < this->tiles.size(); i++) {
    this->tiles[i]->make_symmetric(1, this->tiles);
  }


  for (int i = 0; i < graph->num_vertices(); i++) {
    vdata* d = graph->getVertexData(i);


    for (int j = 0; j < this->tiles[i]->edges.size(); j++) {
      graph->edgeData[i].push_back(this->tiles[i]->edges[j]);
    }

    Tile* tile = this->tiles[i];
    d->tile = tile;
    d->vertex_id = i;
    d->mfov_id = tile->mfov_id;
    d->tile_index = tile->index;
    d->tile_id = i;
    d->start_x = tile->x_start;
    d->end_x = tile->x_finish;
    d->start_y = tile->y_start;
    d->end_y = tile->y_finish;
    d->offset_x = 0.0;
    d->offset_y = 0.0;
    d->iteration_count = 0;
    d->z = /*p_align_data->base_section + */this->section_id;
    d->a00 = 1.0;
    d->a01 = 0.0;
    d->a10 = 0.0;
    d->a11 = 1.0;
    d->original_center_point =
      cv::Point2f((tile->x_finish-tile->x_start)/2,
                  (tile->y_finish-tile->y_start)/2);
  }

  printf("Num vertices is %d\n", graph->num_vertices());
  graph->section_id = this->section_id;
}

std::vector<tfk::Tile*> tfk::Section::get_all_close_tiles_with_min_overlap(Tile* a_tile, int min_dim_overlap) {
  std::vector<Tile*> neighbor_tiles(0);
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* b_tile = this->tiles[i];
    if (a_tile == b_tile) continue;
    if (a_tile->overlaps_with_threshold(b_tile, min_dim_overlap)) {
      neighbor_tiles.push_back(b_tile);
    }
  }
  return neighbor_tiles;
}


std::vector<tfk::Tile*> tfk::Section::get_all_close_tiles(Tile* a_tile) {
  std::vector<Tile*> neighbor_tiles(0);
  for (int i = 0; i < this->tiles.size(); i++) {
    Tile* b_tile = this->tiles[i];
    if (a_tile == b_tile) continue;
    if (a_tile->overlaps_with(b_tile)) {
      neighbor_tiles.push_back(b_tile);
    }
  }
  return neighbor_tiles;
}


std::vector<int> tfk::Section::get_all_close_tiles(int atile_id) {
  std::vector<int> neighbor_index_list(0);

  Tile* a_tile = this->tiles[atile_id];
  for (int i = 0/*atile_id+1*/; i < this->tiles.size(); i++) {
    Tile* b_tile = this->tiles[i];
    if (a_tile->overlaps_with(b_tile)) {
      neighbor_index_list.push_back(i);
    }
  }

  return neighbor_index_list;
}

// Section from protobuf
tfk::Section::Section(SectionData& section_data, std::pair<cv::Point2f, cv::Point2f> bounding_box,
                      bool use_bbox_prefilter) {
  //this->off_grid = new std::vector<cv::Point2f>();
  this->elastic_transform_ready = false;
  this->section_id = section_data.section_id();
  this->real_section_id = section_data.section_id();
  this->num_tiles_replaced = 0;
  this->n_tiles = section_data.tiles_size();
  this->a00 = 1.0;
  this->a11 = 1.0;
  this->a01 = 0.0;
  this->a10 = 0.0;
  this->offset_x = 0.0;
  this->offset_y = 0.0;
  this->num_bad_2d_matches = 0;
  if (section_data.has_out_d1()) {
    this->out_d1 = section_data.out_d1();
  }
  if (section_data.has_out_d2()) {
    this->out_d2 = section_data.out_d2();
  }

  this->section_mesh_matches_mutex = new std::mutex();

  std::vector<Tile*> tmp_tiles;
  tmp_tiles.resize(section_data.tiles_size());

  float min_x, min_y;
  // need to find min_x and min_y and make it zero.
  for (int j = 0; j < section_data.tiles_size(); j++) {
    TileData tile_data = section_data.tiles(j);
    Tile* tile = new Tile(tile_data);
    if (j == 0) {
      min_x = tile->x_start;
      min_y = tile->y_start;
    } else {
      if (tile->x_start < min_x) min_x = tile->x_start;
      if (tile->y_start < min_y) min_y = tile->y_start;
    }
  }


  std::mt19937 g1(234);  // mt19937 is a standard mersenne_twister_engine
  std::uniform_int_distribution<int> distribution(0, 100000000);


  int added_count = 0;
  for (int j = 0; j < section_data.tiles_size(); j++) {
    TileData tile_data = section_data.tiles(j);

    Tile* tile = new Tile(tile_data);
    tile->tile_id = j;
    tile->random_int = distribution(g1);

    double x_diff = tile->x_finish - tile->x_start;
    double y_diff = tile->y_finish - tile->y_start;

    tile->x_start -= min_x;
    tile->y_start -= min_y;

    tile->x_start = std::round(tile->x_start);
    tile->y_start = std::round(tile->y_start);

    tile->x_finish = tile->x_start + x_diff;
    tile->y_finish = tile->y_start + y_diff;


    if (tile->offset_x != 0.0 || tile->offset_y != 0.0) {
      printf("error the tlie doesn't have offsets 0 %f %f\n", tile->offset_x, tile->offset_y); 
    }

    std::string new_filepath = "new_tiles/sec_"+std::to_string(this->real_section_id) +
        "_tileid_"+std::to_string(tile->tile_id) + ".bmp";

    std::string test_filepath = "new_tiles/thumbnail_sec_"+std::to_string(this->real_section_id) +
        "_tileid_"+std::to_string(tile->tile_id) + ".jpg";
    if (!use_bbox_prefilter || tile->overlaps_with(bounding_box)) {
      this->tiles.push_back(tile);
      tile->tile_id = added_count++;
    } else {
      //auto bbox = tile->get_bbox();
    }
  }
  // passing down the pointer to ml_models
}





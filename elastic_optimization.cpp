// Copyright 2016 - Supertech Research Group

#include "./elastic_optimization.h"
#include <cilk/reducer_opadd.h>
using namespace tfk;

void elastic_gradient_descent_section(Section* _this, Section* _neighbor) {
  double cross_slice_weight = 1.0;
  double cross_slice_winsor = 20.0;
  double intra_slice_weight = 1.0;
  double intra_slice_winsor = 200.0;
  int max_iterations = 10000;
  double stepsize = 0.1;
  double momentum = 0.9;

  if (_neighbor == NULL || _this->real_section_id == _neighbor->real_section_id) {
    max_iterations = 1;
  }

  std::map<int, double> gradient_momentum;

  Section neighbor = *_neighbor;

  // init my section.

  {
    Section* section = _this;
    section->gradients = new cv::Point2f[section->triangle_mesh->mesh_orig->size()];
    section->gradients_with_momentum = new cv::Point2f[section->triangle_mesh->mesh_orig->size()];
    section->rest_lengths = new double[section->triangle_mesh->triangle_edges->size()];
    section->rest_areas = new double[section->triangle_mesh->triangles->size()];
    section->mesh_old = new std::vector<cv::Point2f>();

    for (int j = 0; j < section->triangle_mesh->mesh_orig->size(); j++) {
      section->mesh_old->push_back((*(section->triangle_mesh->mesh))[j]);
    }

    // init
    for (int j = 0; j < section->triangle_mesh->mesh_orig->size(); j++) {
      section->gradients[j] = cv::Point2f(0.0, 0.0);
      section->gradients_with_momentum[j] = cv::Point2f(0.0, 0.0);
    }
    
    for (int j = 0; j < section->triangle_mesh->triangle_edges->size(); j++) {
      //cv::Point2f p1 = (*(section->triangle_mesh->mesh))[
      //                 (*(section->triangle_mesh->triangle_edges))[j].first];
      //cv::Point2f p2 = (*(section->triangle_mesh->mesh))[
      //                 (*(section->triangle_mesh->triangle_edges))[j].second];
      cv::Point2f p1 = (*(section->triangle_mesh->mesh_orig))[
                       (*(section->triangle_mesh->triangle_edges))[j].first];
      cv::Point2f p2 = (*(section->triangle_mesh->mesh_orig))[
                       (*(section->triangle_mesh->triangle_edges))[j].second];
      double dx = p1.x-p2.x;
      double dy = p1.y-p2.y;
      double len = std::sqrt(dx*dx+dy*dy);
      section->rest_lengths[j] = len;
    }
    for (int j = 0; j < section->triangle_mesh->triangles->size(); j++) {
      tfkTriangle tri = (*(section->triangle_mesh->triangles))[j];
      //cv::Point2f p1 = (*(section->triangle_mesh->mesh))[tri.index1];
      //cv::Point2f p2 = (*(section->triangle_mesh->mesh))[tri.index2];
      //cv::Point2f p3 = (*(section->triangle_mesh->mesh))[tri.index3];
      cv::Point2f p1 = (*(section->triangle_mesh->mesh_orig))[tri.index1];
      cv::Point2f p2 = (*(section->triangle_mesh->mesh_orig))[tri.index2];
      cv::Point2f p3 = (*(section->triangle_mesh->mesh_orig))[tri.index3];
      section->rest_areas[j] = computeTriangleArea(p1, p2, p3);
    }
  }

  {
    Section* section = &neighbor;
    section->gradients = new cv::Point2f[section->triangle_mesh->mesh_orig->size()];
    section->gradients_with_momentum = new cv::Point2f[section->triangle_mesh->mesh_orig->size()];
    section->rest_lengths = new double[section->triangle_mesh->triangle_edges->size()];
    section->rest_areas = new double[section->triangle_mesh->triangles->size()];
    section->mesh_old = new std::vector<cv::Point2f>();

    for (int j = 0; j < section->triangle_mesh->mesh_orig->size(); j++) {
      section->mesh_old->push_back((*(section->triangle_mesh->mesh_orig))[j]);
    }

    // init
    for (int j = 0; j < section->triangle_mesh->mesh_orig->size(); j++) {
      section->gradients[j] = cv::Point2f(0.0, 0.0);
      section->gradients_with_momentum[j] = cv::Point2f(0.0, 0.0);
    }
    for (int j = 0; j < section->triangle_mesh->triangle_edges->size(); j++) {
      cv::Point2f p1 = (*(section->triangle_mesh->mesh_orig))[
                       (*(section->triangle_mesh->triangle_edges))[j].first];
      cv::Point2f p2 = (*(section->triangle_mesh->mesh_orig))[
                       (*(section->triangle_mesh->triangle_edges))[j].second];
      double dx = p1.x-p2.x;
      double dy = p1.y-p2.y;
      double len = std::sqrt(dx*dx+dy*dy);
      section->rest_lengths[j] = len;
    }
    for (int j = 0; j < section->triangle_mesh->triangles->size(); j++) {
      tfkTriangle tri = (*(section->triangle_mesh->triangles))[j];
      cv::Point2f p1 = (*(section->triangle_mesh->mesh_orig))[tri.index1];
      cv::Point2f p2 = (*(section->triangle_mesh->mesh_orig))[tri.index2];
      cv::Point2f p3 = (*(section->triangle_mesh->mesh_orig))[tri.index3];
      section->rest_areas[j] = computeTriangleArea(p1, p2, p3);
    }
  }

    double prev_cost = 0.0;
    for (int iter = 0; iter < max_iterations; iter++) {
      double cost = 0.0;
      {
        Section* section = _this;
        cilk_for (int j = 0; j < section->triangle_mesh->mesh->size(); j++) {
          ((section->gradients))[j] = cv::Point2f(0.0, 0.0);
        }
      }

        cilk::reducer_opadd<double> cost_reducer(0.0);
      {
        Section* section = _this;

        // internal_mesh_derivs
        double all_weight = intra_slice_weight;
        double sigma = intra_slice_winsor;
        std::vector<cv::Point2f>* mesh = section->triangle_mesh->mesh;
        cv::Point2f* gradients = section->gradients;

        std::vector<std::pair<int, int> >* triangle_edges = section->triangle_mesh->triangle_edges;
        std::vector<tfkTriangle >* triangles = section->triangle_mesh->triangles;
        double* rest_lengths = section->rest_lengths;
        double* rest_areas = section->rest_areas;


        // update all edges
        cilk_for (int j = 0; j < triangle_edges->size(); j++) {
          *cost_reducer += internal_mesh_derivs(mesh, gradients, (*triangle_edges)[j], rest_lengths[j],
                                       all_weight/(triangle_edges->size()), sigma);
        }

        // update all triangles
        cilk_for (int j = 0; j < triangles->size(); j++) {
          int triangle_indices[3] = {(*triangles)[j].index1,
                                     (*triangles)[j].index2,
                                     (*triangles)[j].index3};
          *cost_reducer += area_mesh_derivs(mesh, gradients, triangle_indices, rest_areas[j],
                                   all_weight/(triangles->size()));
        }
      }


      {
        Section* section = _this;
        std::vector<tfkMatch>& mesh_matches = section->section_mesh_matches;
        //printf("num mesh matches %zu\n", mesh_matches.size());
        cilk_for (int j = 0; j < mesh_matches.size(); j++) {
          Section* _my_section = (Section*) mesh_matches[j].my_section;
          Section* _n_section = (Section*) mesh_matches[j].n_section;

          if (_my_section->section_id != _this->section_id ||
              _n_section->section_id != _neighbor->section_id) continue;
          Section* my_section = _this;
          //Section* n_section = &neighbor;
          std::vector<cv::Point2f>* mesh1 = my_section->triangle_mesh->mesh;
          //std::vector<cv::Point2f>* mesh2 = n_section->triangle_mesh->mesh_orig;

          cv::Point2f* gradients1 = my_section->gradients;
          double* barys1 = mesh_matches[j].my_barys;
          double all_weight = cross_slice_weight / sqrt(mesh_matches.size()); // NOTE(TFK) may want sqrt(
          double sigma = cross_slice_winsor;

          int indices1[3] = {mesh_matches[j].my_tri.index1,
                             mesh_matches[j].my_tri.index2,
                             mesh_matches[j].my_tri.index3};

          *cost_reducer += crosslink_mesh_derivs(mesh1,
                                        gradients1,
                                        indices1,
                                        barys1,
                                        all_weight, sigma, mesh_matches[j].dest_p);
        }
      }
      cost = cost_reducer.get_value();
      if (iter == 0) prev_cost = cost+10.0;

      if (cost <= prev_cost) {
        stepsize *= 1.01;
        if (stepsize > 10.0) {
          stepsize = 10.0;
        }

        {
          Section* section = _this;
          std::vector<cv::Point2f>* mesh = section->triangle_mesh->mesh;
          std::vector<cv::Point2f>* mesh_old = section->mesh_old;
          cv::Point2f* gradients = section->gradients;
          cv::Point2f* gradients_with_momentum = section->gradients_with_momentum;
          cilk_for (int j = 0; j < mesh->size(); j++) {
            gradients_with_momentum[j] = gradients[j] + momentum*gradients_with_momentum[j];
          }

          cilk_for (int j = 0; j < mesh->size(); j++) {
            (*mesh_old)[j] = ((*mesh)[j]);
          }
          cilk_for (int j = 0; j < mesh->size(); j++) {
            (*mesh)[j].x -= (float)(stepsize * (gradients_with_momentum)[j].x);
            (*mesh)[j].y -= (float)(stepsize * (gradients_with_momentum)[j].y);
          }
        }

          if (max_iterations - iter < 1000) {
            if (prev_cost - cost > 0.1/sqrt(_this->triangle_mesh->triangles->size()) && max_iterations < 100000) {
              max_iterations += 1000;
            }
          }

        if (iter%100 == 0) {
          printf("Good step old cost %f, new cost %f, iteration %d, max %d\n",
                 prev_cost, cost, iter, max_iterations);
        }
        prev_cost = cost;
      } else {
        stepsize *= 0.95;
        {
          Section* section = _this;
          std::vector<cv::Point2f>* mesh = section->triangle_mesh->mesh;
          std::vector<cv::Point2f>* mesh_old = section->mesh_old;
          cv::Point2f* gradients_with_momentum = section->gradients_with_momentum;
          cilk_for (int j = 0; j < mesh->size(); j++) {
            gradients_with_momentum[j] = cv::Point2f(0.0, 0.0);
          }

          cilk_for (int j = 0; j < mesh->size(); j++) {
            (*mesh)[j] = (*mesh_old)[j];
          }
        }
        if (iter%1000 == 0) {
          printf("Bad step old cost %f, new cost %f, iteration %d\n", prev_cost, cost, iter);
        }
      }
    }

    // unclear if I want to do this.
    //_this->save_elastic_mesh(_neighbor);
}




#include "stack.hpp"
#include "stack_helpers.cpp"
#include "./othersift2.hpp"
#include "./stack_tile.cpp"
#include "./stack_section.cpp"


//#include "cilk_tools/engine.h"

tfk::Stack::Stack(int base_section, int n_sections,
    std::string input_filepath, std::string output_dirpath) {
  this->base_section = base_section;
  this->n_sections = n_sections;
  this->input_filepath = input_filepath;
  this->output_dirpath = output_dirpath;
}

void tfk::Stack::render_error(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix) {

  /*cilk_*/for (int i = 1; i < this->sections.size()-2; i++) {
    std::cout << "starting section "  << i << std::endl;
    Section* section = this->sections[i];
    std::pair<std::vector<std::pair<cv::Point2f, cv::Point2f>>, std::vector<std::pair<cv::Point2f, cv::Point2f> > > res = section->render_error(this->sections[i-1], this->sections[i+1], this->sections[i+2], bbox, filename_prefix+std::to_string(i)+".png");

  }


}

void tfk::Stack::render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix,
    Resolution res) {
  for (int i = 0; i < this->sections.size(); i++) {
    Section* section = this->sections[i];
    section->render(bbox, filename_prefix+std::to_string(i)+".tif", res);
  }
}

void tfk::Stack::recompute_alignment() {
  for (int i = 0; i < this->sections.size(); i++) {
    this->sections[i]->recompute_keypoints();
  }
  this->elastic_align();
}

void tfk::Stack::coarse_affine_align() {
  // get affine transform matrix for each section
  //   matrix A_i aligning section i to i-1.
  cilk_for (int i = 1; i < this->sections.size(); i++) {
    this->sections[i]->coarse_affine_align(this->sections[i-1]);
  }

  // cascade the affine transforms down.
  for (int i = 1; i < this->sections.size(); i++) {
    for (int j = 0; j < i; j++) {
      this->sections[j]->affine_transforms.push_back(this->sections[i]->coarse_transform);
    }
    //for (int k = 0; k < this->sections[i-1]->affine_transforms.size(); k++) {
    //  this->sections[i]->affine_transforms.push_back(this->sections[i-1]->affine_transforms[k]);
    //}
    //for (int j = i+1; j < this->sections.size(); j++) {

    //  for (int k = 0; k < this->sections[j]->affine_transforms.size(); k++) {
    //      this->sections[i]->affine_transforms.push_back(this->sections[j]->affine_transforms[k]);
    //    }
    //}
  }

  for (int i = 0; i < this->sections.size(); i++) {
    this->sections[i]->apply_affine_transforms();
  }

}

void tfk::Stack::get_elastic_matches() {
  cilk_for (int section = 1; section < this->sections.size(); section++) {
    std::vector<Section*> neighbors;
    int section_a = section;
    for (int section_b = section-2; section_b < section+1; section_b++) {
    //for (int section_b = section-1; section_b < section; section_b++) {
      if (section_b < 0 || section_b == section_a || section_b >= this->sections.size()) {
        continue;
      }
      neighbors.push_back(this->sections[section_b]);
    }
    this->sections[section]->get_elastic_matches(neighbors);
  }
}


//START
// only do triangles with vertices in bad areas
void tfk::Stack::elastic_align() {
  cilk_for (int i = 0; i < this->sections.size(); i++) {
    this->sections[i]->construct_triangles();
    this->sections[i]->affine_transform_mesh();
  }

  this->get_elastic_matches();

  this->elastic_gradient_descent();


  //std::vector<tfkMatch> mesh_matches;
  //fine_alignment_3d_2(merged_graph, p_align_data,64.0, mesh_matches);

}

void tfk::Stack::elastic_gradient_descent() {
    double cross_slice_weight = 1.0;
    double cross_slice_winsor = 20.0;
    double intra_slice_weight = 1.0;
    double intra_slice_winsor = 200.0;
    int max_iterations = 10000; //ORIGINALL 5000
    //double min_stepsize = 1e-20;
    double stepsize = 0.0001;
    double momentum = 0.5;

    std::map<int, double> gradient_momentum;
    //std::map<int, graph_section_data> section_data_map;
    //section_data_map.clear();

    //for (int i = 0; i < mesh_matches.size(); i++) {
    //  int az = mesh_matches[i].my_section_data.z;
    //  int bz = mesh_matches[i].n_section_data.z;
    //  if (section_data_map.find(az) == section_data_map.end()) {
    //    section_data_map[az] = mesh_matches[i].my_section_data;
    //  }
    //  if (section_data_map.find(bz) == section_data_map.end()) {
    //    section_data_map[bz] = mesh_matches[i].n_section_data;
    //  }
    //}

    //  for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
    //       it != section_data_map.end(); ++it) {
    //    it->second.gradients = new cv::Point2f[it->second.mesh->size()];
    //    it->second.gradients_with_momentum = new cv::Point2f[it->second.mesh->size()];
    //    it->second.rest_lengths = new double[it->second.triangle_edges->size()];
    //    it->second.rest_areas = new double[it->second.triangles->size()];
    //    it->second.mesh_old = new std::vector<cv::Point2f>();

    //    for (int j = 0; j < it->second.mesh->size(); j++) {
    //      it->second.mesh_old->push_back((*(it->second.mesh))[j]);
    //    }

    //    // init
    //    for (int j = 0; j < it->second.mesh->size(); j++) {
    //      it->second.gradients[j] = cv::Point2f(0.0,0.0);  
    //      it->second.gradients_with_momentum[j] = cv::Point2f(0.0,0.0);  
    //    }

    //    for (int j = 0; j < it->second.triangle_edges->size(); j++) {
    //      cv::Point2f p1 = (*(it->second.mesh))[(*(it->second.triangle_edges))[j].first];
    //      cv::Point2f p2 = (*(it->second.mesh))[(*(it->second.triangle_edges))[j].second];
    //      double dx = p1.x-p2.x;
    //      double dy = p1.y-p2.y;
    //      double len = std::sqrt(dx*dx+dy*dy);
    //      it->second.rest_lengths[j] = len;
    //      //printf("Rest length is %f\n", len);
    //    }

    //    // now triangle areas.
    //    for (int j = 0; j < it->second.triangles->size(); j++) {
    //      tfkTriangle tri = (*(it->second.triangles))[j];
    //      cv::Point2f p1 = (*(it->second.mesh))[tri.index1];
    //      cv::Point2f p2 = (*(it->second.mesh))[tri.index2];
    //      cv::Point2f p3 = (*(it->second.mesh))[tri.index3];
    //      it->second.rest_areas[j] = computeTriangleArea(p1,p2,p3);
    //      //printf("Rest area is %f\n", it->second.rest_areas[j]);
    //    }
    //  }


    // INITIALIZE ALL THE SECTION DATA
    for (int i = 0; i < this->sections.size(); i++) {
      Section* section = this->sections[i];
      section->gradients = new cv::Point2f[section->mesh->size()];
      section->gradients_with_momentum = new cv::Point2f[section->mesh->size()];
      section->rest_lengths = new double[section->triangle_edges->size()];
      section->rest_areas = new double[section->triangles->size()];
      section->mesh_old = new std::vector<cv::Point2f>();

      for (int j = 0; j < section->mesh->size(); j++) {
        section->mesh_old->push_back((*(section->mesh))[j]);
      }

      // init
      for (int j = 0; j < section->mesh->size(); j++) {
        section->gradients[j] = cv::Point2f(0.0,0.0);
        section->gradients_with_momentum[j] = cv::Point2f(0.0,0.0);
      }
      for (int j = 0; j < section->triangle_edges->size(); j++) {
        cv::Point2f p1 = (*(section->mesh))[(*(section->triangle_edges))[j].first];
        cv::Point2f p2 = (*(section->mesh))[(*(section->triangle_edges))[j].second];
        double dx = p1.x-p2.x;
        double dy = p1.y-p2.y;
        double len = std::sqrt(dx*dx+dy*dy);
        section->rest_lengths[j] = len;
      }
      for (int j = 0; j < section->triangles->size(); j++) {
        tfkTriangle tri = (*(section->triangles))[j];
        cv::Point2f p1 = (*(section->mesh))[tri.index1];
        cv::Point2f p2 = (*(section->mesh))[tri.index2];
        cv::Point2f p3 = (*(section->mesh))[tri.index3];
        section->rest_areas[j] = computeTriangleArea(p1,p2,p3);
      }

    }


    // BEGIN THE GRADIENT DESCENT.

    double prev_cost = 0.0;
    for (int iter = 0; iter < max_iterations; iter++) {
      double cost = 0.0;

      //for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
      //     it != section_data_map.end(); ++it) {
      //  // clear old gradients.
      //  for (int j = 0; j < it->second.mesh->size(); j++) {
      //    it->second.gradients[j] = cv::Point2f(0.0,0.0);  
      //  }
      //}

      // reset the old gradients.
      for (int i = 0; i < this->sections.size(); i++) {
        Section* section = this->sections[i];
        for (int j = 0; j < section->mesh->size(); j++) {
          ((section->gradients))[j] = cv::Point2f(0.0,0.0);
        }
      }

      // compute internal gradients.
      //for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
      //     it != section_data_map.end(); ++it) {
      for (int i = 0; i < this->sections.size(); i++) {
        Section* section = this->sections[i];

        // internal_mesh_derivs
        double all_weight = intra_slice_weight;
        double sigma = intra_slice_winsor;
        std::vector<cv::Point2f>* mesh = section->mesh;
        cv::Point2f* gradients = section->gradients;

        std::vector<std::pair<int, int> >* triangle_edges = section->triangle_edges;
        std::vector<tfkTriangle >* triangles = section->triangles;
        double* rest_lengths = section->rest_lengths;
        double* rest_areas = section->rest_areas;

        //// update all edges
        for (int j = 0; j < triangle_edges->size(); j++) {
          cost += internal_mesh_derivs(mesh, gradients, (*triangle_edges)[j], rest_lengths[j],
                                       all_weight, sigma);
        }

        //// update all triangles
        for (int j = 0; j < triangles->size(); j++) {
          int triangle_indices[3] = {(*triangles)[j].index1,
                                     (*triangles)[j].index2,
                                     (*triangles)[j].index3};
          cost += area_mesh_derivs(mesh, gradients, triangle_indices, rest_areas[j], all_weight);
        }
      }


      for (int i = 0; i < this->sections.size(); i++) {
        Section* section = this->sections[i];
        std::vector<tfkMatch>& mesh_matches = section->section_mesh_matches;
        for (int j = 0; j < mesh_matches.size(); j++) {
          Section* my_section = (Section*) mesh_matches[j].my_section;
          Section* n_section = (Section*) mesh_matches[j].n_section;

          std::vector<cv::Point2f>* mesh1 = my_section->mesh;//mesh_matches[j].my_section_data.mesh;
          std::vector<cv::Point2f>* mesh2 = n_section->mesh;//mesh_matches[j].n_section_data.mesh;

          //int myz = mesh_matches[j].my_section_data.z;
          //int nz = mesh_matches[j].n_section_data.z;
          //int myz = my_section->section_id;
          //int nz = n_section->section_id;

          cv::Point2f* gradients1 = my_section->gradients;
          cv::Point2f* gradients2 = n_section->gradients;
          double* barys1 = mesh_matches[j].my_barys;
          double* barys2 = mesh_matches[j].n_barys;
          double all_weight = cross_slice_weight;
          double sigma = cross_slice_winsor;

          int indices1[3] = {mesh_matches[j].my_tri.index1,
                             mesh_matches[j].my_tri.index2,
                             mesh_matches[j].my_tri.index3};
          int indices2[3] = {mesh_matches[j].n_tri.index1,
                             mesh_matches[j].n_tri.index2,
                             mesh_matches[j].n_tri.index3};

          cost += crosslink_mesh_derivs(mesh1, mesh2,
                                        gradients1, gradients2,
                                        indices1, indices2,
                                        barys1, barys2,
                                        all_weight, sigma);

          cost += crosslink_mesh_derivs(mesh2, mesh1,
                                        gradients2, gradients1,
                                        indices2, indices1,
                                        barys2, barys1,
                                        all_weight, sigma);
        }
      }
      if (iter == 0) prev_cost = cost+10.0;

      if (cost <= prev_cost) {
        stepsize *= 1.1;
        if (stepsize > 1.0) {
          stepsize = 1.0;
        }
        // TODO(TFK): momentum.

        //for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
        //     it != section_data_map.end(); ++it) {
        for (int i = 0; i < this->sections.size(); i++) {
          Section* section = this->sections[i];
          std::vector<cv::Point2f>* mesh = section->mesh;
          std::vector<cv::Point2f>* mesh_old = section->mesh_old;
          cv::Point2f* gradients = section->gradients;
          cv::Point2f* gradients_with_momentum = section->gradients_with_momentum;
          for (int j = 0; j < mesh->size(); j++) {
            gradients_with_momentum[j] = gradients[j] + momentum*gradients_with_momentum[j];
          }

          for (int j = 0; j < mesh->size(); j++) {
            (*mesh_old)[j] = ((*mesh)[j]);
          }
          for (int j = 0; j < mesh->size(); j++) {
            (*mesh)[j].x -= (float)(stepsize * (gradients_with_momentum)[j].x);
            (*mesh)[j].y -= (float)(stepsize * (gradients_with_momentum)[j].y);
          }
        }
        if (iter%100 == 0) {
          printf("Good step old cost %f, new cost %f, iteration %d\n", prev_cost, cost, iter);
        }
        prev_cost = cost;
      } else {
        stepsize *= 0.5;
        // bad step undo.
        //for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
        //     it != section_data_map.end(); ++it) {
        for (int i = 0; i < this->sections.size(); i++) {
          Section* section = this->sections[i];
          std::vector<cv::Point2f>* mesh = section->mesh;
          std::vector<cv::Point2f>* mesh_old = section->mesh_old;
          cv::Point2f* gradients_with_momentum = section->gradients_with_momentum;
          for (int j = 0; j < mesh->size(); j++) {
            gradients_with_momentum[j] = cv::Point2f(0.0,0.0);
          }

          //if (mesh_old->size() != mesh->size()) continue;
          for (int j = 0; j < mesh->size(); j++) {
            (*mesh)[j] = (*mesh_old)[j];
          }
        }
        if (iter%1000 == 0) {
          printf("Bad step old cost %f, new cost %f, iteration %d\n", prev_cost, cost, iter);
        }
      }
    }
    printf("Done with the elastic gradient descent\n");
}


void tfk::Stack::init() {
  printf("Do the init\n");

  AlignData align_data;
  // Read the existing address book.
  std::fstream input(this->input_filepath, std::ios::in | std::ios::binary);
  if (!align_data.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse protocal buffer." << std::endl;
    exit(1);
  }
  // first deeal with AlignData level
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


void tfk::Stack::pack_graph() {
  std::vector<Graph* > graph_list;
  for (int i = 0; i < graph_list.size(); i++) {
    graph_list.push_back(this->sections[i]->graph);
  }

  this->merged_graph = new Graph();
  // Merging the graphs in graph_list into a single merged graph.
  int total_size = 0;
  for (int i = 0; i < graph_list.size(); i++) {
    total_size += graph_list[i]->num_vertices();
  }
  this->merged_graph->resize(total_size);

  int vertex_id_offset = 0;
  for (int i = 0; i < graph_list.size(); i++) {
    for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
      vdata* d = this->merged_graph->getVertexData(j+vertex_id_offset);
      *d = *(graph_list[i]->getVertexData(j));
      d->vertex_id += vertex_id_offset;
    }
    vertex_id_offset += graph_list[i]->num_vertices();
  }

  vertex_id_offset = 0;
  // now insert the edges.
  for (int i = 0; i < graph_list.size(); i++) {
    for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
      for (int k = 0; k < graph_list[i]->edgeData[j].size(); k++) {
        edata edge = graph_list[i]->edgeData[j][k];
        edge.neighbor_id += vertex_id_offset;
        this->merged_graph->insertEdge(j+vertex_id_offset, edge);
      }
    }
    vertex_id_offset += graph_list[i]->num_vertices();
  }
}

void tfk::Stack::unpack_graph() {


  for (int _i = 0; _i < this->sections.size(); _i++) {
    Section* section = this->sections[_i];
    std::string section_id_string =
        std::to_string(section->section_id +
        this->base_section+1);
    FILE* wafer_file = fopen((std::string(this->output_dirpath)+std::string("/W01_Sec") +
        matchPadTo(section_id_string, 3)+std::string("_montaged.json")).c_str(), "w+");
    section->write_wafer(wafer_file, this->base_section);
    fclose(wafer_file);
  }

  return;
}


void tfk::Stack::align_2d() {
  cilk_for (int i = 0; i < this->sections.size(); i++) {
    this->sections[i]->compute_keypoints_and_matches();
  }

  cilk_for (int section_index = 0; section_index < this->sections.size(); section_index++) {

    int ncolors = this->sections[section_index]->graph->compute_trivial_coloring();
    printf("ncolors is %d\n", ncolors);
    Scheduler* scheduler;
    engine* e;
    scheduler =
        new Scheduler(this->sections[section_index]->graph->vertexColors, ncolors+1, this->sections[section_index]->graph->num_vertices());
    scheduler->graph_void = (void*) this->sections[section_index]->graph;
    scheduler->roundNum = 0;
    e = new engine(this->sections[section_index]->graph, scheduler);

    for (int trial = 0; trial < 5; trial++) {
      //global_learning_rate = 0.49;
      std::vector<int> vertex_ids;
      for (int i = 0; i < this->sections[section_index]->graph->num_vertices(); i++) {
        vertex_ids.push_back(i);
      }
      //std::srand(trial);
      //std::random_shuffle(vertex_ids.begin(), vertex_ids.end());
      // pick one section to be "converged"
      for (int i = 0; i < this->sections[section_index]->graph->num_vertices(); i++) {
        this->sections[section_index]->graph->getVertexData(i)->iteration_count = 0;
      }
      std::set<int> section_list;
      for (int _i = 0; _i < this->sections[section_index]->graph->num_vertices(); _i++) {
        int i = _i;//vertex_ids[_i];
        int z = this->sections[section_index]->graph->getVertexData(i)->z;
        this->sections[section_index]->graph->getVertexData(i)->iteration_count = 0;
        if (section_list.find(z) == section_list.end()) {
          if (this->sections[section_index]->graph->edgeData[i].size() > 4) {
            section_list.insert(z);
          }
        }
      }

      scheduler->isStatic = false;
      for (int i = 0; i < this->sections[section_index]->graph->num_vertices(); i++) {
        scheduler->add_task_static(i, updateTile2DAlign); //updateVertex2DAlignFULLFast);
      }
      scheduler->isStatic = true;

      printf("starting run\n");
      e->run();
      printf("ending run\n");

      break;
    }
  }

}


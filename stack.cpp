#include "stack.hpp"

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

//  for (int _i = 0; _i < graph_list.size(); _i++) {
//    Graph* graph = graph_list[_i];
//    int sec_id = _i;//graph->section_id;
//    std::string section_id_string =
//        std::to_string(this->sections[sec_id]->section_id +
//        this->base_section+1);
//
//
//    //std::string transform_string = "";
//    //for (int i = 0; i < graph->num_vertices(); i++) {
//    //  if (i == 0) {
//    //    transform_string += get_point_transform_string(graph, graph->getVertexData(i));
//    //  } else {
//    //    transform_string += " " + get_point_transform_string(graph, graph->getVertexData(i));
//    //  }
//    //  
//    //}
//    FILE* wafer_file = fopen((std::string(this->output_dirpath)+std::string("/W01_Sec") +
//        matchPadTo(section_id_string, 3)+std::string("_montaged.json")).c_str(), "w+");
//    fprintf(wafer_file, "[\n");
//
//    for (int i = 0; i < graph->num_vertices(); i++) {
//      //printf("affine params %f %f %f %f\n", graph->getVertexData(i)->a00, graph->getVertexData(i)->a01, graph->getVertexData(i)->a10, graph->getVertexData(i)->a11);
//      vdata* vd = graph->getVertexData(i);
//      this->sections[sec_id]->tiles[vd->tile_id]->x_start = vd->start_x+vd->offset_x;
//      this->sections[sec_id]->tiles[vd->tile_id]->x_finish = vd->end_x + vd->offset_x;
//      this->sections[sec_id]->tiles[vd->tile_id]->y_start = vd->start_y+vd->offset_y;
//      this->sections[sec_id]->tiles[vd->tile_id]->y_finish = vd->end_y + vd->offset_y;
//      this->sections[sec_id]->tiles[vd->tile_id]->a00 = vd->a00;
//      this->sections[sec_id]->tiles[vd->tile_id]->a01 = vd->a01;
//      this->sections[sec_id]->tiles[vd->tile_id]->a10 = vd->a10;
//      this->sections[sec_id]->tiles[vd->tile_id]->a11 = vd->a11;
//      this->sections[sec_id]->tiles[vd->tile_id]->offset_x = vd->offset_x + vd->start_x;
//      this->sections[sec_id]->tiles[vd->tile_id]->offset_y = vd->offset_y + vd->start_y;
//
//      #ifdef ALIGN3D
//      // find all the mesh triangles that overlap with this tile.
//      std::set<int> mesh_point_set;
//      for (int j = 0; j < vd->my_mesh_points->size(); j++) {
//        int pt_index = (*(vd->my_mesh_points))[j];
//        //cv::Point2f pt = (*vd->section_data->mesh_orig)[pt_index];
//        mesh_point_set.insert(pt_index);
//      }
//
//      std::set<int> tri_added_set; 
//      std::vector<renderTriangle>* mesh_triangles = new std::vector<renderTriangle>(); 
//      for (int j = 0; j < vd->section_data->triangles->size(); j++) {
//        tfkTriangle tri = (*vd->section_data->triangles)[j];
//        int points[3];
//        points[0] = tri.index1;
//        points[1] = tri.index2;
//        points[2] = tri.index3;
//        for (int k = 0; k < 3; k++) {
//          if (mesh_point_set.find(points[k]) != mesh_point_set.end()) {
//            renderTriangle r_tri;
//            r_tri.p[0] = (*vd->section_data->mesh_orig)[tri.index1]; 
//            r_tri.p[1] = (*vd->section_data->mesh_orig)[tri.index2]; 
//            r_tri.p[2] = (*vd->section_data->mesh_orig)[tri.index3];
//            r_tri.q[0] = (*vd->section_data->mesh)[tri.index1]; 
//            r_tri.q[1] = (*vd->section_data->mesh)[tri.index2]; 
//            r_tri.q[2] = (*vd->section_data->mesh)[tri.index3];
//			r_tri.key = std::make_pair(vd->z, j);
//            if (tri_added_set.find(j) == tri_added_set.end()) {
//              tri_added_set.insert(j);
//              mesh_triangles->push_back(r_tri);
//            }
//          }
//        }
//      }
//
//
//      this->sections[sec_id]->tiles[vd->tile_id]->mesh_triangles = mesh_triangles;
//      #endif // ALIGN3D
//
//      fprintf(wafer_file, "\t{\n");
//      fprintf(wafer_file, "\t\t\"bbox\": [\n");
//
//      fprintf(wafer_file,
//          "\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f\n],",
//          vd->start_x+vd->offset_x, (vd->end_x+vd->offset_x),
//          vd->start_y+vd->offset_y, (vd->end_y+vd->offset_y));
//      fprintf(wafer_file, "\t\t\"height\": %d,\n",SIFT_D1_SHIFT_3D);
//      fprintf(wafer_file, "\t\t\"layer\": %d,\n",this->sections[sec_id]->section_id + this->base_section+1);
//      fprintf(wafer_file, "\t\t\"maxIntensity\": %f,\n",255.0);
//      fprintf(wafer_file, "\t\t\"mfov\": %d,\n",
//          graph->getVertexData(i)->mfov_id);
//      fprintf(wafer_file, "\t\t\"minIntensity\": %f,\n",
//          0.0);
//      fprintf(wafer_file, "\t\t\"mipmapLevels\": {\n");
//      fprintf(wafer_file, "\t\t\"0\": {\n");
//      fprintf(wafer_file, "\t\t\t\"imageUrl\": \"%s\"\n", this->sections[sec_id]->tiles[graph->getVertexData(i)->tile_id]->filepath.c_str());
//      fprintf(wafer_file, "\t\t\t}\n");
//      fprintf(wafer_file, "\t\t},\n");
//      fprintf(wafer_file, "\t\t\"tile_index\": %d,\n",
//          graph->getVertexData(i)->tile_index);
//      fprintf(wafer_file, "\t\t\"transforms\": [\n");
//      // {'className': 'mpicbg.trakem2.transform.AffineModel2D', 'dataString': '0.1 0.0 0.0 0.1 0.0 0.0'}
//
//      fprintf(wafer_file, "\t\t\t{\n");
//      fprintf(wafer_file,
//          "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.AffineModel2D\",\n");
//      fprintf(wafer_file,
//          "\t\t\t\t\"dataString\": \"%f %f %f %f %f %f\"\n", vd->a00,
//          vd->a10, vd->a01, vd->a11, vd->start_x+vd->offset_x, vd->start_y+vd->offset_y);
//      #ifdef ALIGN3D
//      if (true) {
//      #else
//      if (false) {
//      #endif 
//      fprintf(wafer_file,
//          "\t\t\t},\n");
//
//      fprintf(wafer_file, "\t\t\t{\n");
//      fprintf(wafer_file,
//          "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.PointsTransformModel\",\n");
//      fprintf(wafer_file,
//          "\t\t\t\t\"dataString\": \"%s\"\n", get_point_transform_string(graph, vd).c_str());
//      fprintf(wafer_file,
//          "\t\t\t}\n");
//      } else {
//       fprintf(wafer_file,
//          "\t\t\t}\n");
//      }
//
//      fprintf(wafer_file,
//          "\t\t],\n");
//      fprintf(wafer_file,
//          "\t\t\"width\":%d\n",SIFT_D2_SHIFT_3D);
//
//      if (i != graph->num_vertices()-1) {
//        fprintf(wafer_file,
//            "\t},\n");
//      } else {
//        fprintf(wafer_file,
//            "\t}\n]");
//      }
//    }
//    fclose(wafer_file);
//  }
//


}


void tfk::Stack::align_2d() {
  for (int i = 0; i < this->sections.size(); i++) {
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

      //for (int i = 0; i < this->sections[section_index]->graph->num_vertices(); i++) {
      //  this->sections[section_index]->graph->getVertexData(i)->iteration_count = 0;
      //  computeError2DAlign(i, (void*) scheduler);
      //}
      break;
    }
  }

}


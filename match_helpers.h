Graph<vdata, edata>* pack_graph() {//std::vector<Graph<vdata, edata>* >& graph_list) {
  // Merging the graphs in graph_list into a single merged graph.
  int total_size = 0;
  for (int i = 0; i < graph_list.size(); i++) {
    total_size += graph_list[i]->num_vertices();
  }
  Graph<vdata, edata>* merged_graph = new Graph<vdata, edata>();
  merged_graph->resize(total_size);

  int vertex_id_offset = 0;
  for (int i = 0; i < graph_list.size(); i++) {
    for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
      vdata* d = merged_graph->getVertexData(j+vertex_id_offset);
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
        merged_graph->insertEdge(j+vertex_id_offset, edge);
      }
    }
    vertex_id_offset += graph_list[i]->num_vertices();
  }

  return merged_graph;
}




// Takes the merged graph (with all sections in one graph) and unpacks it into the
//   already correctly-sized graphs in the graph_list --- where graph_list has
//   one graph per-section.
void unpack_graph(align_data_t* p_align_data, Graph<vdata,edata>* merged_graph) {
  int vertex_id_offset = 0;
  // Unpack the graphs within the merged graph.
  vertex_id_offset = 0;
  for (int i = 0; i < graph_list.size(); i++) {
    for (int j = 0; j < graph_list[i]->num_vertices(); j++) {
      vdata* d = merged_graph->getVertexData(j+vertex_id_offset);
      *(graph_list[i]->getVertexData(j)) = *d;
      (graph_list[i]->getVertexData(j))->vertex_id -= vertex_id_offset;
    }
    vertex_id_offset += graph_list[i]->num_vertices();
  }

  for (int _i = 0; _i < graph_list.size(); _i++) {
    Graph<vdata, edata>* graph = graph_list[_i];
    int sec_id = graph->section_id;
    std::string section_id_string =
        std::to_string(p_align_data->sec_data[sec_id].section_id +
        p_align_data->base_section+1);


    //std::string transform_string = "";
    //for (int i = 0; i < graph->num_vertices(); i++) {
    //  if (i == 0) {
    //    transform_string += get_point_transform_string(graph, graph->getVertexData(i));
    //  } else {
    //    transform_string += " " + get_point_transform_string(graph, graph->getVertexData(i));
    //  }
    //  
    //}

    FILE* wafer_file = fopen((std::string(p_align_data->output_dirpath)+std::string("/W01_Sec") +
        matchPadTo(section_id_string, 3)+std::string("_montaged.json")).c_str(), "w+");
    fprintf(wafer_file, "[\n");

    for (int i = 0; i < graph->num_vertices(); i++) {
      //printf("affine params %f %f %f %f\n", graph->getVertexData(i)->a00, graph->getVertexData(i)->a01, graph->getVertexData(i)->a10, graph->getVertexData(i)->a11);
      vdata* vd = graph->getVertexData(i);
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].x_start = vd->start_x+vd->offset_x;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].x_finish = vd->end_x + vd->offset_x;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].y_start = vd->start_y+vd->offset_y;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].y_finish = vd->end_y + vd->offset_y;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].a00 = vd->a00;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].a01 = vd->a01;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].a10 = vd->a10;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].a11 = vd->a11;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].offset_x = vd->offset_x + vd->start_x;
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].offset_y = vd->offset_y + vd->start_y;


      // find all the mesh triangles that overlap with this tile.
      std::set<int> mesh_point_set;
      for (int j = 0; j < vd->my_mesh_points->size(); j++) {
        int pt_index = (*(vd->my_mesh_points))[j];
        //cv::Point2f pt = (*vd->section_data->mesh_orig)[pt_index];
        mesh_point_set.insert(pt_index);
      }

      std::set<int> tri_added_set; 
      std::vector<renderTriangle>* mesh_triangles = new std::vector<renderTriangle>(); 
      for (int j = 0; j < vd->section_data->triangles->size(); j++) {
        tfkTriangle tri = (*vd->section_data->triangles)[j];
        int points[3];
        points[0] = tri.index1;
        points[1] = tri.index2;
        points[2] = tri.index3;
        for (int k = 0; k < 3; k++) {
          if (mesh_point_set.find(points[k]) != mesh_point_set.end()) {
            renderTriangle r_tri;
            r_tri.p[0] = (*vd->section_data->mesh_orig)[tri.index1]; 
            r_tri.p[1] = (*vd->section_data->mesh_orig)[tri.index2]; 
            r_tri.p[2] = (*vd->section_data->mesh_orig)[tri.index3];
            r_tri.q[0] = (*vd->section_data->mesh)[tri.index1]; 
            r_tri.q[1] = (*vd->section_data->mesh)[tri.index2]; 
            r_tri.q[2] = (*vd->section_data->mesh)[tri.index3];
			r_tri.key = std::make_pair(vd->z, j);
            if (tri_added_set.find(j) == tri_added_set.end()) {
              tri_added_set.insert(j);
              mesh_triangles->push_back(r_tri);
            }
          }
        }
      }
      
      p_align_data->sec_data[sec_id].tiles[vd->tile_id].mesh_triangles = mesh_triangles;

      fprintf(wafer_file, "\t{\n");
      fprintf(wafer_file, "\t\t\"bbox\": [\n");

      fprintf(wafer_file,
          "\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f,\n\t\t\t%f\n],",
          vd->start_x+vd->offset_x, (vd->end_x+vd->offset_x),
          vd->start_y+vd->offset_y, (vd->end_y+vd->offset_y));
      fprintf(wafer_file, "\t\t\"height\": %d,\n",SIFT_D1_SHIFT_3D);
      fprintf(wafer_file, "\t\t\"layer\": %d,\n",p_align_data->sec_data[sec_id].section_id + p_align_data->base_section+1);
      fprintf(wafer_file, "\t\t\"maxIntensity\": %f,\n",255.0);
      fprintf(wafer_file, "\t\t\"mfov\": %d,\n",
          graph->getVertexData(i)->mfov_id);
      fprintf(wafer_file, "\t\t\"minIntensity\": %f,\n",
          0.0);
      fprintf(wafer_file, "\t\t\"mipmapLevels\": {\n");
      fprintf(wafer_file, "\t\t\"0\": {\n");
      fprintf(wafer_file, "\t\t\t\"imageUrl\": \"%s\"\n", p_align_data->sec_data[sec_id].tiles[graph->getVertexData(i)->tile_id].filepath);
      fprintf(wafer_file, "\t\t\t}\n");
      fprintf(wafer_file, "\t\t},\n");
      fprintf(wafer_file, "\t\t\"tile_index\": %d,\n",
          graph->getVertexData(i)->tile_index);
      fprintf(wafer_file, "\t\t\"transforms\": [\n");
      // {'className': 'mpicbg.trakem2.transform.AffineModel2D', 'dataString': '0.1 0.0 0.0 0.1 0.0 0.0'}

      fprintf(wafer_file, "\t\t\t{\n");
      fprintf(wafer_file,
          "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.AffineModel2D\",\n");
      fprintf(wafer_file,
          "\t\t\t\t\"dataString\": \"%f %f %f %f %f %f\"\n", vd->a00,
          vd->a10, vd->a01, vd->a11, vd->start_x+vd->offset_x, vd->start_y+vd->offset_y);
      #ifdef ALIGN3D
      if (true) {
      #else
      if (false) {
      #endif 
      fprintf(wafer_file,
          "\t\t\t},\n");

      fprintf(wafer_file, "\t\t\t{\n");
      fprintf(wafer_file,
          "\t\t\t\t\"className\": \"mpicbg.trakem2.transform.PointsTransformModel\",\n");
      fprintf(wafer_file,
          "\t\t\t\t\"dataString\": \"%s\"\n", get_point_transform_string(graph, vd).c_str());
      fprintf(wafer_file,
          "\t\t\t}\n");
      } else {
       fprintf(wafer_file,
          "\t\t\t}\n");
      }

      fprintf(wafer_file,
          "\t\t],\n");
      fprintf(wafer_file,
          "\t\t\"width\":%d\n",SIFT_D2_SHIFT_3D);
      if (i != graph->num_vertices()-1) {
        fprintf(wafer_file,
            "\t},\n");
      } else {
        fprintf(wafer_file,
            "\t}\n]");
      }
    }
    fclose(wafer_file);
  }


}




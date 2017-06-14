
bool section_data_exists(int section, align_data_t* p_align_data) {
//  return false;
  std::string filename = std::string("cached_data/prefix_")+std::to_string(section+p_align_data->base_section+1);
  cv::FileStorage fs(filename+std::string("_3d_keypoints.yml.gz"), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  } else {
    return true;
  }
}

void store_3d_matches(int section, align_data_t* p_align_data) {
  std::string filename = std::string("cached_data/prefix_")+std::to_string(section+p_align_data->base_section+1);
  cv::FileStorage fs(filename+std::string("_3d_keypoints.yml.gz"), cv::FileStorage::WRITE);
  for (int i = 0; i < p_align_data->sec_data[section].n_tiles; i++) {
    cv::write(fs, "keypoints_"+std::to_string(i), *(p_align_data->sec_data[section].tiles[i].p_kps_3d));
    cv::write(fs, "descriptors_" + std::to_string(i), *(p_align_data->sec_data[section].tiles[i].p_kps_desc_3d));
  }
  fs.release();
}

void read_3d_matches(int section, align_data_t* p_align_data) {
  std::string filename = std::string("cached_data/prefix_")+std::to_string(section+p_align_data->base_section+1);
  cv::FileStorage fs(filename+std::string("_3d_keypoints.yml.gz"), cv::FileStorage::READ);
  int count = 0;
  for (int i = 0; i < p_align_data->sec_data[section].n_tiles; i++) {
    fs["keypoints_"+std::to_string(i)] >> *(p_align_data->sec_data[section].tiles[i].p_kps_3d);
    count += p_align_data->sec_data[section].tiles[i].p_kps_3d->size();
    fs["descriptors_"+std::to_string(i)] >> *(p_align_data->sec_data[section].tiles[i].p_kps_desc_3d);
  }
  fs.release();
  printf("Read %d 3d matches for section %d\n", count, section);
}


void store_2d_graph(Graph<vdata, edata>* graph, int section,
    align_data_t* p_align_data) {

  std::string filename = std::string("cached_data/prefix_")+std::to_string(section+p_align_data->base_section+1);
  cv::FileStorage fs(filename+std::string("_2d_matches.yml.gz"), cv::FileStorage::WRITE);
  for (int i = 0; i < graph->num_vertices(); i++) {
      cv::write(fs, "num_edges_"+std::to_string(i), (int) graph->edgeData[i].size());
    for (int j = 0; j < graph->edgeData[i].size(); j++) {
      cv::write(fs, "neighbor_id_"+std::to_string(i) + "_" + std::to_string(j),
          graph->edgeData[i][j].neighbor_id);
      cv::write(fs, "weight_"+std::to_string(i) + "_" + std::to_string(j),
          graph->edgeData[i][j].neighbor_id);
      cv::write(fs, "v_points_"+std::to_string(i)+"_"+std::to_string(j),
          *(graph->edgeData[i][j].v_points));
      cv::write(fs, "n_points_"+std::to_string(i)+"_"+std::to_string(j),
          *(graph->edgeData[i][j].n_points));
    }
  }
  fs.release();
}

void read_graph_from_file(Graph<vdata, edata>* graph, int section, align_data_t* p_align_data) {
  std::string filename = std::string("cached_data/prefix_")+std::to_string(section+p_align_data->base_section+1);
  cv::FileStorage fs(filename+std::string("_2d_matches.yml.gz"), cv::FileStorage::READ);
  for (int i = 0; i < graph->num_vertices(); i++) {
    int edge_size;
    fs["num_edges_"+std::to_string(i)] >> edge_size; 
    std::vector<edata> edge_data;
    std::set<int> n_ids_seen;
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
      edge_data.push_back(edge);
    }
    graph->edgeData[i] = edge_data;
  }
  //printf("Read %d 3d matches for section %d\n", count, section);
  fs.release();
}





//void store_3d_matches(int section, align_data_t* p_align_data) {
//  std::string filename = std::string("cached_data/prefix_")+std::to_string(section);
//  cv::FileStorage fs(filename+std::string("_3d_keypoints"), FileStorage::WRITE);
//  for (int i = 0; i < p_align_data->sec_data[section].n_tiles; i++) {
//    cv::write(fs, "keypoints_"+std::to_string(i), *(p_align_data->sec_data[section].tiles[i]->p_kps_3d)));
//    cv::write(fs, "descriptors_" + std::to_string(i), *(p_align_data->sec_data[section].tiles[i]->p_kps_desc_3d));
//  }
//  fs.release();
//}


void store_3d_matches(int section, align_data_t* p_align_data) {
  std::string filename = std::string("prefix_")+std::to_string(section);
  cv::FileStorage fs(filename+std::string("_3d_keypoints"), FileStorage::WRITE);
  for (int i = 0; i < p_align_data->sec_data[section].n_tiles; i++) {
    cv::write(fs, "keypoints_"+std::to_string(i), *(p_align_data->sec_data[section].tiles[i]->p_kps_3d)));
    cv::write(fs, "descriptors_" + std::to_string(i), *(p_align_data->sec_data[section].tiles[i]->p_kps_desc_3d));
  }
  fs.release();
}

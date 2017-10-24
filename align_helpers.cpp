#include "AlignData.pb.cc"


std::string padTo(std::string str, const size_t num, const char paddingChar = '0')
{
    if(num > str.size())
        str.insert(0, num - str.size(), paddingChar);
    return str;
}




//void output_section_image(section_data_t* section, int min_x, int min_y,
//   int max_x,
//   int max_y, std::string filename) {
//
//  //int min_x = 100000000.0;
//  //int min_y = 100000000.0;
//  //int max_x = 0.0;
//  //int max_y = 0.0;
//  //for (int i = 0; i < section->n_tiles; i++) {
//  //  tile_data_t tile = section->tiles[i];
//  //  if (tile.x_start < min_x) {
//  //    min_x = tile.x_start;
//  //  }
//  //  if (tile.x_end > max_x) {
//  //    max_x = tile.x_end;
//  //  }
//  //  if (tile.y_start < min_y) {
//  //    min_y = tile.y_start;
//  //  }
//  //  if (tile.y_end > max_y) {
//  //    max_y = tile.y_end;
//  //  }
//  //}
//
//  int nrows = max_y-min_y;
//  int ncols = max_x-min_x;
//  section->p_out = new cv::Mat();
//  (*section->p_out).create(nrows, ncols, CV_8UC1);
//
//  for (int i = 0; i < section->n_tiles; i++) {
//    tile_data_t tile = section->tiles[i];
//    (*tile.p_image).create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
//    (*tile.p_image) = cv::imread(
//        tile.filepath,
//        CV_LOAD_IMAGE_UNCHANGED);
//
//    for (int y = tile.y_start; y < tile.y_finish; y++) {
//      for (int x = tile.x_start; x < tile.x_finish; x++) {
//        unsigned char val =
//            tile.p_image->at<unsigned char>((int)(y-tile.y_start), (int)(x-tile.x_start));
//        section->p_out->at<unsigned char>(y-min_y, x-min_x) = val;
//      }
//    }
//    tile.p_image->release();
//  }
//
//    //cv::Mat outImage;
//    //drawKeypoints(
//    //    *(section->p_out),
//    //    *(section->p_kps),
//    //    outImage,
//    //    cv::Scalar::all(-1), 
//    //    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//    //sprintf(filepath, "%s/kps_tile_%.4d_%.4d_%.4d.tif", 
//    //sprintf(filepath, "%s/kps_tile_%.4d_%.4d_%.4d.tif", 
//    //    LOG_DIR, 
//    //    p_tile_data->section_id,
//    //    p_tile_data->mfov_id,
//    //    p_tile_data->index);
//
//    //cv::imwrite(std::string("raw_") + filename, outImage);
//
//
//  cv::Mat outImage2;
//  cv::resize((*section->p_out), outImage2, cv::Size(), 0.5,0.5);
//  //cv::imwrite("outimagetest1.tif", (*section->p_out));
//  cv::imwrite(filename, outImage2);
//}


void create_sift_hdf5(const char* filename, int num_points, std::vector<float> sizes,
    std::vector<float> responses, std::vector<float> octaves,
    std::vector<float> locations, const char* imageUrl){
   hid_t       file_id, dataset_id, dataspace_id;  /* identifiers */
   hsize_t     dims[2];

   /* Create a new file using default properties. */
   file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

   /* Create the data space for the dataset. */
   dims[0] = num_points; 
   dims[1] = 128; 
   dataspace_id = H5Screate_simple(2, dims, NULL);


   hsize_t dims2[1];
   dims2[0] = 1;
   hid_t dataspace_id2 = H5Screate_simple(0, dims2, NULL);

   /* Create the dataset. */
   dataset_id = H5Dcreate2(file_id, "/descs", H5T_STD_U8LE, dataspace_id, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   ///* End access to the dataset and release resources used by it. */
   H5Dclose(dataset_id);


   hid_t filetype = H5Tcopy(H5T_C_S1);
   H5Tset_size(filetype, strlen(imageUrl));
   H5Tset_strpad(filetype, H5T_STR_NULLPAD);
   hid_t memtype = H5Tcopy(H5T_C_S1);
   H5Tset_size(memtype, strlen(imageUrl));
   dataset_id = H5Dcreate2(file_id, "/imageUrl", filetype, dataspace_id2, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dataset_id, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, imageUrl); 
   H5Dclose(dataset_id);


   hid_t group_id = H5Gcreate2(file_id, "/pts", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Gclose(group_id);


   hsize_t dims3[2];
   dims3[0] = num_points;
   dims3[1] = 2;
   hid_t dataspace_id3 = H5Screate_simple(2, dims3, NULL);
   dataset_id = H5Dcreate2(file_id, "/pts/locations", H5T_IEEE_F64LE, dataspace_id3, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(locations[0])); 
  

   hsize_t dims4[1];
   dims4[0] = num_points; 
   hid_t dataspace_id4 = H5Screate_simple(1, dims4, NULL);
   dataset_id = H5Dcreate2(file_id, "/pts/octaves", H5T_IEEE_F64LE, dataspace_id4, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(octaves[0])); 

   dataspace_id4 = H5Screate_simple(1, dims4, NULL);
   dataset_id = H5Dcreate2(file_id, "/pts/responses", H5T_IEEE_F64LE, dataspace_id4, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(responses[0])); 

   dataspace_id4 = H5Screate_simple(1, dims4, NULL);
   dataset_id = H5Dcreate2(file_id, "/pts/sizes", H5T_IEEE_F64LE, dataspace_id4, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(sizes[0])); 

   H5Dclose(dataset_id);

   /* Terminate access to the data space. */ 
   H5Sclose(dataspace_id);

   /* Close the file. */
   H5Fclose(file_id);

}

/////////////////////////////////////////////////////////////////////////////////////////
// INTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////

void read_input(align_data_t *p_align_data) {
    FILE *fp;
    int n_tiles;
    int n_objs_read;
    char section_found[MAX_SECTIONS] = {0,};
    char str_input[MAX_INPUT_BUF] = {0,};
    
    section_data_t *p_sec_data;
    tile_data_t *p_cur_tile;
    int in_tile_id;
    int in_section_id;
    int in_mfov_id;
    int in_index;
    int in_x_start;
    int in_x_finish;
    int in_y_start;
    int in_y_finish;
    char in_filepath[MAX_FILEPATH];
    
    char magic_str_total_tiles[] = "total-tiles:";
    char magic_str_tile_start[] = "tile-start:";
    char magic_str_tile_end[] = "tile-end:";
    
    char magic_str_tile_section[] = "tile-section:";
    char magic_str_tile_mfov[] = "tile-mfov:";
    char magic_str_tile_index[] = "tile-index:";
    char magic_str_tile_filepath[] = "tile-filepath:";
    char magic_str_tile_bbox[] = "tile-bbox:";
    
    ////TRACE_1("start\n");
    
    ASSERT(p_align_data->input_filepath != NULL);
    
    fp = fopen(p_align_data->input_filepath, "rb");
    ASSERT_MSG(fp != NULL, "failed to open %s\n", p_align_data->input_filepath);
    
    n_objs_read = fscanf(fp, "%s %d\n", str_input, &n_tiles);
    ASSERT(n_objs_read == 2);
    ASSERT(0 == strcmp(str_input, magic_str_total_tiles));
    
    int cur_section_idx = 0;
 
    for (int i = 0; i < n_tiles; i++) {
        
        //printf("String input %s\n", str_input);
        n_objs_read = fscanf(fp, "%s [%d]\n", str_input, &in_tile_id);
        ASSERT(n_objs_read == 2);
        ASSERT(0 == strcmp(str_input, magic_str_tile_start));
        ASSERT(in_tile_id == i);
        n_objs_read = fscanf(fp, "\t%s %d\n", str_input, &in_section_id);
        //printf("in_section id is %d\n", in_section_id);
        ASSERT(n_objs_read == 2);
        ASSERT(0 == strcmp(str_input, magic_str_tile_section));
        ASSERT(in_section_id >= 1);
        in_section_id--;
        //ASSERT(in_section_id >= 0);
        
        n_objs_read = fscanf(fp, "\t%s %d\n", str_input, &in_mfov_id);
        ASSERT(n_objs_read == 2);
        ASSERT(0 == strcmp(str_input, magic_str_tile_mfov));
        
        n_objs_read = fscanf(fp, "\t%s %d\n", str_input, &in_index);
        ASSERT(n_objs_read == 2);
        ASSERT(0 == strcmp(str_input, magic_str_tile_index));
        
        n_objs_read = fscanf(fp, "\t%s [%d][%d][%d][%d]\n", str_input, 
            &in_x_start,
            &in_x_finish,
            &in_y_start,
            &in_y_finish);
        ASSERT(n_objs_read == 5);
        ASSERT(0 == strcmp(str_input, magic_str_tile_bbox));
                
        n_objs_read = fscanf(fp, "\t%s %s\n", str_input, in_filepath);
        ASSERT(n_objs_read == 2);
        ASSERT(0 == strcmp(str_input, magic_str_tile_filepath));
        
        n_objs_read = fscanf(fp, "%s [%d]\n", str_input, &in_tile_id);
        ASSERT(n_objs_read == 2);
        ASSERT(0 == strcmp(str_input, magic_str_tile_end));
        ASSERT(in_tile_id == i);
        
        if ((in_section_id < p_align_data->base_section) || 
            (in_section_id >= (p_align_data->base_section + p_align_data->n_sections))) {
            continue;
        }
       
        if (p_align_data->do_subvolume) {
          float MIN_X = p_align_data->min_x;
          float MIN_Y = p_align_data->min_y;
          float MAX_X = p_align_data->max_x;
          float MAX_Y = p_align_data->max_y;
  
          if (in_x_start < MIN_X ||
              in_y_start < MIN_Y ||
              in_x_finish > MAX_X ||
              in_y_finish > MAX_Y) {
            continue;
          }
        }

        cur_section_idx = in_section_id - p_align_data->base_section;
        
        ASSERT(cur_section_idx < MAX_SECTIONS);
        //TRACE_2("\nread tile\n");
        //TRACE_2("   section_id : %d\n", in_section_id);
        //TRACE_2("   mfov_id    : %d\n", in_mfov_id);
        //TRACE_2("   index      : %d\n", in_index);
        //TRACE_2("   bbox       : [%d-%d, %d-%d]\n", in_x_start, in_x_finish, in_y_start, in_y_finish);
        //TRACE_2("   filepath   : %s\n", in_filepath);
        
        p_sec_data = &(p_align_data->sec_data[cur_section_idx]);
        p_cur_tile = &(p_sec_data->tiles[p_sec_data->n_tiles]);
        //printf("Init tile with sindex %d tindex %d\n", cur_section_idx, p_sec_data->n_tiles); 
        init_tile(
            p_cur_tile,
            cur_section_idx,
            in_mfov_id,
            in_index,
            in_x_start,
            in_x_finish,
            in_y_start,
            in_y_finish,
            in_filepath);
        
        p_sec_data->n_tiles++;
        
        section_found[cur_section_idx] = 1;
        
    }
    
    int n_sections = 0;
    int zero_idx = 0;
    for (int sec_id = 0; sec_id < MAX_SECTIONS; sec_id++) {
        if (section_found[sec_id] == 1) {
            n_sections++;
        } else {
            zero_idx = sec_id;
            break;
        }
    }
    
    for (int sec_id = zero_idx; sec_id < MAX_SECTIONS; sec_id++) {
        ASSERT(section_found[sec_id] == 0);
    }
    
    ASSERT(n_sections == p_align_data->n_sections);
    
    //TRACE_2("-------------------------\n");
    //TRACE_2("n_sections: %d\n", p_align_data->n_sections);
    //TRACE_2("-------------------------\n");
    
    //TRACE_1("finish\n");
}

int protobuf_to_struct(align_data_t *p_tile_data) {
  // TODO does not deal with the mesh triangles
  AlignData align_data;
  // Read the existing address book.
  std::fstream input(p_tile_data->input_filepath, std::ios::in | std::ios::binary);
  if (!align_data.ParseFromIstream(&input)) {
    std::cerr << "Failed to parse protocal buffer." << std::endl;
    return 0;
  }
  // first deeal with AlignData level
  if (align_data.has_mode()) {
    p_tile_data->mode = align_data.mode();
  }
  if (align_data.has_work_dirpath()) {
    strcpy(p_tile_data->work_dirpath, align_data.work_dirpath().c_str());
  }
  if (align_data.has_output_dirpath()) {
    strcpy(p_tile_data->output_dirpath , align_data.output_dirpath().c_str());
  }
  if (align_data.has_base_section()) {
    p_tile_data->base_section = align_data.base_section();
  }
  if (align_data.has_n_sections()) {
    p_tile_data->n_sections = align_data.n_sections();
  }
  if (align_data.has_do_subvolume()) {
    p_tile_data->do_subvolume = align_data.do_subvolume();
    p_tile_data->min_x = align_data.min_x();
    p_tile_data->min_y = align_data.min_y();
    p_tile_data->max_x = align_data.max_x();
    p_tile_data->max_y = align_data.max_y();
  }

  // then do the section data
  for (int i = p_tile_data->base_section; i < (p_tile_data->base_section + p_tile_data->n_sections); i++) {
       

    SectionData section_data = align_data.sec_data(i);
    section_data_t *p_sec_data = &(p_tile_data->sec_data[i - p_tile_data->base_section]);
    p_sec_data->section_id = section_data.section_id();
    p_sec_data->n_tiles = 0;
    if (section_data.has_out_d1()) {
      p_sec_data->out_d1 = section_data.out_d1();
    }
    if (section_data.has_out_d2()) {
      p_sec_data->out_d2 = section_data.out_d2();
    }


    //p_out
    if (section_data.has_p_out()) {
      p_sec_data->p_out = new cv::Mat(section_data.p_out().rows(), section_data.p_out().cols(), CV_8UC1);
      for (int row = 0; row < section_data.p_out().rows(); row++) {
        for (int col = 0; col < section_data.p_out().cols(); col++) {
          p_sec_data->p_out->at<int>(row, col) = section_data.p_out().data(row*section_data.p_out().cols() + col);
        }
      }

    }

    //p_kps
    if (section_data.p_kps_size() > 0) {
      p_sec_data->p_kps = new std::vector<cv::KeyPoint>();
      for (int kp = 0; kp < section_data.p_kps_size(); kp++) {
        cv::KeyPoint key_point = cv::KeyPoint(section_data.p_kps(kp).x(), section_data.p_kps(kp).y(),
                                              section_data.p_kps(kp).size(), section_data.p_kps(kp).angle(),
                                              section_data.p_kps(kp).response(), section_data.p_kps(kp).octave(),
                                              section_data.p_kps(kp).class_id());
         p_sec_data->p_kps->push_back(key_point);
      }

    }

    for (int j = 0; j < section_data.tiles_size(); j++) {
      TileData tile_data = section_data.tiles(j);
      if (p_tile_data->do_subvolume) {
        float MIN_X =  p_tile_data->min_x;
        float MIN_Y =  p_tile_data->min_y;
        float MAX_X =  p_tile_data->max_x;
        float MAX_Y =  p_tile_data->max_y;
        if (tile_data.x_start() < MIN_X ||
          tile_data.y_start() < MIN_Y ||
          tile_data.x_finish() > MAX_X ||
          tile_data.y_finish() > MAX_Y) {
            continue;
        }
      }

      tile_data_t *p_cur_tile = &(p_sec_data->tiles[p_sec_data->n_tiles]);
      p_sec_data->n_tiles++;
      char in_filepath[MAX_FILEPATH];
      strcpy(in_filepath, tile_data.tile_filepath().c_str());
      init_tile(
            p_cur_tile,
            tile_data.section_id(),
            tile_data.tile_mfov(),
            tile_data.tile_index(),
            tile_data.x_start(),
            tile_data.x_finish(),
            tile_data.y_start(),
            tile_data.y_finish(),
            in_filepath);

      if (tile_data.has_p_image()) {
        p_cur_tile->p_image = new cv::Mat(tile_data.p_image().rows(), tile_data.p_image().cols(), CV_8UC1);
        for (int row = 0; row < tile_data.p_image().rows(); row++) {
          for (int col = 0; col < tile_data.p_image().cols(); col++) {
            p_cur_tile->p_image->at<int>(row, col) = tile_data.p_image().data(row*tile_data.p_image().cols() + col);
          }
        }

      }
      if (tile_data.p_kps_size() > 0) {
        p_cur_tile->p_kps = new std::vector<cv::KeyPoint>();
        for (int kp = 0; kp < tile_data.p_kps_size(); kp++) {
          cv::KeyPoint key_point = cv::KeyPoint(tile_data.p_kps(kp).x(), tile_data.p_kps(kp).y(),
                                                tile_data.p_kps(kp).size(), tile_data.p_kps(kp).angle(),
                                                tile_data.p_kps(kp).response(), tile_data.p_kps(kp).octave(),
                                                tile_data.p_kps(kp).class_id());
          p_cur_tile->p_kps->push_back(key_point);
        }

      }
      if (tile_data.has_p_kps_desc()) {
        p_cur_tile->p_kps_desc = new cv::Mat(tile_data.p_kps_desc().rows(), tile_data.p_kps_desc().cols(), CV_8UC1);
        for (int row = 0; row < tile_data.p_kps_desc().rows(); row++) {
          for (int col = 0; col < tile_data.p_kps_desc().cols(); col++) {
            p_cur_tile->p_kps_desc->at<int>(row, col) = tile_data.p_kps_desc().data(row*tile_data.p_kps_desc().cols() + col);
          }
        }

      }
      if (tile_data.p_kps_3d_size() > 0) {
        p_cur_tile->p_kps_3d = new std::vector<cv::KeyPoint>();
        for (int kp = 0; kp < tile_data.p_kps_3d_size(); kp++) {
          cv::KeyPoint key_point = cv::KeyPoint(tile_data.p_kps_3d(kp).x(), tile_data.p_kps_3d(kp).y(),
                                                tile_data.p_kps_3d(kp).size(), tile_data.p_kps_3d(kp).angle(),
                                                tile_data.p_kps_3d(kp).response(), tile_data.p_kps_3d(kp).octave(),
                                                tile_data.p_kps_3d(kp).class_id());
          p_cur_tile->p_kps_3d->push_back(key_point);
        }

      }
      if (tile_data.has_p_kps_desc_3d()) {
        p_cur_tile->p_kps_desc_3d = new cv::Mat(tile_data.p_kps_desc_3d().rows(), tile_data.p_kps_desc_3d().cols(), CV_8UC1);
        for (int row = 0; row < tile_data.p_kps_desc_3d().rows(); row++) {
          for (int col = 0; col < tile_data.p_kps_desc_3d().cols(); col++) {
            p_cur_tile->p_kps_desc_3d->at<int>(row, col) = tile_data.p_kps_desc_3d().data(row*tile_data.p_kps_desc_3d().cols() + col);
          }
        }

      }
      if (tile_data.has_ignore()) {
        bool *ignore = (bool *) malloc(sizeof(bool));
        *ignore = tile_data.ignore();
        p_cur_tile->ignore = ignore;
      }
      if (tile_data.has_a00()) {
        p_cur_tile->a00 = tile_data.a00();
      }
      if (tile_data.has_a10()) {
        p_cur_tile->a10 = tile_data.a10();
      }
      if (tile_data.has_a01()) {
        p_cur_tile->a10 = tile_data.a01();
      }
      if (tile_data.has_a11()) {
        p_cur_tile->a11 = tile_data.a11();
      }
      if (tile_data.has_level()) {
        p_cur_tile->level = tile_data.level();
      }
      if (tile_data.has_bad()) {
        p_cur_tile->bad = tile_data.bad();
      }
      if (tile_data.has_number_overlaps()) {
        p_cur_tile->number_overlaps = tile_data.number_overlaps();
      }
      if (tile_data.has_corralation_sum()) {
        p_cur_tile->corralation_sum = tile_data.corralation_sum();
      }
    }
  }
  return 1;

}

void struct_to_protobuf_matrix(cv::Mat *mat_struct, Matrix *mat_proto) {
  mat_proto->set_rows(mat_struct->rows);
  mat_proto->set_cols(mat_struct->cols);
  for (int row = 0; row < mat_proto->rows(); row++) {
    for (int col = 0; col < mat_proto->cols(); col++) {
       mat_proto->add_data(mat_struct->at<int>(row, col));
    }
  }
}

void struct_to_protobuf_key_point(const cv::KeyPoint kp_struct, KeyPoint *kp_proto) {
        kp_proto->set_x(kp_struct.pt.x);
        kp_proto->set_y(kp_struct.pt.y);
        kp_proto->set_size(kp_struct.size);
        kp_proto->set_angle(kp_struct.angle);
        kp_proto->set_response(kp_struct.response);
        kp_proto->set_octave(kp_struct.octave);
        kp_proto->set_class_id(kp_struct.class_id);
}

int struct_to_protobuf(align_data_t *p_tile_data) {
  // TODO does not deal with the mesh triangles
  int totel_tile_count = 0;
  AlignData align_data;
  // Read the existing address book.
  // first deeal with AlignData level
  align_data.set_mode(p_tile_data->mode);

  if (p_tile_data->work_dirpath) {
    align_data.set_work_dirpath(p_tile_data->work_dirpath);
  }
  if (p_tile_data->output_dirpath) {
    align_data.set_output_dirpath(p_tile_data->output_dirpath);
  }
  align_data.set_base_section(p_tile_data->base_section);
  
  align_data.set_n_sections(p_tile_data->n_sections);
  
  if (p_tile_data->do_subvolume) {
    align_data.set_do_subvolume(p_tile_data->do_subvolume);
    align_data.set_min_x(p_tile_data->min_x);
    align_data.set_min_y(p_tile_data->min_y);
    align_data.set_max_x(p_tile_data->max_x);
    align_data.set_max_y(p_tile_data->max_y);
  }

  // then do the section data
  for (int i = p_tile_data->base_section; i < (p_tile_data->base_section + p_tile_data->n_sections); i++) {
       

    SectionData *section_data = align_data.add_sec_data();
    section_data_t *p_sec_data = &(p_tile_data->sec_data[i - p_tile_data->base_section]);

    section_data->set_section_id(p_sec_data->section_id);
    section_data->set_n_tiles(p_sec_data->n_tiles);
    section_data->set_out_d1(p_sec_data->out_d1);
    section_data->set_out_d2(p_sec_data->out_d2);
    


    //p_out
    if (p_sec_data->p_out) {
      Matrix matrix;
      struct_to_protobuf_matrix(p_sec_data->p_out, &matrix);
      section_data->set_allocated_p_out(&matrix);
    }

    //p_kps
    if (p_sec_data->p_kps) {
      for (auto const& value: *p_sec_data->p_kps) {
        KeyPoint *key_point = section_data->add_p_kps();
        struct_to_protobuf_key_point(value, key_point);
      }

    }

    for (int j = 0; j < section_data->tiles_size(); j++) {
      TileData *tile_data = section_data->add_tiles();

      tile_data_t *p_cur_tile = &(p_sec_data->tiles[p_sec_data->n_tiles]);

      tile_data->set_section_id(p_cur_tile->section_id);
      tile_data->set_tile_index(p_cur_tile->index);
      tile_data->set_tile_mfov(p_cur_tile->mfov_id);
      tile_data->set_tile_id(totel_tile_count++);
      tile_data->set_x_start(p_cur_tile->x_start);
      tile_data->set_x_finish(p_cur_tile->x_finish);
      tile_data->set_y_start(p_cur_tile->y_start);
      tile_data->set_y_finish(p_cur_tile->y_finish);
      tile_data->set_tile_filepath(p_cur_tile->filepath);

      
      if (p_cur_tile->p_image) {
        Matrix matrix;
        struct_to_protobuf_matrix(p_cur_tile->p_image, &matrix);
        tile_data->set_allocated_p_image(&matrix);
      }

      if (p_cur_tile->p_kps) {
        for (auto const& value: *p_cur_tile->p_kps) {
          KeyPoint *key_point = tile_data->add_p_kps();
          struct_to_protobuf_key_point(value, key_point);
        }

      }

      if (p_cur_tile->p_kps_desc) {
        Matrix matrix;
        struct_to_protobuf_matrix(p_cur_tile->p_kps_desc, &matrix);
        tile_data->set_allocated_p_kps_desc(&matrix);
      }

      if (p_cur_tile->p_kps_3d) {
        for (auto const& value: *p_cur_tile->p_kps_3d) {
          KeyPoint *key_point = tile_data->add_p_kps_3d();
          struct_to_protobuf_key_point(value, key_point);
        }

      }

      if (p_cur_tile->p_kps_desc_3d) {
        Matrix matrix;
        struct_to_protobuf_matrix(p_cur_tile->p_kps_desc_3d, &matrix);
        tile_data->set_allocated_p_kps_desc_3d(&matrix);
      }
      if (p_cur_tile->ignore) {
        tile_data->set_ignore(p_cur_tile->ignore);
      }

      tile_data->set_a00(p_cur_tile->a00);
      tile_data->set_a10(p_cur_tile->a10);
      tile_data->set_a01(p_cur_tile->a10);
      tile_data->set_a11(p_cur_tile->a11);
      tile_data->set_level(p_cur_tile->level);
      tile_data->set_bad(p_cur_tile->bad);
      tile_data->set_number_overlaps(p_cur_tile->number_overlaps);
      tile_data->set_corralation_sum(p_cur_tile->corralation_sum);
    }
  }
  // Write the new address book back to disk.
  std::fstream output(p_tile_data->input_filepath, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!align_data.SerializeToOstream(&output)) {
    std::cerr << "Failed to write align data." << std::endl;
    return -1;
  }
  return 1;
  
}

void free_tiles(align_data_t *p_align_data) {
//NOTE(TFK): This was used when we weren't freeing tiles incrementally.
    //TRACE_1("read_tiles: start\n");
    
//    cilk_for (int sec_id = 0; sec_id < p_align_data->n_sections; sec_id++) {
//        section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
//        
//        cilk_for (int tile_id = 0; tile_id < p_sec_data->n_tiles; tile_id++) {
//            tile_data_t *p_tile = &(p_sec_data->tiles[tile_id]);
//            //p_tile->p_image->release();
///*
//            (*p_tile->p_image) = cv::imread(
//                p_tile->filepath, 
//                CV_LOAD_IMAGE_UNCHANGED);*/
//        }
//        
//    }
    
    //TRACE_1("read_tiles: finish\n");
    
}

void read_tiles(align_data_t *p_align_data) {

    //TRACE_1("read_tiles: start\n");
    
    cilk_for (int sec_id = 0; sec_id < p_align_data->n_sections; sec_id++) {
        section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
        
        cilk_for (int tile_id = 0; tile_id < p_sec_data->n_tiles; tile_id++) {
            tile_data_t *p_tile = &(p_sec_data->tiles[tile_id]);
            p_tile->tile_id = tile_id+1;
            
            //TRACE_1("  -- read[%d-%d]: %s\n", sec_id, tile_id, p_tile->filepath);
            
            //(*p_tile->p_image).create(4096, 4096, CV_8UC1);
            //(*p_tile->p_image).create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
            //
            //(*p_tile->p_image) = cv::imread(
            //    p_tile->filepath, 
            //    CV_LOAD_IMAGE_UNCHANGED);
            
            //ASSERT_MSG(
            //    !(p_tile->p_image->empty()),
            //    "cv::imread: failed to load image %s\n", p_tile->filepath);

            //ASSERT_MSG(
            //    p_tile->p_image->channels() == 1,
            //    "cv::imread: unexpected number of channels [%d] (expected: %d)\n", 
            //    p_tile->p_image->channels(), 1);
            
        }
        
    }
    
    //TRACE_1("read_tiles: finish\n");
    
}



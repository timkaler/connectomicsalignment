#include <sstream>

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

    std::map <std::string, int> int_vars;
    std::map <std::string, std::string> string_vars;

    ////TRACE_1("start\n");
    
    ASSERT(p_align_data->input_filepath != NULL);
    /*
    input file is of the form
    tilespec location
    string variables as a comma separated list
    integer variables as a comma separate list
    format string
    variables in order of format string
    */
    /*
    required variables are
    in_filepath,in_tile_id,in_section_id,in_mfov_id,in_index,in_x_start,in_x_finish,in_y_start,in_y_finish
    */
    fp = fopen(p_align_data->input_filepath, "rb");
    ASSERT_MSG(fp != NULL, "failed to open %s\n", p_align_data->input_filepath);
    char line[MAX_INPUT_BUF];
    fgets(line, MAX_INPUT_BUF, fp);
    size_t ln = strlen(line) - 1;
    if (line[ln] == '\n')
        line[ln] = '\0';
    std::string data_filepath = line;


    std::string string_variables;
    fgets(line, MAX_INPUT_BUF, fp);
    std::istringstream ss(line);
    std::string token;
    while(std::getline(ss, token, ',')) {
      if (!token.empty() && token[token.length()-1] == '\n') {
          token.erase(token.length()-1);
      }

      string_vars[token] = "";
    }
    std::string int_variables;
    fgets(line, MAX_INPUT_BUF, fp);
    std::istringstream ss2(line);
    while(std::getline(ss2, token, ',')) {
      if (!token.empty() && token[token.length()-1] == '\n') {
          token.erase(token.length()-1);
      }
      int_vars[token] = 0;
    }

    fgets(line, MAX_INPUT_BUF, fp);
    std::string format_string = line;

    std::map<std::string, int> variables;

    fgets(line, MAX_INPUT_BUF, fp);
    std::istringstream ss3(line);
    int variable_count = 0;
    while(std::getline(ss3, token, ',')) {
      if (!token.empty() && token[token.length()-1] == '\n') {
          token.erase(token.length()-1);
      }
      variables[token] = variable_count++;
    }



    printf("read in spec, file name = %s\n", data_filepath.c_str());
    fp = fopen(data_filepath.c_str(), "rb");
    std::cout << "open metadata file\n";
    n_objs_read = fscanf(fp, "total-tiles: %d\n", &n_tiles);
    ASSERT(n_objs_read == 1);
    
    int cur_section_idx = 0;
    auto it = format_string.find("\\n");
    while (it !=std::string::npos) {
      format_string.replace(it,2,1,'\n');
      it = format_string.find("\\n");
    }
    it = format_string.find("\\t");
    while (it != std::string::npos) {
      format_string.replace(it,2,1,'\t');
      it = format_string.find("\\t");
    }
    std::cout << format_string;

 
    for (int i = 0; i < n_tiles; i++) {
      char vars[25][MAX_INPUT_BUF];
        
        //printf("String input %s\n", str_input);;
        n_objs_read = fscanf(fp, format_string.c_str(),
                             vars[0],vars[1],vars[2],vars[3],vars[4],
                             vars[5],vars[6],vars[7],vars[8],vars[9],
                             vars[10],vars[11],vars[12],vars[13],vars[14],
                             vars[15],vars[16],vars[17],vars[18],vars[19],
                             vars[20],vars[21],vars[22],vars[23],vars[24]);

        ASSERT(n_objs_read == variable_count);

        // first get required variables
        char in_filepath[MAX_FILEPATH];
        strcpy(in_filepath, vars[variables["in_filepath"]]);

        int in_section_id;
        int in_x_start;
        int in_x_finish;
        int in_y_start;
        int in_y_finish;
        std::map<std::string, int> extra_vars; 

        for (auto it : int_vars) {
            // first do the required variables
            if ( it.first.compare("in_section_id") == 0) {
              in_section_id = atoi(vars[variables["in_section_id"]]);
            } else if ( it.first.compare("in_x_start")  == 0) {
              in_x_start = atoi(vars[variables["in_x_start"]]);
            } else if ( it.first.compare("in_x_finish") == 0 ) {
              in_x_finish = atoi(vars[variables["in_x_finish"]]);
            } else if ( it.first.compare("in_y_start") == 0) {
              in_y_start = atoi(vars[variables["in_y_start"]]);
            } else if ( it.first.compare("in_y_finish") == 0 ) {
              in_y_finish = atoi(vars[variables["in_y_finish"]]);
            } else {
              extra_vars[it.first] = atoi(vars[variables[it.first]]);
            }
        }
        // printf("%d, %d, %d, %d, %d, %s\n",in_section_id, in_x_start, in_x_finish, in_y_start, in_y_finish, in_filepath);
        

        //int in_tile_id = atoi(vars[variables["in_tile_id"]]);
        //int in_mfov_id = atoi(vars[variables["in_mfov_id"]]);
        //int in_index = atoi(vars[variables["in_index"]]);

        // this is due to the fact the input data starts its numbering with 1 for sections
        in_section_id --;

        
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
        // printf("Init tile with sindex %d tindex %d\n", cur_section_idx, p_sec_data->n_tiles); 
        init_tile(
            p_cur_tile,
            cur_section_idx,
            in_x_start,
            in_x_finish,
            in_y_start,
            in_y_finish,
            in_filepath,
            extra_vars);
        
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



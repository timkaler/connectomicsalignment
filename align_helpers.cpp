
std::string padTo(std::string str, const size_t num, const char paddingChar = '0')
{
    if(num > str.size())
        str.insert(0, num - str.size(), paddingChar);
    return str;
}

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
        
        n_objs_read = fscanf(fp, "%s [%d]\n", str_input, &in_tile_id);
        ASSERT(n_objs_read == 2);
        ASSERT(0 == strcmp(str_input, magic_str_tile_start));
        ASSERT(in_tile_id == i);
        
        n_objs_read = fscanf(fp, "\t%s %d\n", str_input, &in_section_id);
        ASSERT(n_objs_read == 2);
        ASSERT(0 == strcmp(str_input, magic_str_tile_section));
        ASSERT(in_section_id >= 1);
        in_section_id--;
        ASSERT(in_section_id >= 0);
        ASSERT(in_section_id < MAX_SECTIONS);
        
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
        
        cur_section_idx = in_section_id - p_align_data->base_section;
        
        //TRACE_2("\nread tile\n");
        //TRACE_2("   section_id : %d\n", in_section_id);
        //TRACE_2("   mfov_id    : %d\n", in_mfov_id);
        //TRACE_2("   index      : %d\n", in_index);
        //TRACE_2("   bbox       : [%d-%d, %d-%d]\n", in_x_start, in_x_finish, in_y_start, in_y_finish);
        //TRACE_2("   filepath   : %s\n", in_filepath);
        
        p_sec_data = &(p_align_data->sec_data[cur_section_idx]);
        p_cur_tile = &(p_sec_data->tiles[p_sec_data->n_tiles]);
        
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

void free_tiles(align_data_t *p_align_data) {

    //TRACE_1("read_tiles: start\n");
    
    cilk_for (int sec_id = 0; sec_id < p_align_data->n_sections; sec_id++) {
        section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
        
        cilk_for (int tile_id = 0; tile_id < p_sec_data->n_tiles; tile_id++) {
            tile_data_t *p_tile = &(p_sec_data->tiles[tile_id]);
            //p_tile->p_image->release();
/*
            (*p_tile->p_image) = cv::imread(
                p_tile->filepath, 
                CV_LOAD_IMAGE_UNCHANGED);*/
        }
        
    }
    
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
            //(*p_tile->p_image).create(3128, 2724, CV_8UC1);
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



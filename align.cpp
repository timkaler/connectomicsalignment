/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include "common.h"
#include "align.h"
#include "fasttime.h"
//#include "ezsift/ezsift.h"

#include <set>
#include <mutex>
#include <cilk/cilk.h>
#include "othersift.cpp"

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
// DEFINES
/////////////////////////////////////////////////////////////////////////////////////////
 
/////////////////////////////////////////////////////////////////////////////////////////
// TYPES
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// GLOBALS
/////////////////////////////////////////////////////////////////////////////////////////

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
    
    TRACE_1("start\n");
    
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
        
        TRACE_2("\nread tile\n");
        TRACE_2("   section_id : %d\n", in_section_id);
        TRACE_2("   mfov_id    : %d\n", in_mfov_id);
        TRACE_2("   index      : %d\n", in_index);
        TRACE_2("   bbox       : [%d-%d, %d-%d]\n", in_x_start, in_x_finish, in_y_start, in_y_finish);
        TRACE_2("   filepath   : %s\n", in_filepath);
        
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
    
    TRACE_2("-------------------------\n");
    TRACE_2("n_sections: %d\n", p_align_data->n_sections);
    TRACE_2("-------------------------\n");
    
    TRACE_1("finish\n");
}
void read_tiles(align_data_t *p_align_data) {

    TRACE_1("read_tiles: start\n");
    
    cilk_for (int sec_id = 0; sec_id < p_align_data->n_sections; sec_id++) {
        section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
        
        cilk_for (int tile_id = 0; tile_id < p_sec_data->n_tiles; tile_id++) {
            tile_data_t *p_tile = &(p_sec_data->tiles[tile_id]);
            
            TRACE_1("  -- read[%d-%d]: %s\n", sec_id, tile_id, p_tile->filepath);
            
            //(*p_tile->p_image).create(4096, 4096, CV_8UC1);
            (*p_tile->p_image).create(3128, 2724, CV_8UC1);
            
            (*p_tile->p_image) = cv::imread(
                p_tile->filepath, 
                CV_LOAD_IMAGE_UNCHANGED);
            
            ASSERT_MSG(
                !(p_tile->p_image->empty()),
                "cv::imread: failed to load image %s\n", p_tile->filepath);

            ASSERT_MSG(
                p_tile->p_image->channels() == 1,
                "cv::imread: unexpected number of channels [%d] (expected: %d)\n", 
                p_tile->p_image->channels(), 1);
            
        }
        
    }
    
    TRACE_1("read_tiles: finish\n");
    
}

bool is_tiles_overlap(tile_data_t *p_tile_data_1, tile_data_t *p_tile_data_2) {

    TRACE_3("is_tiles_overlap: start\n");
    TRACE_3("  -- tile_1: %d\n", p_tile_data_1->index);
    TRACE_3("  -- tile_2: %d\n", p_tile_data_2->index);
    
    int x1_start = p_tile_data_1->x_start;
    int x1_finish = p_tile_data_1->x_finish;
    int y1_start = p_tile_data_1->y_start;
    int y1_finish = p_tile_data_1->y_finish;
    
    int x2_start = p_tile_data_2->x_start;
    int x2_finish = p_tile_data_2->x_finish;
    int y2_start = p_tile_data_2->y_start;
    int y2_finish = p_tile_data_2->y_finish;
    
    bool res = false;
    
    if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
        (y1_start < y2_finish) && (y1_finish > y2_start)) {
        res = true;
    }
    
    TRACE_3("is_tiles_overlap: finish\n");
    
    return res;
    
}

void SIFT_initialize()
{
	cv::xfeatures2d::generateBoxBlurExecutionPlan();
}

void compute_SIFT_parallel(align_data_t *p_align_data) {

    static double totalTime = 0;
    
    int imageProcessed = 0;
    
    TRACE_1("compute_SIFT_parallel: start\n");
    
    SIFT_initialize();
    
    for (int sec_id = 0; sec_id < p_align_data->n_sections; sec_id++) {
        section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
        
        cilk_for (int tile_id = 0; tile_id < p_sec_data->n_tiles; tile_id++) {
            tile_data_t *p_tile_data = &(p_sec_data->tiles[tile_id]);
            
            int rows = p_tile_data->p_image->rows;
            int cols = p_tile_data->p_image->cols;
            
            /*TRACE_1("  -- tile[%d-%d]: detectAndCompute (%d, %d)\n", 
                sec_id, 
                tile_id,
                rows, 
                cols);*/

            // Define the SIFT features that we're going to compute.
            cv::Ptr<cv::Feature2D> p_sift;
            // DEFAULT Settings. 
            //p_sift = cv::xfeatures2d::SIFT::create(
            //    0,
            //    //6,
            //    3,
            //    //0.08,
            //    0.04,
            //    //5,
            //    10,
            //    1.6);

             // FASTER Settings.
             p_sift = new cv::xfeatures2d::SIFT_Impl( 
//cv::xfeatures2d::SIFT::create(
                0,
                6,
                0.04,//0.08,
                //5,
                10,
                1.6);
            
            std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
            cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];
            
            ASSERT((rows % SIFT_D1_SHIFT) == 0);
            ASSERT((cols % SIFT_D2_SHIFT) == 0);
            
            int max_rows = rows / SIFT_D1_SHIFT;
            int max_cols = cols / SIFT_D2_SHIFT;
            int n_sub_images = max_rows * max_cols;
            
            cilk_for (int cur_d1 = 0; cur_d1 < rows; cur_d1 += SIFT_D1_SHIFT) {
                cilk_for (int cur_d2 = 0; cur_d2 < cols; cur_d2 += SIFT_D2_SHIFT) {
                   
                    // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT 
                    cv::Mat sub_im = (*p_tile_data->p_image)(cv::Rect(
                        cur_d2, cur_d1, SIFT_D2_SHIFT, SIFT_D1_SHIFT));
                    // Mask for subimage 
                    cv::Mat sum_im_mask = cv::Mat::ones(SIFT_D1_SHIFT, SIFT_D2_SHIFT, CV_8UC1); 
                    // Compute a subimage ID, refering to a tile within larger
                    //   2d image. 
                    int cur_d1_id = cur_d1 / SIFT_D1_SHIFT;
                    int cur_d2_id = cur_d2 / SIFT_D2_SHIFT;
                    int sub_im_id = cur_d1_id * max_cols + cur_d2_id;
                    // Detect the SIFT features within the subimage. 
			  fasttime_t tstart=gettime();
                    p_sift->detectAndCompute(
                        sub_im,
                        sum_im_mask,
                        v_kps[sub_im_id],
                        m_kps_desc[sub_im_id]);
                    fasttime_t tend=gettime();
	            totalTime += tdiff(tstart,tend); 
			  
                    for (size_t i = 0; i < v_kps[sub_im_id].size(); i++) {
                        v_kps[sub_im_id][i].pt.x += cur_d2;
                        v_kps[sub_im_id][i].pt.y += cur_d1;
                    }
                }
            }
            
            for (int i = 0; i < n_sub_images; i++) {
                for (size_t j = 0; j < v_kps[i].size(); j++) {
                    (*p_tile_data->p_kps).push_back(v_kps[i][j]);
                }
            }

            cv::vconcat( m_kps_desc, n_sub_images, *(p_tile_data->p_kps_desc));


            #ifndef SKIPOUTPUT
            // NOTE(TFK): Begin HDF5 preparation 
            std::vector<float> locations;
            std::vector<float> octaves;
            std::vector<float> responses;
            std::vector<float> sizes;
            int NUM_KEYPOINTS = p_tile_data->p_kps->size();
            for (int i = 0; i < NUM_KEYPOINTS; i++) {
              locations.push_back((*(p_tile_data->p_kps))[i].pt.x);
              locations.push_back((*(p_tile_data->p_kps))[i].pt.y);
              octaves.push_back((*(p_tile_data->p_kps))[i].octave);
              responses.push_back((*(p_tile_data->p_kps))[i].response);
              sizes.push_back((*(p_tile_data->p_kps))[i].size);
            }
            std::string filepath = std::string(p_tile_data->filepath);
            std::string timagename = filepath.substr(filepath.find_last_of("/")+1);
            std::string real_section_id = timagename.substr(0,timagename.find("_"));
            std::string image_path = std::string(p_align_data->output_dirpath) + "/sifts/W01_Sec"+real_section_id+"/" + padTo(std::to_string(p_tile_data->mfov_id),6);
            system((std::string("mkdir -p ") + image_path).c_str());
            printf("timagename is %s, real_section_id %s\n", timagename.c_str(), real_section_id.c_str());
            image_path = image_path + std::string("/W01_Sec")+real_section_id + std::string("_sifts_")+timagename.substr(0,timagename.find_last_of(".")) + std::string(".hdf5");
          
            printf("TFK DEBUG: the image name is %s\n", image_path.c_str());


            create_sift_hdf5(image_path.c_str(), p_tile_data->p_kps->size(),sizes,responses, octaves,
                locations, p_tile_data->filepath);
            cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(cv::String(image_path.c_str()));
            int offset1[2] = {0,0};
            int offset2[2] = {(int)p_tile_data->p_kps->size(), 128};
            h5io->dswrite(*(p_tile_data->p_kps_desc), cv::String("descs"), &(offset1[0]), &(offset2[0]));
            h5io->close(); 
            // NOTE(TFK): End HDF5 preparation.
            #endif
            TRACE_1("    -- n_kps      : %lu\n", p_tile_data->p_kps->size());
            TRACE_1("    -- n_kps_desc : %d %d\n", p_tile_data->p_kps_desc->rows, p_tile_data->p_kps_desc->cols);

            #ifdef LOGIMAGES
            LOG_KPS(p_tile_data);
            #endif

	/*
	imageProcessed++;
	if (imageProcessed == 10) 
	{
		printf("first 10 images processed, net processing time = %.6lf\n", totalTime);
		exit(0);
	}
	*/
        }
    }
    
    printf("net processing time = %.6lf\n", totalTime);
    
    TRACE_1("compute_SIFT_parallel: finish\n");
}

 

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////
void align_execute(align_data_t *p_align_data)
{    
    TIMER_VAR(t_timer);
    TIMER_VAR(timer);
    
    START_TIMER(&t_timer);
            
    read_input(p_align_data);
    
    if (p_align_data->mode == MODE_COMPUTE_KPS_AND_MATCH) {
        START_TIMER(&timer);
        read_tiles(p_align_data);
        STOP_TIMER(&timer, "read_tiles time:");
        
        START_TIMER(&timer);
        compute_SIFT_parallel(p_align_data);
        STOP_TIMER(&timer, "compute_SIFT time:");
    } 
    
    STOP_TIMER(&t_timer, "t_total-time:");
         
}

// Copyright 2016 - Supertech Research Group
//   Tim Kaler, Tao B. Schardl, Haoran Xu, Charles E. Leiserson, Alex Matveev.


////////////////////////////////////////////////////////////////////////////////
// INCLUDES
////////////////////////////////////////////////////////////////////////////////
#include <cilk/cilk.h>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "./common.h"
#include "./align.h"
#include "./match.h"
#include "./fasttime.h"
#include "./simple_mutex.h"
//#include "./sift.config.h"
#include "./othersift2.cpp"

// Helper functions
#include "align_helpers.cpp"

void SIFT_initialize() {
  generateBoxBlurExecutionPlan();
}

void compute_SIFT_parallel(align_data_t *p_align_data) {
  static double totalTime = 0;

  //TRACE_1("compute_SIFT_parallel: start\n");

  SIFT_initialize();

  std::set<std::string> created_paths;
  simple_mutex_t created_paths_lock;
  simple_mutex_init(&created_paths_lock);
  #pragma cilk grainsize=1
  cilk_for (int sec_id = 0; sec_id < p_align_data->n_sections; sec_id++) {
    section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
    cilk_for (int tile_id = 0; tile_id < p_sec_data->n_tiles; tile_id++) {
      tile_data_t *p_tile_data = &(p_sec_data->tiles[tile_id]);


      (*p_tile_data->p_image).create(3128, 2724, CV_8UC1);
      (*p_tile_data->p_image) = cv::imread(
          p_tile_data->filepath, 
          CV_LOAD_IMAGE_UNCHANGED);




      int rows = p_tile_data->p_image->rows;
      int cols = p_tile_data->p_image->cols;

      ASSERT((rows % SIFT_D1_SHIFT) == 0);
      ASSERT((cols % SIFT_D2_SHIFT) == 0);
      cv::Ptr<cv::Feature2D> p_sift;



      std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
      cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];

      int n_sub_images;
      if ((p_tile_data->tile_id > MFOV_BOUNDARY_THRESH)) {

      p_sift = new cv::xfeatures2d::SIFT_Impl(
                0,  // num_features --- unsupported.
                6,  // number of octaves
                0.04,  // contrast threshold.
                5,  // edge threshold.
                1.6);  // sigma.

        // THEN: This tile is on the boundary, we need to compute SIFT features
        // on the entire section.
        int max_rows = rows / SIFT_D1_SHIFT;
        int max_cols = cols / SIFT_D2_SHIFT;
        n_sub_images = max_rows * max_cols;

        cilk_for (int cur_d1 = 0; cur_d1 < rows; cur_d1 += SIFT_D1_SHIFT) {
          cilk_for (int cur_d2 = 0; cur_d2 < cols; cur_d2 += SIFT_D2_SHIFT) {
            // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
            cv::Mat sub_im = (*p_tile_data->p_image)(cv::Rect(cur_d2, cur_d1,
                SIFT_D2_SHIFT, SIFT_D1_SHIFT));

            // Mask for subimage
            cv::Mat sum_im_mask = cv::Mat::ones(SIFT_D1_SHIFT, SIFT_D2_SHIFT,
                CV_8UC1);

            // Compute a subimage ID, refering to a tile within larger
            //   2d image.
            int cur_d1_id = cur_d1 / SIFT_D1_SHIFT;
            int cur_d2_id = cur_d2 / SIFT_D2_SHIFT;
            int sub_im_id = cur_d1_id * max_cols + cur_d2_id;

            // Detect the SIFT features within the subimage.
            fasttime_t tstart = gettime();
            p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
                m_kps_desc[sub_im_id]);

            fasttime_t tend = gettime();
            totalTime += tdiff(tstart, tend);

            for (size_t i = 0; i < v_kps[sub_im_id].size(); i++) {
              v_kps[sub_im_id][i].pt.x += cur_d2;
              v_kps[sub_im_id][i].pt.y += cur_d1;
            }
          }
        }
        // Regardless of whether we were on or off MFOV boundary, we concat
        //   the keypoints and their descriptors here.
        for (int i = 0; i < n_sub_images; i++) {
            for (int j = 0; j < v_kps[i].size(); j++) {
                (*p_tile_data->p_kps).push_back(v_kps[i][j]);
            }
        }

      } else {

      p_sift = new cv::xfeatures2d::SIFT_Impl(
                0,  // num_features --- unsupported.
                6,  // number of octaves
                0.04,  // contrast threshold.
                5,  // edge threshold.
                1.6);  // sigma.

        // ELSE THEN: This tile is in the interior of the MFOV. Only need to
        //     compute features along the boundary.
        n_sub_images = 4;

        // BEGIN TOP SLICE
        {
          // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
          cv::Mat sub_im = (*p_tile_data->p_image)(cv::Rect(
              0, 0, 3128, OVERLAP_2D));

          // Mask for subimage
          cv::Mat sum_im_mask = cv::Mat::ones(OVERLAP_2D, 3128, CV_8UC1);
          int sub_im_id = 0;

          // Detect the SIFT features within the subimage.
          fasttime_t tstart = gettime();
          p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
              m_kps_desc[sub_im_id]);
          fasttime_t tend = gettime();
          totalTime += tdiff(tstart, tend);
          for (size_t i = 0; i < v_kps[sub_im_id].size(); i++) {
              v_kps[sub_im_id][i].pt.x += 0;  // cur_d2;
              v_kps[sub_im_id][i].pt.y += 0;  // cur_d1;
          }
        }
        // END TOP SLICE

        // BEGIN LEFT SLICE
        {
          // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
          cv::Mat sub_im = (*p_tile_data->p_image)(cv::Rect(
              0, OVERLAP_2D, OVERLAP_2D, 2724-OVERLAP_2D));

          // Mask for subimage
          cv::Mat sum_im_mask = cv::Mat::ones(2724-OVERLAP_2D, OVERLAP_2D,
              CV_8UC1);
          int sub_im_id = 1;
          // Detect the SIFT features within the subimage.
          fasttime_t tstart = gettime();
          p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
             m_kps_desc[sub_im_id]);
          fasttime_t tend = gettime();
          totalTime += tdiff(tstart, tend);
          for (int i = 0; i < v_kps[sub_im_id].size(); i++) {
              v_kps[sub_im_id][i].pt.x += 0;  // cur_d2;
              v_kps[sub_im_id][i].pt.y += OVERLAP_2D;  // cur_d1;
          }
        }
        // END LEFT SLICE

        // BEGIN RIGHT SLICE
        {
          // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
          cv::Mat sub_im = (*p_tile_data->p_image)(cv::Rect(
              3128-OVERLAP_2D, OVERLAP_2D, OVERLAP_2D, 2724-OVERLAP_2D));

          // Mask for subimage
          cv::Mat sum_im_mask = cv::Mat::ones(2724-OVERLAP_2D, OVERLAP_2D,
              CV_8UC1);
          int sub_im_id = 2;
          // Detect the SIFT features within the subimage.
          fasttime_t tstart = gettime();

          p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
              m_kps_desc[sub_im_id]);
          fasttime_t tend = gettime();
          totalTime += tdiff(tstart, tend);
          for (int i = 0; i < v_kps[sub_im_id].size(); i++) {
              v_kps[sub_im_id][i].pt.x += 3128-OVERLAP_2D;  // cur_d2;
              v_kps[sub_im_id][i].pt.y += OVERLAP_2D;  // cur_d1;
          }
        }
        // END RIGHT SLICE

        // BEGIN BOTTOM SLICE
        {
          // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
          cv::Mat sub_im = (*p_tile_data->p_image)(cv::Rect(
              OVERLAP_2D, 2724-OVERLAP_2D, 3128-OVERLAP_2D, OVERLAP_2D));

          // Mask for subimage
          cv::Mat sum_im_mask = cv::Mat::ones(OVERLAP_2D, 3128-OVERLAP_2D,
              CV_8UC1);
          // Compute a subimage ID, refering to a tile within larger
          //   2d image.
          int sub_im_id = 3;
          // Detect the SIFT features within the subimage.
          fasttime_t tstart = gettime();

          p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
              m_kps_desc[sub_im_id]);
          fasttime_t tend = gettime();
          totalTime += tdiff(tstart, tend);
          for (int i = 0; i < v_kps[sub_im_id].size(); i++) {
              v_kps[sub_im_id][i].pt.x += OVERLAP_2D;  // cur_d2;
              v_kps[sub_im_id][i].pt.y += (2724-OVERLAP_2D);  // cur_d1;
          }
        }
        // END BOTTOM SLICE

        // Regardless of whether we were on or off MFOV boundary, we concat
        //   the keypoints and their descriptors here.
        for (int i = 0; i < n_sub_images; i++) {
            for (size_t j = 0; j < v_kps[i].size(); j++) {
                (*p_tile_data->p_kps).push_back(v_kps[i][j]);
            }
        }
      }

      cv::vconcat(m_kps_desc, n_sub_images, *(p_tile_data->p_kps_desc));

      int NUM_KEYPOINTS = p_tile_data->p_kps->size();
      if (NUM_KEYPOINTS > 0) {
        #ifndef SKIPHDF5
        // NOTE(TFK): Begin HDF5 preparation
        std::vector<float> locations;
        std::vector<float> octaves;
        std::vector<float> responses;
        std::vector<float> sizes;
        for (int i = 0; i < NUM_KEYPOINTS; i++) {
          locations.push_back((*(p_tile_data->p_kps))[i].pt.x);
          locations.push_back((*(p_tile_data->p_kps))[i].pt.y);
          octaves.push_back((*(p_tile_data->p_kps))[i].octave);
          responses.push_back((*(p_tile_data->p_kps))[i].response);
          sizes.push_back((*(p_tile_data->p_kps))[i].size);
        }
        std::string filepath = std::string(p_tile_data->filepath);
        std::string timagename = filepath.substr(filepath.find_last_of("/")+1);
        std::string real_section_id =
            timagename.substr(0, timagename.find("_"));
        std::string image_path = std::string(p_align_data->output_dirpath) +
            "/sifts/W01_Sec"+real_section_id+"/" +
            padTo(std::to_string(p_tile_data->mfov_id), 6);

        simple_acquire(&created_paths_lock);
        if (created_paths.find(image_path) == created_paths.end()) {
          system((std::string("mkdir -p ") + image_path).c_str());
          created_paths.insert(image_path);
        }
        simple_release(&created_paths_lock);

        printf("timagename is %s, real_section_id %s\n", timagename.c_str(),
           real_section_id.c_str());
        image_path = image_path + std::string("/W01_Sec")+real_section_id +
            std::string("_sifts_") +
            timagename.substr(0, timagename.find_last_of(".")) +
            std::string(".hdf5");

        //printf("creating hdf5 with size %d\n", sizes.size());
        printf("creating hdf5 with path %s\n", image_path.c_str());
        create_sift_hdf5(image_path.c_str(), p_tile_data->p_kps->size(), sizes,
            responses, octaves, locations, p_tile_data->filepath);
        cv::Ptr<cv::hdf::HDF5> h5io =
            cv::hdf::open(cv::String(image_path.c_str()));
        int offset1[2] = {0, 0};
        int offset2[2] = {(int)p_tile_data->p_kps->size(), 128};
        h5io->dswrite(*(p_tile_data->p_kps_desc), cv::String("descs"),
            &(offset1[0]), &(offset2[0]));
        h5io->close();
        // NOTE(TFK): End HDF5 preparation.
        #endif
        //TRACE_1("    -- n_kps      : %lu\n", p_tile_data->p_kps->size());
        //TRACE_1("    -- n_kps_desc : %d %d\n", p_tile_data->p_kps_desc->rows,
        //    p_tile_data->p_kps_desc->cols);

        #ifdef LOGIMAGES
        LOG_KPS(p_tile_data);
        #endif
        } else {
           printf("WARNING::: NO KEYPOINTS FOUND!\n");
        }
        (*p_tile_data->p_image).release();
      }

     
      compute_tile_matches(p_align_data, sec_id);

    }
  //TRACE_1("compute_SIFT_parallel: finish\n");
}

////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
void align_execute(align_data_t *p_align_data) {
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
        free_tiles(p_align_data);
    START_TIMER(&timer);
    //compute_tile_matches(p_align_data);
    STOP_TIMER(&timer, "compute_tile_matches time:");
    STOP_TIMER(&t_timer, "t_total-time:");
}

// function for debug purpose only
// #include "tests.cpp"
// void testcv()
// {
//   //test_minfilter();
// }

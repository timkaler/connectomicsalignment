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
#include <algorithm>

float CONTRAST_THRESH = 0.04;
float CONTRAST_THRESH_3D = 0.04;
float EDGE_THRESH_3D = 5.0;
float EDGE_THRESH_2D = 5.0;
#include "./common.h"
#include "./align.h"
#include "./match.h"
#include "./fasttime.h"
#include "./simple_mutex.h"
//#include "./sift.config.h"
//#include "./othersift.cpp"
#include "./othersift2.cpp"

// Helper functions
#include "align_helpers.cpp"
#include "cilk_tools/Graph.h"

void SIFT_initialize() {
  generateBoxBlurExecutionPlan();
}



bool section_data_exists(int section, align_data_t* p_align_data) {
  std::string filename = std::string("prefix_")+std::to_string(section+p_align_data->base_section+1);
  cv::FileStorage fs(filename+std::string("_3d_keypoints.yml.gz"), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  } else {
    return true;
  }
}

void store_3d_matches(int section, align_data_t* p_align_data) {
  std::string filename = std::string("prefix_")+std::to_string(section+p_align_data->base_section+1);
  cv::FileStorage fs(filename+std::string("_3d_keypoints.yml.gz"), cv::FileStorage::WRITE);
  for (int i = 0; i < p_align_data->sec_data[section].n_tiles; i++) {
    cv::write(fs, "keypoints_"+std::to_string(i), *(p_align_data->sec_data[section].tiles[i].p_kps_3d));
    cv::write(fs, "descriptors_" + std::to_string(i), *(p_align_data->sec_data[section].tiles[i].p_kps_desc_3d));
  }
  fs.release();
}

void read_3d_matches(int section, align_data_t* p_align_data) {
  std::string filename = std::string("prefix_")+std::to_string(section+p_align_data->base_section+1);
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

  std::string filename = std::string("prefix_")+std::to_string(section+p_align_data->base_section+1);
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
  std::string filename = std::string("prefix_")+std::to_string(section+p_align_data->base_section+1);
  cv::FileStorage fs(filename+std::string("_2d_matches.yml.gz"), cv::FileStorage::READ);
  for (int i = 0; i < graph->num_vertices(); i++) {
    int edge_size;
    fs["num_edges_"+std::to_string(i)] >> edge_size; 
    std::vector<edata> edge_data; 
    for (int j = 0; j < edge_size; j++) {
      edata edge;
      std::vector<cv::Point2f>* v_points = new std::vector<cv::Point2f>();
      std::vector<cv::Point2f>* n_points = new std::vector<cv::Point2f>();
      fs["neighbor_id_"+std::to_string(i) + "_" + std::to_string(j)] >> edge.neighbor_id;
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


void compute_SIFT_parallel_3d(align_data_t *p_align_data) {
  static double totalTime = 0;

  //TRACE_1("compute_SIFT_parallel: start\n");

  SIFT_initialize();

  std::set<std::string> created_paths;
  simple_mutex_t created_paths_lock;
  simple_mutex_init(&created_paths_lock);
  for (int sec_id = 0; sec_id < p_align_data->n_sections; sec_id++) {
    section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
    cilk_for (int tile_id = 0; tile_id < p_sec_data->n_tiles; tile_id++) {
      tile_data_t *p_tile_data = &(p_sec_data->tiles[tile_id]);


      (*p_tile_data->p_image).create(3128, 2724, CV_8UC1);
      (*p_tile_data->p_image) = cv::imread(
          p_tile_data->filepath, 
          CV_LOAD_IMAGE_UNCHANGED);

      int rows = p_tile_data->p_image->rows;
      int cols = p_tile_data->p_image->cols;
      ASSERT((rows % SIFT_D1_SHIFT_3D) == 0);
      ASSERT((cols % SIFT_D2_SHIFT_3D) == 0);
      cv::Ptr<cv::Feature2D> p_sift;
      std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
      cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];
      int n_sub_images;

      p_sift = new cv::xfeatures2d::SIFT_Impl(
                32,  // num_features --- unsupported.
                12,  // number of octaves
                CONTRAST_THRESH_3D,  // contrast threshold.
                EDGE_THRESH_3D,  // edge threshold.
                1.6);  // sigma.

        int max_rows = rows / SIFT_D1_SHIFT_3D;
        int max_cols = cols / SIFT_D2_SHIFT_3D;
        n_sub_images = max_rows * max_cols;
            cv::Mat sub_im_mask = cv::Mat::ones(0,0,
                CV_8UC1);
            int sub_im_id = 0;
            // Detect the SIFT features within the subimage.
            fasttime_t tstart = gettime();
            p_sift->detectAndCompute((*p_tile_data->p_image), sub_im_mask, v_kps[sub_im_id],
                m_kps_desc[sub_im_id], false);

            fasttime_t tend = gettime();
            totalTime += tdiff(tstart, tend);
        // Regardless of whether we were on or off MFOV boundary, we concat
        //   the keypoints and their descriptors here.
        for (int i = 0; i < n_sub_images; i++) {
            for (int j = 0; j < v_kps[i].size(); j++) {
                (*p_tile_data->p_kps_3d).push_back(v_kps[i][j]);
            }
        }
  
      //cv::Mat m_kps_desc_filtered = m_kps_desc[0].clone();
      *(p_tile_data)->p_kps_desc_3d = m_kps_desc[0].clone();
      m_kps_desc[0].release();
      //cv::vconcat(m_kps_desc_filtered, n_sub_images, *(p_tile_data->p_kps_desc_3d));

      int NUM_KEYPOINTS = p_tile_data->p_kps_3d->size();
      //printf("The number of keypoints is %d\n", NUM_KEYPOINTS);
      if (NUM_KEYPOINTS > 0) {
        #ifndef SKIPHDF5
        // NOTE(TFK): Begin HDF5 preparation
        std::vector<float> locations;
        std::vector<float> octaves;
        std::vector<float> responses;
        std::vector<float> sizes;
        for (int i = 0; i < NUM_KEYPOINTS; i++) {
          locations.push_back((*(p_tile_data->p_kps_3d))[i].pt.x);
          locations.push_back((*(p_tile_data->p_kps_3d))[i].pt.y);
          octaves.push_back((*(p_tile_data->p_kps_3d))[i].octave);
          responses.push_back((*(p_tile_data->p_kps_3d))[i].response);
          sizes.push_back((*(p_tile_data->p_kps_3d))[i].size);
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
        create_sift_hdf5(image_path.c_str(), p_tile_data->p_kps_3d->size(), sizes,
            responses, octaves, locations, p_tile_data->filepath);
        cv::Ptr<cv::hdf::HDF5> h5io =
            cv::hdf::open(cv::String(image_path.c_str()));
        int offset1[2] = {0, 0};
        int offset2[2] = {(int)p_tile_data->p_kps_3d->size(), 128};
        h5io->dswrite(*(p_tile_data->p_kps_desc_3d), cv::String("descs"),
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
        #ifdef ALIGN3D
        cilk_sync;
        #endif
        (*p_tile_data->p_image).release();
      }
     
    }
  //TRACE_1("compute_SIFT_parallel: finish\n");
}

/*
  Get all the neighbors for all of the elements in the active set
*/
void get_neighbors(std::set<int>& /* OUTPUT*/ neighbor_set, std::set<int> active_set, section_data_t *p_sec_data){
  // loop over every element in the active set and then all of their neighbors
  for (auto active : active_set) {
    int indices_to_check[50];
    int indices_to_check_len = get_all_close_tiles(active, p_sec_data, indices_to_check);
    for (int i = 0; i < indices_to_check_len; i++) {
      // if any are not in the active set add them to the neighbor set
      if ( active_set.find(indices_to_check[i]) == active_set.end()) {
        neighbor_set.insert(indices_to_check[i]);
      }
    }
  }
}

/*
  This will get us an active set of tiles
  Hopefully we will do something smart here eventually 
    like a sliding window or the boundry of the expanding region
  but for now assume nothing
  it should work regardless of what this active set includes
  they do nt have to be next to each other or in any pattern
  not efficiency is really bad if they are not in some form of clump

  The active set is roughly a rectangle it is deined with a height, which is the number of tiles in the first demension it is
  and also by a number of tiles  in the set

  this function also cleans up the tile data when we no longer need it, it should do its best to not delete things we will need to calculate again
*/
// by row
void get_next_active_set_and_neighbor_set(std::set<int>& /* OUTOUT */ active_set, std::set<int>& /* OUTPUT */ neighbor_set,
                                          std::set<int> finished_set, section_data_t *p_sec_data, 
                                          std::set<int> /* MODIFIED */ known_set) {
  int total = p_sec_data->n_tiles;
  int tiles[total];
  int max_size_of_active_set = total;
  int height_of_active_set = 8;
  for (int i = 0; i < total; i++) {
    tiles[i] = i;
  }
  struct X_Sort {
    X_Sort(section_data_t *p_sec_data) { this->p_sec_data = p_sec_data; }
    bool operator () (int i, int j) {
      return p_sec_data->tiles[i].x_start < p_sec_data->tiles[j].x_start;
    }

    section_data_t *p_sec_data;
  };
  struct Y_Sort {
    Y_Sort(section_data_t *p_sec_data) { this->p_sec_data = p_sec_data; }
    bool operator () (int i, int j) {
      return p_sec_data->tiles[i].y_start < p_sec_data->tiles[j].y_start;
    }

    section_data_t *p_sec_data;
  };
  std::sort( tiles,tiles+total, X_Sort(p_sec_data));

  if (finished_set.size() == total) {
    // clean up the memory
    for (auto del : known_set) {
      tile_data_t *a_tile = &(p_sec_data->tiles[del]);
      a_tile->p_kps->clear();
      std::vector<cv::KeyPoint>().swap(*(a_tile->p_kps));
      ((a_tile->p_kps_desc))->release();
    }
    return;
  }
  // this section of code makes the active set all of the tiles, use this if you want to test the original implementation
  // it will be almost exactly the same as doing the original implementation
  if (false) {
    for (int i = 0; i < total; i++) {
      active_set.insert(i);
    }
    return;
  }
  int starting_tile = 0;
  for (int i = 0; i < total; i++) {
    if (finished_set.find(tiles[i]) == finished_set.end()) {
      starting_tile = i;
      break;
    }
  }
  int x_bound = p_sec_data->tiles[tiles[starting_tile]].x_start + height_of_active_set * (
              p_sec_data->tiles[tiles[starting_tile]].x_finish - p_sec_data->tiles[tiles[starting_tile]].x_start);
  int tiles_in_x[total];
  int num_pos = 0;
  for (int i = starting_tile; i < total; i++) {
    if (x_bound < p_sec_data->tiles[tiles[i]].x_start) {
      break;
    }
    if (finished_set.find(tiles[i]) == finished_set.end()) {
      tiles_in_x[num_pos]= tiles[i];
      num_pos++;
    }
  }
  std::sort(tiles_in_x,tiles_in_x+num_pos, Y_Sort(p_sec_data));
  for (int i = 0; (i < num_pos) && (i < max_size_of_active_set); i++) {
    active_set.insert(tiles_in_x[i]);
  }

  get_neighbors(neighbor_set, active_set, p_sec_data);

  // forget the things in know that we don't need anymore
  std::set<int> to_delete;
  for (auto known : known_set) {
    if ((active_set.find(known) == active_set.end()) && (neighbor_set.find(known) == neighbor_set.end())) {
      to_delete.insert(known);
    }
  }
  for (auto del : to_delete) {
    known_set.erase(del);
    tile_data_t *a_tile = &(p_sec_data->tiles[del]);
    a_tile->p_kps->clear();
    std::vector<cv::KeyPoint>().swap(*(a_tile->p_kps));
    ((a_tile->p_kps_desc))->release();
  }
  printf("active size = %d, neighbor size = %d, finished size = %d, known size() = %d\n",
    active_set.size(), neighbor_set.size(), finished_set.size(), known_set.size());


}


void compute_SIFT_parallel(align_data_t *p_align_data) {
  static double totalTime = 0;

  //TRACE_1("compute_SIFT_parallel: start\n");

  SIFT_initialize();

  // the graph_list
  std::vector<Graph<vdata, edata>* > graph_list;
  graph_list.resize(p_align_data->n_sections);

  std::set<std::string> created_paths;
  simple_mutex_t created_paths_lock;
  simple_mutex_init(&created_paths_lock);

  #ifdef MEMCHECK
  int max_known = 0;
  #endif

  //cilk_for (int sec_id_split = 0; sec_id_split < 2; sec_id_split++) {

  //int split = sec_id_split*p_align_data->n_sections/2;
  //int split_start = split*sec_id_split;
  //int split_end = p_align_data->n_sections;
  //if (sec_id_split == 0) {
  //  split_end = split;
  //}

  //for (int sec_id = split_start; sec_id < split_end/*p_align_data->n_sections*/; sec_id++) {
  cilk_for (int sec_id = 0; sec_id < p_align_data->n_sections; sec_id++) {
    section_data_t *p_sec_data = &(p_align_data->sec_data[sec_id]);
    std::set<int> active_set;
    std::set<int> finished_set;
    std::set<int> neighbor_set;
    std::set<int> known_set;
    get_next_active_set_and_neighbor_set(active_set, neighbor_set, finished_set, p_sec_data, known_set);

    // each section has its own graph
    Graph<vdata, edata>* graph;
    graph = new Graph<vdata, edata>();
    printf("Resizing the graph to be size %d\n", p_sec_data->n_tiles);
    graph->resize(p_sec_data->n_tiles);
    //graph_list.push_back(graph);
    graph_list[sec_id] = (graph);
    int work_count_total = 0;
    if (section_data_exists(sec_id, p_align_data)) {
      goto read_graph_from_file;
    }
    while (active_set.size() > 0) {
      std::set<int> active_and_neighbors;
      active_and_neighbors.insert(active_set.begin(), active_set.end());
      active_and_neighbors.insert(neighbor_set.begin(), neighbor_set.end());
      int active_and_neighbors_array[active_and_neighbors.size()];
      int tiles_to_get = 0;
      for (auto tile_id : active_and_neighbors) {
        if (known_set.find(tile_id) == known_set.end()) {
          work_count_total++;
          active_and_neighbors_array[tiles_to_get] = tile_id;
          tiles_to_get++;
        }
      }
      cilk_for (int i = 0; i < tiles_to_get; i++) {
        int tile_id = active_and_neighbors_array[i];
        if (known_set.find(tile_id) == known_set.end()) {
          #ifndef MEMCHECK
          tile_data_t *p_tile_data = &(p_sec_data->tiles[tile_id]);


          (*p_tile_data->p_image).create(3128, 2724, CV_8UC1);
          (*p_tile_data->p_image) = cv::imread(
              p_tile_data->filepath, 
              CV_LOAD_IMAGE_UNCHANGED);


#ifdef ALIGN3D
 cilk_spawn {
      int rows = p_tile_data->p_image->rows;
      int cols = p_tile_data->p_image->cols;
      ASSERT((rows % SIFT_D1_SHIFT_3D) == 0);
      ASSERT((cols % SIFT_D2_SHIFT_3D) == 0);
      cv::Ptr<cv::Feature2D> p_sift;
      std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
      cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];
      int n_sub_images;

      p_sift = new cv::xfeatures2d::SIFT_Impl(
                32,  // num_features --- unsupported.
                6,  // number of octaves
                CONTRAST_THRESH_3D,  // contrast threshold.
                EDGE_THRESH_3D,  // edge threshold.
                1.6*2);  // sigma.

        int max_rows = rows / SIFT_D1_SHIFT_3D;
        int max_cols = cols / SIFT_D2_SHIFT_3D;
        n_sub_images = max_rows * max_cols;
            cv::Mat sub_im_mask = cv::Mat::ones(0,0,
                CV_8UC1);
            int sub_im_id = 0;
            // Detect the SIFT features within the subimage.
            fasttime_t tstart = gettime();
            p_sift->detectAndCompute((*p_tile_data->p_image), sub_im_mask, v_kps[sub_im_id],
                m_kps_desc[sub_im_id], false);

            fasttime_t tend = gettime();
            totalTime += tdiff(tstart, tend);
        // Regardless of whether we were on or off MFOV boundary, we concat
        //   the keypoints and their descriptors here.
        for (int _i = 0; _i < n_sub_images; _i++) {
            for (int _j = 0; _j < v_kps[_i].size(); _j++) {
                (*p_tile_data->p_kps_3d).push_back(v_kps[_i][_j]);
            }
        }
  
      //cv::Mat m_kps_desc_filtered = m_kps_desc[0].clone();
      *(p_tile_data)->p_kps_desc_3d = m_kps_desc[0].clone();
}
#endif

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
                    4,  // num_features --- unsupported.
                    6,  // number of octaves
                    CONTRAST_THRESH,  // contrast threshold.
                    EDGE_THRESH_2D,  // edge threshold.
                    1.6);  // sigma.

            //cv::xfeatures2d::min_size = 4.0; 
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
                    4,  // num_features --- unsupported.
                    6,  // number of octaves
                    CONTRAST_THRESH,  // contrast threshold.
                    EDGE_THRESH_2D,  // edge threshold.
                    1.6);  // sigma.

            //cv::xfeatures2d::min_size = 4.0; 
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
          //printf("The number of keypoints is %d\n", NUM_KEYPOINTS);
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
          #endif
        }
      }
      #ifndef MEMCHECK
      compute_tile_matches_active_set(p_align_data, sec_id, active_set, graph);
      #endif
      #ifdef MEMCHECK
      if (active_set.size() + neighbor_set.size() > max_known) {
        max_known = active_set.size() + neighbor_set.size();
      }
      #endif
      finished_set.insert(active_set.begin(), active_set.end());
      known_set.clear();
      known_set.insert(active_set.begin(), active_set.end());
      known_set.insert(neighbor_set.begin(), neighbor_set.end());
      active_set.clear();
      neighbor_set.clear();
      get_next_active_set_and_neighbor_set(active_set, neighbor_set, finished_set, p_sec_data, known_set);
    }

    // Initialize data in the graph representation.
    printf("Size of the graph is %d\nThe average work per tile was %f\n", graph->num_vertices(), ((float) work_count_total)/ p_sec_data->n_tiles);
    #ifdef MEMCHECK
     printf("max size of active_set and neighbor_set = %d\n", max_known);
    #endif

    store_2d_graph(graph, sec_id, p_align_data);
    store_3d_matches(sec_id, p_align_data);

    goto skip_read_graph_from_file;
    read_graph_from_file:
       read_graph_from_file(graph, sec_id, p_align_data);
       read_3d_matches(sec_id, p_align_data);
    skip_read_graph_from_file:

    for (int i = 0; i < graph->num_vertices(); i++) {
      vdata* d = graph->getVertexData(i);
      _tile_data tdata = p_sec_data->tiles[i];
      d->vertex_id = i;
      d->mfov_id = tdata.mfov_id;
      d->tile_index = tdata.index;
      d->tile_id = i;
      d->start_x = tdata.x_start;
      d->end_x = tdata.x_finish;
      d->start_y = tdata.y_start;
      d->end_y = tdata.y_finish;
      d->offset_x = 0.0;
      d->offset_y = 0.0;
      d->iteration_count = 0;
      d->last_radius_value = 9.0;
      d->z = /*p_align_data->base_section + */sec_id;
      d->a00 = 1.0;
      d->a01 = 0.0;
      d->a10 = 0.0;
      d->a11 = 1.0;
      d->neighbor_grad_x = 0.0;
      d->neighbor_grad_y = 0.0;
      d->converged = 0;
    }
    graph->section_id = sec_id;
  }
  //}
  set_graph_list(graph_list, true);

  //for (int i = 0; i < graph_list.size(); i++) {
  //  store_2d_graph(graph_list[i], i, p_align_data);
  //}
  //for (int i = 0; i < p_align_data->n_sections; i++) {
  //  store_3d_matches(i, p_align_data);
  //}
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
        //#ifdef ALIGN3D
        //compute_SIFT_parallel_3d(p_align_data);
        //#endif
        compute_SIFT_parallel(p_align_data);
        STOP_TIMER(&timer, "compute_SIFT time:");
    }
        free_tiles(p_align_data);
    START_TIMER(&timer);

    

    // Before running the graph optimize code, set the graph_list by calling:
    // set_graph_list(graph_list, true) (from match.h) 
    // Runs the graph optimize code.
    //for (int i = 0; i < p_align_data->n_sections; i++) {
    //  store_3d_matches(i, p_align_data);
    //}

    compute_tile_matches(p_align_data, -1);



    //output_section_image(&(p_align_data->sec_data[0]), 0,0,40000,40000, "labeled_image0.tif");
    //output_section_image(&(p_align_data->sec_data[1]), 0,0,40000,40000, "labeled_image1.tif");

    STOP_TIMER(&timer, "compute_tile_matches time:");
    STOP_TIMER(&t_timer, "t_total-time:");
}

// function for debug purpose only
// #include "tests.cpp"
// void testcv()
// {
//   //test_minfilter();
// }

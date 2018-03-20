#include "stack.hpp"
#include "stack_helpers.cpp"
#include "./othersift2.hpp"
#include "./stack_tile.cpp"
#include "./stack_section.cpp"


namespace tfk {
    MatchTilesTask::MatchTilesTask (Tile* tile, std::vector<Tile*> neighbors) {
      this->tile = tile;
      this->neighbors = neighbors;
      //this->param_adjustments.resize(7);
      //this->param_train_deltas.resize(7);
    }

    void MatchTilesTask::compute_tile_matches_pair(Tile* a_tile, Tile* b_tile,
      std::vector< cv::KeyPoint >& a_tile_keypoints, std::vector< cv::KeyPoint >& b_tile_keypoints,
      cv::Mat& a_tile_desc, cv::Mat& b_tile_desc,
      std::vector< cv::Point2f > &filtered_match_points_a,
      std::vector< cv::Point2f > &filtered_match_points_b, float ransac_thresh) {
    
      //std::vector<int> neighbors = get_all_close_tiles(a_tile->tile_id);
    
      if (a_tile_keypoints.size() < MIN_FEATURES_NUM) return;
      if (b_tile_keypoints.size() < MIN_FEATURES_NUM) return;
    
      // Filter the features, so that only features that are in the
      //   overlapping tile will be matches.
      std::vector< cv::KeyPoint > atile_kps_in_overlap, btile_kps_in_overlap;
    
      atile_kps_in_overlap.reserve(a_tile_keypoints.size());
      btile_kps_in_overlap.reserve(b_tile_keypoints.size());
    
      // atile_kps_in_overlap.clear(); btile_kps_in_overlap.clear();
      cv::Mat atile_kps_desc_in_overlap, btile_kps_desc_in_overlap;
    
      { // Begin scoped block A.
        // Compute bounding box of overlap
        int overlap_x_start = a_tile->x_start > b_tile->x_start ?
                                  a_tile->x_start : b_tile->x_start;
        int overlap_x_finish = a_tile->x_finish < b_tile->x_finish ?
                                  a_tile->x_finish : b_tile->x_finish;
        int overlap_y_start = a_tile->y_start > b_tile->y_start ?
                                  a_tile->y_start : b_tile->y_start;
        int overlap_y_finish = a_tile->y_finish < b_tile->y_finish ?
                                  a_tile->y_finish : b_tile->y_finish;
        // Add 50-pixel offset
        const int OFFSET = 50;
        overlap_x_start -= OFFSET;
        overlap_x_finish += OFFSET;
        overlap_y_start -= OFFSET;
        overlap_y_finish += OFFSET;
    
        std::vector< cv::Mat > atile_kps_desc_in_overlap_list;
        atile_kps_desc_in_overlap_list.reserve(a_tile_keypoints.size());
        std::vector< cv::Mat > btile_kps_desc_in_overlap_list;
        btile_kps_desc_in_overlap_list.reserve(b_tile_keypoints.size());
    
        // Filter the points in a_tile.
        for (size_t pt_idx = 0; pt_idx < a_tile_keypoints.size(); ++pt_idx) {
          cv::Point2f pt = a_tile_keypoints[pt_idx].pt;
          if (bbox_contains(pt.x + a_tile->x_start,
                            pt.y + a_tile->y_start,  // transformed_pt[0],
                            overlap_x_start, overlap_x_finish,
                            overlap_y_start, overlap_y_finish)) {
            atile_kps_in_overlap.push_back(a_tile_keypoints[pt_idx]);
            atile_kps_desc_in_overlap_list.push_back(
                a_tile_desc.row(pt_idx).clone());
          }
        }
        cv::vconcat(atile_kps_desc_in_overlap_list,
            (atile_kps_desc_in_overlap));
    
        // Filter the points in b_tile.
        for (size_t pt_idx = 0; pt_idx < b_tile_keypoints.size(); ++pt_idx) {
          cv::Point2f pt = b_tile_keypoints[pt_idx].pt;
          if (bbox_contains(pt.x + b_tile->x_start,
                            pt.y + b_tile->y_start,  // transformed_pt[0],
                            overlap_x_start, overlap_x_finish,
                            overlap_y_start, overlap_y_finish)) {
            btile_kps_in_overlap.push_back(b_tile_keypoints[pt_idx]);
            btile_kps_desc_in_overlap_list.push_back(b_tile_desc.row(pt_idx).clone());
          }
        }
        cv::vconcat(btile_kps_desc_in_overlap_list,
            (btile_kps_desc_in_overlap));
      } // End scoped block A
    
      if (atile_kps_in_overlap.size() < MIN_FEATURES_NUM) return;
      if (btile_kps_in_overlap.size() < MIN_FEATURES_NUM) return;
    
      float trial_rod;
      for (int trial = 0; trial < 4; trial++) {
        if (trial == 0) trial_rod = 0.7;
        if (trial == 1) trial_rod = 0.8;
        if (trial == 2) trial_rod = 0.92;
        if (trial == 3) trial_rod = 0.96;
        // Match the features
        std::vector< cv::DMatch > matches;
        match_features(matches,
                       atile_kps_desc_in_overlap,
                       btile_kps_desc_in_overlap,
                       trial_rod);
    
        // Filter the matches with RANSAC
        std::vector<cv::Point2f> match_points_a, match_points_b;
        for (size_t tmpi = 0; tmpi < matches.size(); ++tmpi) {
          match_points_a.push_back(
              atile_kps_in_overlap[matches[tmpi].queryIdx].pt);
          match_points_b.push_back(
              btile_kps_in_overlap[matches[tmpi].trainIdx].pt);
        }
    
        if (matches.size() < MIN_FEATURES_NUM) {
          continue;
        }
    
        bool* mask = (bool*) calloc(match_points_a.size(), 1);
        double thresh = ransac_thresh;//5.0;
        tfk_simple_ransac(match_points_a, match_points_b, thresh, mask);
    
    
        filtered_match_points_a.clear();
        filtered_match_points_b.clear();
    
        int num_matches_filtered = 0;
        // Use the output mask to filter the matches
        for (size_t i = 0; i < matches.size(); ++i) {
          if (mask[i]) {
            num_matches_filtered++;
            filtered_match_points_a.push_back(
                atile_kps_in_overlap[matches[i].queryIdx].pt);
            filtered_match_points_b.push_back(
                btile_kps_in_overlap[matches[i].trainIdx].pt);
          }
        }
        free(mask);
        if (num_matches_filtered >= MIN_FEATURES_NUM) {
          //a_tile->insert_matches(b_tile, filtered_match_points_a, filtered_match_points_b);
          break;
        } else {
          filtered_match_points_a.clear();
          filtered_match_points_b.clear();
        }
      }
    }


    void MatchTilesTask::set_random_train() {
       int index = rand()%param_train_deltas.size();
       for (int i = 0; i < param_train_deltas.size(); i++) {
         param_train_deltas[i] = 0;
       }
       int sign = rand()%2 ? -1 : 1;
       param_train_deltas[index] = sign;
    }

    //jvoid MatchTilesTask::update_result(float last_correct, float next_correct,
    //j    std::vector<int>& param_adjustments, std::vector<int>& param_train_deltas) {
    //j  if (next_correct > last_correct) {
    //j    for (int i = 0; i < param_train_deltas.size(); i++) {
    //j      param_adjustments[i] += param_train_deltas[i];
    //j      param_train_deltas[i] = 0;
    //j    }
    //j  }
    //j  printf("params:\n");
    //j  printf("scale_x %f\n", 0.25 + param_adjustments[0]*0.05);
    //j  printf("scale_y %f\n", 0.25 + param_adjustments[1]*0.05);
    //j  printf("num_features %f\n", 1.0 + param_adjustments[2]);
    //j  printf("num_octaves %f\n", 6.0 + param_adjustments[3]);
    //j  printf("contrast_thresh %f\n", 0.01 + param_adjustments[4]*0.001);
    //j  printf("edge_thresh %f\n", 20.0 + param_adjustments[5]);
    //j  printf("edge_thresh %f\n", 1.2 + param_adjustments[6]*0.05);
    //j}

    void MatchTilesTask::compute(float probability_correct, std::vector<int>& param_adjustments,
        std::vector<int>& param_train_deltas) {
      Tile* a_tile = tile;
      std::vector<cv::KeyPoint> a_tile_keypoints;
      cv::Mat a_tile_desc;
/*
scale_x 0.150000
scale_y -0.050000
num_features -1.000000
num_octaves 8.000000
contrast_thresh 0.011000
edge_thresh 17.000000
edge_thresh 1.000000
Result is 0.570000


*/
      tfk::params new_params;
      new_params.scale_x = 0.15;// + param_adjustments[0]*0.05 + param_train_deltas[0]*0.05;
      new_params.scale_y = 0.15;// + param_adjustments[1]*0.05 + param_train_deltas[1]*0.05;
      new_params.num_features = -1 + param_adjustments[2] + param_train_deltas[2];
      if (new_params.num_features < 0) new_params.num_features = 0;
      new_params.num_octaves = 8 + param_adjustments[3] + param_train_deltas[3];
      new_params.contrast_threshold = 0.011 + param_adjustments[4]*0.001 + param_train_deltas[4]*0.001;
      new_params.edge_threshold = 17 + param_adjustments[5] + param_train_deltas[5];
      new_params.sigma = 1.0 + param_adjustments[6]*0.05 + param_train_deltas[6]*0.05;

      a_tile->compute_sift_keypoints2d_params(new_params, a_tile_keypoints,
                                              a_tile_desc, a_tile);

      if (a_tile_keypoints.size() < MIN_FEATURES_NUM) return; // failure.

      int neighbor_success_count = 0;
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
         
        std::vector<cv::KeyPoint> b_tile_keypoints;
        cv::Mat b_tile_desc;
        
        b_tile->compute_sift_keypoints2d_params(new_params, b_tile_keypoints,
                                                b_tile_desc, a_tile);
        if (b_tile_keypoints.size() < MIN_FEATURES_NUM) continue;
        
        std::vector< cv::Point2f > filtered_match_points_a(0);
        std::vector< cv::Point2f > filtered_match_points_b(0);

       
        this->compute_tile_matches_pair(a_tile, b_tile,
          a_tile_keypoints, b_tile_keypoints,
          a_tile_desc, b_tile_desc,
          filtered_match_points_a,
          filtered_match_points_b, 5.0);

        // store the matched points.
        neighbor_to_matched_points[b_tile] =
            std::make_pair(filtered_match_points_a, filtered_match_points_b);  
      }
    }

    bool MatchTilesTask::error_check(float false_negative_rate) {
      Tile* a_tile = tile;
      int neighbor_success_count = 0;
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        std::vector<cv::Point2f> filtered_match_points_a =
            neighbor_to_matched_points[b_tile].first;
        std::vector<cv::Point2f> filtered_match_points_b =
            neighbor_to_matched_points[b_tile].second;
        Tile tmp_a_tile = *a_tile;
        
        // put b at 0,0
        if (filtered_match_points_a.size() >= MIN_FEATURES_NUM) {
          for (int _i = 0; _i < 1000; _i++) {
            float dx = 0.0;
            float dy = 0.0;
            for (int j = 0; j < filtered_match_points_a.size(); j++) {
              cv::Point2f dp = b_tile->rigid_transform(filtered_match_points_b[j]) - tmp_a_tile.rigid_transform(filtered_match_points_a[j]);
              dx += 2*dp.x * 1.0 / (filtered_match_points_a.size());
              dy += 2*dp.y * 1.0 / (filtered_match_points_a.size());
            }
            tmp_a_tile.offset_x += 0.4*dx;
            tmp_a_tile.offset_y += 0.4*dy;
          }
        }
        float val = tmp_a_tile.error_tile_pair(b_tile);
        if (val >= 0.7) {
          neighbor_to_success[b_tile] = true;
          neighbor_success_count++;
        } else {
          neighbor_to_success[b_tile] = false;
        }
      }
      if (neighbor_success_count >= neighbors.size()*4.0/5.0) {
        return true;
      } else { 
        return false;
      }
    }

    void MatchTilesTask::commit() {
      for (int i = 0; i < neighbors.size(); i++) {
        Tile* b_tile = neighbors[i];
        std::vector<cv::Point2f> filtered_match_points_a =
            neighbor_to_matched_points[b_tile].first;
        std::vector<cv::Point2f> filtered_match_points_b =
            neighbor_to_matched_points[b_tile].second;
        if (neighbor_to_success[b_tile]) {
          tile->insert_matches(b_tile, filtered_match_points_a, filtered_match_points_b);
        }
      }
    }
}



//#include "cilk_tools/engine.h"

tfk::Stack::Stack(int base_section, int n_sections,
    std::string input_filepath, std::string output_dirpath) {
  this->base_section = base_section;
  this->n_sections = n_sections;
  this->input_filepath = input_filepath;
  this->output_dirpath = output_dirpath;
}

void tfk::Stack::render_error(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix) {

  cilk_for (int i = 1; i < this->sections.size()-2; i++) {
    std::cout << "starting section "  << i << std::endl;
    Section* section = this->sections[i];
    std::pair<std::vector<std::pair<cv::Point2f, cv::Point2f>>, std::vector<std::pair<cv::Point2f, cv::Point2f> > > res = section->render_error(this->sections[i-1], this->sections[i+1], this->sections[i+2], bbox, filename_prefix+std::to_string(i)+".png");

  }


}


void tfk::Stack::test_io() {
  for (int i = 0; i < this->sections.size(); i++) {
    Section* section = this->sections[i];
    cilk_for (int j = 0; j < section->tiles.size(); j++) {
      Tile* tile = section->tiles[j];
      cv::Mat mat = tile->get_tile_data(Resolution::FILEIOTEST);
      mat.release();
      printf("tile %d of section %d\n", j, i);
    }
  }
}

void tfk::Stack::render(std::pair<cv::Point2f, cv::Point2f> bbox, std::string filename_prefix,
    Resolution res) {
  cilk_for (int i = 0; i < this->sections.size(); i++) {
    Section* section = this->sections[i];
    section->render(bbox, filename_prefix+std::to_string(i+this->base_section)+".tif", res);
  }
}

void tfk::Stack::recompute_alignment() {
  for (int i = 0; i < this->sections.size(); i++) {
    this->sections[i]->recompute_keypoints();
  }
  this->elastic_align();
}

void tfk::Stack::coarse_affine_align() {
  // get affine transform matrix for each section
  //   matrix A_i aligning section i to i-1.


  this->sections[0]->coarse_affine_align(NULL);
  //cilk_spawn this->sections[0]->coarse_affine_align(this->sections[0]);
  for (int i = 1; i < this->sections.size(); i++) {
    cilk_spawn this->sections[i]->coarse_affine_align(this->sections[i-1]);
  }
  cilk_sync;

  //// simply apply the coarse transform for now.
  //for (int i = 1; i < this->sections.size(); i++) {
  //  this->sections[i]->affine_transforms.clear();
  //  this->sections[i]->affine_transforms.push_back(this->sections[i]->coarse_transform);
  //}

  ////// apply the transforms.
  for (int i = 0; i < this->sections.size(); i++) {
    this->sections[i]->apply_affine_transforms();
  }

  // cascade the affine transforms down.
  //for (int i = 1; i < this->sections.size(); i++) {
  //  for (int j = 0; j < i; j++) {
  //    this->sections[j]->affine_transforms.push_back(this->sections[i]->coarse_transform);
  //  }
  //  //for (int k = 0; k < this->sections[i-1]->affine_transforms.size(); k++) {
  //  //  this->sections[i]->affine_transforms.push_back(this->sections[i-1]->affine_transforms[k]);
  //  //}
  //  //for (int j = i+1; j < this->sections.size(); j++) {

  //  //  for (int k = 0; k < this->sections[j]->affine_transforms.size(); k++) {
  //  //      this->sections[i]->affine_transforms.push_back(this->sections[j]->affine_transforms[k]);
  //  //    }
  //  //}
  //}


}

void tfk::Stack::get_elastic_matches() {
  std::pair<cv::Point2f, cv::Point2f> stack_bbox = this->sections[0]->get_bbox();
  //stack_bbox = this->sections[0]->affine_transform_bbox(stack_bbox);
  //stack_bbox = this->sections[0]->elastic_transform_bbox(stack_bbox);



  //for (int i = 1; i < this->sections.size(); i++) {
  //  auto bbox = this->sections[i]->get_bbox();
  //  //bbox = this->sections[i]->affine_transform_bbox(bbox);
  //  bbox = this->sections[i]->elastic_transform_bbox(bbox);
  //  stack_bbox.first.x = std::min(stack_bbox.first.x, bbox.first.x);
  //  stack_bbox.first.y = std::min(stack_bbox.first.y, bbox.first.y);
  //  stack_bbox.second.x = std::max(stack_bbox.second.x, bbox.second.x);
  //  stack_bbox.second.y = std::max(stack_bbox.second.y, bbox.second.y);
  //}

  //double min_x = stack_bbox.first.x;
  //double max_x = stack_bbox.second.x;
  //double min_y = stack_bbox.first.y;
  //double max_y = stack_bbox.second.y;

  //std::vector<std::pair<double, double> > valid_boxes;
  //for (double box_iter_x = min_x; box_iter_x < max_x + 24000; box_iter_x += 24000) {
  //  for (double box_iter_y = min_y; box_iter_y < max_y + 24000; box_iter_y += 24000) {
  //    valid_boxes.push_back(std::make_pair(box_iter_x, box_iter_y));
  //  }
  //}

  //// simply apply the coarse transform for now.
  //for (int i = 1; i < this->sections.size(); i++) {
  //  this->sections[i]->affine_transforms.clear();
  //  //this->sections[i]->affine_transforms.push_back(this->sections[i]->coarse_transform);
  //}

  //// apply the transforms.
  //for (int i = 0; i < this->sections.size(); i++) {
  //  this->sections[i]->apply_affine_transforms();
  //}


  for (int section = 1; section < this->sections.size(); section++) {
    cilk_spawn this->sections[section]->get_elastic_matches_relative(this->sections[section-1]);
  }
  cilk_sync;

  //// 0 1 2 3 4 5 6
  ////   1   3   5
  //// 1 computes matches for 1->0, 2->1
  //// 3 computes matches for 3->2, 4->3
  //// 5 computes matches for 5->4 6->5
  //cilk_for (int i = 0; i < valid_boxes.size(); i++) {
  //  auto bbox = valid_boxes[i];
  //  std::vector<cv::KeyPoint> prev_keypoints(0);
  //  cv::Mat prev_desc;

  //  for (int section = 1; section < this->sections.size(); section++) {
  //    //neighbors.push_back(this->sections[section-1]);
  //    std::vector<cv::KeyPoint> my_keypoints(0);
  //    cv::Mat my_desc;
  //    this->sections[section]->get_elastic_matches_one_next_bbox(this->sections[section-1], bbox, prev_keypoints,
  //        prev_desc, my_keypoints, my_desc);
  //    prev_keypoints = my_keypoints;
  //    prev_desc = my_desc; 
  //  }
  //}

  //for (int section = 1; section < this->sections.size(); section++) {

  //  std::vector<Section*> neighbors;
  //  int section_a = section;
  //  neighbors.push_back(this->sections[section-1]);

  //  //for (int section_b = section-2; section_b < section+1; section_b++) {
  //  //for (int section_b = section-1; section_b < section; section_b++) {
  //  //  if (section_b < 0 || section_b == section_a || section_b >= this->sections.size()) {
  //  //    continue;
  //  //  }
  //  //  neighbors.push_back(this->sections[section_b]);
  //  //}

  //  this->sections[section]->get_elastic_matches(neighbors);
  //}
}


//START
// only do triangles with vertices in bad areas
void tfk::Stack::elastic_align() {

  cilk_for (int i = 0; i < this->sections.size(); i++) {
    this->sections[i]->construct_triangles();
    this->sections[i]->affine_transform_mesh();
  }

  this->get_elastic_matches();

  this->elastic_gradient_descent();

  


  //std::vector<tfkMatch> mesh_matches;
  //fine_alignment_3d_2(merged_graph, p_align_data,64.0, mesh_matches);

}

void tfk::Stack::elastic_gradient_descent() {
    printf("Running associative gradient descent.\n");


    // NOTES(TFK): Need to remember to transform the meshes with the affine transforms.

    for (int i = 0; i < this->sections.size(); i++) {
      this->sections[i]->mesh_orig_save = this->sections[i]->mesh_orig;
      this->sections[i]->mesh_orig = new std::vector<cv::Point2f>();
      for (int j = 0; j < this->sections[i]->mesh_orig_save->size(); j++) {
        this->sections[i]->mesh_orig->push_back((*this->sections[i]->mesh_orig_save)[j]);
      }
    } 

    for (int iter = 0; iter < 1; iter++) {
      for (int i = 0; i < this->sections.size(); i++) {
        if (i==0) {
          cilk_spawn this->sections[i]->elastic_gradient_descent_section(this->sections[0]);
        } else {
          cilk_spawn this->sections[i]->elastic_gradient_descent_section(this->sections[i-1]);
        }
      }
      cilk_sync;


      // first need to apply the affine transforms and transform the mesh.
      //for (int i = 2; i < this->sections.size(); i++) {
      //  Section* sec = this->sections[i];
      //  int j = i-1;
      //  for (int k = 0; k < sec->mesh->size(); k++) {
      //    (*sec->mesh)[k] = this->sections[j]->elastic_transform((*sec->mesh)[k]);
      //  }
      //}


      // section i is aligned to section i-1;
      for (int i = 1; i < this->sections.size(); i++) {
        Section* sec = this->sections[i];
        //for (int j = i; --j > 0;) {
        int j = i-1;
          for (int k = 0; k < sec->mesh->size(); k++) {
            (*sec->mesh)[k] = this->sections[j]->elastic_transform((*sec->mesh)[k]);
          }
        //}
      }
      for (int i = 0; i < this->sections.size(); i++) {
        for (int j = 0; j < this->sections[i]->mesh_orig_save->size(); j++) {
          (*this->sections[i]->mesh_orig)[j] = (*this->sections[i]->mesh)[j];
        }   
      }
    }

    for (int i = 0; i < this->sections.size(); i++) {
      delete this->sections[i]->mesh_orig;
      this->sections[i]->mesh_orig = this->sections[i]->mesh_orig_save;
    }

    return;


    double cross_slice_weight = 1.0;
    double cross_slice_winsor = 20.0;
    double intra_slice_weight = 1.0;
    double intra_slice_winsor = 200.0;
    int max_iterations = 10000; //ORIGINALL 5000
    //double min_stepsize = 1e-20;
    double stepsize = 0.0001;
    double momentum = 0.5;

    std::map<int, double> gradient_momentum;
    //std::map<int, graph_section_data> section_data_map;
    //section_data_map.clear();

    //for (int i = 0; i < mesh_matches.size(); i++) {
    //  int az = mesh_matches[i].my_section_data.z;
    //  int bz = mesh_matches[i].n_section_data.z;
    //  if (section_data_map.find(az) == section_data_map.end()) {
    //    section_data_map[az] = mesh_matches[i].my_section_data;
    //  }
    //  if (section_data_map.find(bz) == section_data_map.end()) {
    //    section_data_map[bz] = mesh_matches[i].n_section_data;
    //  }
    //}

    //  for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
    //       it != section_data_map.end(); ++it) {
    //    it->second.gradients = new cv::Point2f[it->second.mesh->size()];
    //    it->second.gradients_with_momentum = new cv::Point2f[it->second.mesh->size()];
    //    it->second.rest_lengths = new double[it->second.triangle_edges->size()];
    //    it->second.rest_areas = new double[it->second.triangles->size()];
    //    it->second.mesh_old = new std::vector<cv::Point2f>();

    //    for (int j = 0; j < it->second.mesh->size(); j++) {
    //      it->second.mesh_old->push_back((*(it->second.mesh))[j]);
    //    }

    //    // init
    //    for (int j = 0; j < it->second.mesh->size(); j++) {
    //      it->second.gradients[j] = cv::Point2f(0.0,0.0);  
    //      it->second.gradients_with_momentum[j] = cv::Point2f(0.0,0.0);  
    //    }

    //    for (int j = 0; j < it->second.triangle_edges->size(); j++) {
    //      cv::Point2f p1 = (*(it->second.mesh))[(*(it->second.triangle_edges))[j].first];
    //      cv::Point2f p2 = (*(it->second.mesh))[(*(it->second.triangle_edges))[j].second];
    //      double dx = p1.x-p2.x;
    //      double dy = p1.y-p2.y;
    //      double len = std::sqrt(dx*dx+dy*dy);
    //      it->second.rest_lengths[j] = len;
    //      //printf("Rest length is %f\n", len);
    //    }

    //    // now triangle areas.
    //    for (int j = 0; j < it->second.triangles->size(); j++) {
    //      tfkTriangle tri = (*(it->second.triangles))[j];
    //      cv::Point2f p1 = (*(it->second.mesh))[tri.index1];
    //      cv::Point2f p2 = (*(it->second.mesh))[tri.index2];
    //      cv::Point2f p3 = (*(it->second.mesh))[tri.index3];
    //      it->second.rest_areas[j] = computeTriangleArea(p1,p2,p3);
    //      //printf("Rest area is %f\n", it->second.rest_areas[j]);
    //    }
    //  }


    // INITIALIZE ALL THE SECTION DATA
    for (int i = 0; i < this->sections.size(); i++) {
      int wid = __cilkrts_get_worker_number();
      Section* section = this->sections[i];
      section->gradients = new cv::Point2f[section->mesh->size()];
      section->gradients_with_momentum = new cv::Point2f[section->mesh->size()];
      section->rest_lengths = new double[section->triangle_edges->size()];
      section->rest_areas = new double[section->triangles[wid]->size()];
      section->mesh_old = new std::vector<cv::Point2f>();

      for (int j = 0; j < section->mesh->size(); j++) {
        section->mesh_old->push_back((*(section->mesh))[j]);
      }

      // init
      for (int j = 0; j < section->mesh->size(); j++) {
        section->gradients[j] = cv::Point2f(0.0,0.0);
        section->gradients_with_momentum[j] = cv::Point2f(0.0,0.0);
      }
      for (int j = 0; j < section->triangle_edges->size(); j++) {
        cv::Point2f p1 = (*(section->mesh))[(*(section->triangle_edges))[j].first];
        cv::Point2f p2 = (*(section->mesh))[(*(section->triangle_edges))[j].second];
        double dx = p1.x-p2.x;
        double dy = p1.y-p2.y;
        double len = std::sqrt(dx*dx+dy*dy);
        section->rest_lengths[j] = len;
      }
      for (int j = 0; j < section->triangles[wid]->size(); j++) {
        tfkTriangle tri = (*(section->triangles[wid]))[j];
        cv::Point2f p1 = (*(section->mesh))[tri.index1];
        cv::Point2f p2 = (*(section->mesh))[tri.index2];
        cv::Point2f p3 = (*(section->mesh))[tri.index3];
        section->rest_areas[j] = computeTriangleArea(p1,p2,p3);
      }

    }


    // BEGIN THE GRADIENT DESCENT.

    double prev_cost = 0.0;
    for (int iter = 0; iter < max_iterations; iter++) {
      double cost = 0.0;

      //for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
      //     it != section_data_map.end(); ++it) {
      //  // clear old gradients.
      //  for (int j = 0; j < it->second.mesh->size(); j++) {
      //    it->second.gradients[j] = cv::Point2f(0.0,0.0);  
      //  }
      //}

      // reset the old gradients.
      for (int i = 0; i < this->sections.size(); i++) {
        Section* section = this->sections[i];
        for (int j = 0; j < section->mesh->size(); j++) {
          ((section->gradients))[j] = cv::Point2f(0.0,0.0);
        }
      }

      // compute internal gradients.
      //for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
      //     it != section_data_map.end(); ++it) {
      for (int i = 0; i < this->sections.size(); i++) {
        int wid = __cilkrts_get_worker_number();
        Section* section = this->sections[i];

        // internal_mesh_derivs
        double all_weight = intra_slice_weight;
        double sigma = intra_slice_winsor;
        std::vector<cv::Point2f>* mesh = section->mesh;
        cv::Point2f* gradients = section->gradients;

        std::vector<std::pair<int, int> >* triangle_edges = section->triangle_edges;
        std::vector<tfkTriangle >* triangles = section->triangles[wid];
        double* rest_lengths = section->rest_lengths;
        double* rest_areas = section->rest_areas;

        //// update all edges
        for (int j = 0; j < triangle_edges->size(); j++) {
          cost += internal_mesh_derivs(mesh, gradients, (*triangle_edges)[j], rest_lengths[j],
                                       all_weight, sigma);
        }

        //// update all triangles
        for (int j = 0; j < triangles->size(); j++) {
          int triangle_indices[3] = {(*triangles)[j].index1,
                                     (*triangles)[j].index2,
                                     (*triangles)[j].index3};
          cost += area_mesh_derivs(mesh, gradients, triangle_indices, rest_areas[j], all_weight);
        }
      }


      for (int i = 0; i < this->sections.size(); i++) {
        Section* section = this->sections[i];
        std::vector<tfkMatch>& mesh_matches = section->section_mesh_matches;
        for (int j = 0; j < mesh_matches.size(); j++) {
          Section* my_section = (Section*) mesh_matches[j].my_section;
          Section* n_section = (Section*) mesh_matches[j].n_section;

          std::vector<cv::Point2f>* mesh1 = my_section->mesh;//mesh_matches[j].my_section_data.mesh;
          std::vector<cv::Point2f>* mesh2 = n_section->mesh;//mesh_matches[j].n_section_data.mesh;

          //int myz = mesh_matches[j].my_section_data.z;
          //int nz = mesh_matches[j].n_section_data.z;
          //int myz = my_section->section_id;
          //int nz = n_section->section_id;

          cv::Point2f* gradients1 = my_section->gradients;
          cv::Point2f* gradients2 = n_section->gradients;
          double* barys1 = mesh_matches[j].my_barys;
          double* barys2 = mesh_matches[j].n_barys;
          double all_weight = cross_slice_weight;
          double sigma = cross_slice_winsor;

          int indices1[3] = {mesh_matches[j].my_tri.index1,
                             mesh_matches[j].my_tri.index2,
                             mesh_matches[j].my_tri.index3};
          int indices2[3] = {mesh_matches[j].n_tri.index1,
                             mesh_matches[j].n_tri.index2,
                             mesh_matches[j].n_tri.index3};

          cost += crosslink_mesh_derivs(mesh1, mesh2,
                                        gradients1, gradients2,
                                        indices1, indices2,
                                        barys1, barys2,
                                        all_weight, sigma);

          cost += crosslink_mesh_derivs(mesh2, mesh1,
                                        gradients2, gradients1,
                                        indices2, indices1,
                                        barys2, barys1,
                                        all_weight, sigma);
        }
      }
      if (iter == 0) prev_cost = cost+10.0;

      if (cost <= prev_cost) {
        stepsize *= 1.1;
        if (stepsize > 1.0) {
          stepsize = 1.0;
        }
        // TODO(TFK): momentum.

        //for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
        //     it != section_data_map.end(); ++it) {
        for (int i = 0; i < this->sections.size(); i++) {
          Section* section = this->sections[i];
          std::vector<cv::Point2f>* mesh = section->mesh;
          std::vector<cv::Point2f>* mesh_old = section->mesh_old;
          cv::Point2f* gradients = section->gradients;
          cv::Point2f* gradients_with_momentum = section->gradients_with_momentum;
          for (int j = 0; j < mesh->size(); j++) {
            gradients_with_momentum[j] = gradients[j] + momentum*gradients_with_momentum[j];
          }

          for (int j = 0; j < mesh->size(); j++) {
            (*mesh_old)[j] = ((*mesh)[j]);
          }
          for (int j = 0; j < mesh->size(); j++) {
            (*mesh)[j].x -= (float)(stepsize * (gradients_with_momentum)[j].x);
            (*mesh)[j].y -= (float)(stepsize * (gradients_with_momentum)[j].y);
          }
        }
        if (iter%100 == 0) {
          printf("Good step old cost %f, new cost %f, iteration %d\n", prev_cost, cost, iter);
        }
        prev_cost = cost;
      } else {
        stepsize *= 0.5;
        // bad step undo.
        //for (std::map<int, graph_section_data>::iterator it = section_data_map.begin();
        //     it != section_data_map.end(); ++it) {
        for (int i = 0; i < this->sections.size(); i++) {
          Section* section = this->sections[i];
          std::vector<cv::Point2f>* mesh = section->mesh;
          std::vector<cv::Point2f>* mesh_old = section->mesh_old;
          cv::Point2f* gradients_with_momentum = section->gradients_with_momentum;
          for (int j = 0; j < mesh->size(); j++) {
            gradients_with_momentum[j] = cv::Point2f(0.0,0.0);
          }

          //if (mesh_old->size() != mesh->size()) continue;
          for (int j = 0; j < mesh->size(); j++) {
            (*mesh)[j] = (*mesh_old)[j];
          }
        }
        if (iter%1000 == 0) {
          printf("Bad step old cost %f, new cost %f, iteration %d\n", prev_cost, cost, iter);
        }
      }
    }
    printf("Done with the elastic gradient descent\n");
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
    sec->section_id = this->sections.size();
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
}

void tfk::Stack::compute_on_tile_neighborhood(tfk::Section* section, tfk::Tile* tile) {
  int distance = 2;
  
  std::vector<Tile*> neighbors = section->get_all_close_tiles(tile);
  std::set<Tile*> active_set;

  active_set.insert(tile);
  for (int i = 0; i < neighbors.size(); i++) {
    active_set.insert(neighbors[i]);
  }

  for (int j = 0; j < 5000; j++) {
    tile->local2DAlignUpdateLimited(&active_set);
    for (int i = 0; i < neighbors.size(); i++) {
      neighbors[i]->local2DAlignUpdateLimited(&active_set);
    }
  }

}


void tfk::Stack::align_2d() {
  int count = 0;

  int j = 0;
  int i = 0;
  while (j < this->sections.size()) {
    j += 4;
    if (j >= this->sections.size()) j = this->sections.size();

    for (; i < j; i++) {
       //cilk_spawn this->sections[i]->compute_keypoints_and_matches();
       cilk_spawn this->sections[i]->align_2d();
      //if ((i+1)%4 == 0) cilk_sync;
    }
    cilk_sync;
  }
  return;
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
      //this->coarse_affine_align();
      //this->elastic_align();
      int count = 0;
      for (int i = 0; i < this->sections[section_index]->tiles.size(); i++) {
        Tile* t = this->sections[section_index]->tiles[i];
        if (t->bad_2d_alignment) printf("Tile has bad 2d alignment\n");
        if (t->bad_2d_alignment) continue;
        for (int k = 0; k < t->edges.size(); k++) {
          Tile* neighbor = this->sections[section_index]->tiles[t->edges[k].neighbor_id];
          if (neighbor->bad_2d_alignment) continue;
          if (t->ideal_offsets.find(neighbor->tile_id) == t->ideal_offsets.end()) continue;
          float val = t->compute_deviation(neighbor);
          if (val > 10.0) {
            printf("bad tile with deviation %f corr %f\n", val, t->neighbor_correlations[neighbor->tile_id]);
            //compute_on_tile_neighborhood(this->sections[section_index],t);
            //float val = t->compute_deviation(neighbor);
            //printf("after bad tile with deviation %f corr %f\n", val, t->neighbor_correlations[neighbor->tile_id]);
            //return;
            auto bbox1 = t->get_bbox();
            t->bad_2d_alignment = true;
            neighbor->bad_2d_alignment = true; 
            //this->render(t->get_bbox(), "errortest"+std::to_string(count++), FULL);
          }
          
        }
      }
      break;
    }
  }

}


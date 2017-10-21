


tfk::Tile::Tile(int section_id, int tile_id, int index, std::string filepath,
    int x_start, int x_finish, int y_start, int y_finish) {
  this->section_id = section_id;
  this->tile_id = tile_id;
  this->index = index;
  this->filepath = filepath;
  this->x_start = x_start;
  this->x_finish = x_finish;
  this->y_start = y_start;
  this->y_finish = y_finish;
}


bool tfk::Tile::overlaps_with(Tile* other) {
    int x1_start = this->x_start;
    int x1_finish = this->x_finish;
    int y1_start = this->y_start;
    int y1_finish = this->y_finish;

    int x2_start = other->x_start;
    int x2_finish = other->x_finish;
    int y2_start = other->y_start;
    int y2_finish = other->y_finish;

    bool res = false;
    if ((x1_start < x2_finish) && (x1_finish > x2_start) &&
        (y1_start < y2_finish) && (y1_finish > y2_start)) {
        res = true;
    }
    return res;
}

tfk::Tile::Tile(TileData& tile_data) {
    //tile_data_t *p_cur_tile = &(p_sec_data->tiles[p_sec_data->n_tiles]);
    //p_sec_data->n_tiles++;

    this->section_id = tile_data.section_id();
    this->mfov_id = tile_data.tile_mfov();
    this->index = tile_data.tile_index();
    this->x_start = tile_data.x_start();
    this->x_finish = tile_data.x_finish();
    this->y_start = tile_data.y_start();
    this->y_finish = tile_data.y_finish();
    this->filepath = tile_data.tile_filepath();
    this->p_image = new cv::Mat();
    this->p_kps = new std::vector<cv::KeyPoint>();
    this->p_kps_desc = new cv::Mat();
    this->p_kps_3d = new std::vector<cv::KeyPoint>();
    this->ignore = NULL;
    this->p_kps_desc_3d = new cv::Mat();
    this->level = 0;
    this->bad = false;

    if (tile_data.has_ignore()) {
      bool *ignore = (bool *) malloc(sizeof(bool));
      *ignore = tile_data.ignore();
      this->ignore = ignore;
    }
    if (tile_data.has_a00()) {
      this->a00 = tile_data.a00();
    }
    if (tile_data.has_a10()) {
      this->a10 = tile_data.a10();
    }
    if (tile_data.has_a01()) {
      this->a10 = tile_data.a01();
    }
    if (tile_data.has_a11()) {
      this->a11 = tile_data.a11();
    }
    if (tile_data.has_level()) {
      this->level = tile_data.level();
    }
    if (tile_data.has_bad()) {
      this->bad = tile_data.bad();
    }
}


void tfk::Tile::compute_sift_keypoints3d() {
  (*this->p_image).create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  (*this->p_image) = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);

  int rows = this->p_image->rows;
  int cols = this->p_image->cols;
  ASSERT((rows % SIFT_D1_SHIFT_3D) == 0);
  ASSERT((cols % SIFT_D2_SHIFT_3D) == 0);
  cv::Ptr<cv::Feature2D> p_sift;
  std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
  cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];
  int n_sub_images;

  // NOTE(TFK): I need to check these parameters against the prefix_ cached ones.
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
        p_sift->detectAndCompute((*this->p_image), sub_im_mask, v_kps[sub_im_id],
            m_kps_desc[sub_im_id], false);

        fasttime_t tend = gettime();
        totalTime += tdiff(tstart, tend);
    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    int point_count_3d = 0;
    for (int _i = 0; _i < n_sub_images; _i++) {
        for (int _j = 0; _j < v_kps[_i].size(); _j++) {
            (*this->p_kps_3d).push_back(v_kps[_i][_j]);
            point_count_3d++;
        }
    }

  //cv::Mat m_kps_desc_filtered = m_kps_desc[0].clone();
  *(this)->p_kps_desc_3d = m_kps_desc[0].clone();

  printf("Number of 3d points is %d\n", point_count_3d);
  this->p_image->release();

}

void tfk::Tile::compute_sift_keypoints2d() {

  (*this->p_image).create(SIFT_D2_SHIFT_3D, SIFT_D1_SHIFT_3D, CV_8UC1);
  (*this->p_image) = cv::imread(this->filepath, CV_LOAD_IMAGE_UNCHANGED);

  //printf("Dimensiosn are %d %d\n", this->p_image->rows, this->p_image->cols);


  int rows = this->p_image->rows;
  int cols = this->p_image->cols;

  ASSERT((rows % SIFT_D1_SHIFT) == 0);
  ASSERT((cols % SIFT_D2_SHIFT) == 0);

  cv::Ptr<cv::Feature2D> p_sift;

  std::vector<cv::KeyPoint> v_kps[SIFT_MAX_SUB_IMAGES];
  cv::Mat m_kps_desc[SIFT_MAX_SUB_IMAGES];

  int n_sub_images;
  if ((this->tile_id > MFOV_BOUNDARY_THRESH)) {
    p_sift = new cv::xfeatures2d::SIFT_Impl(
            4,  // num_features --- unsupported.
            6,  // number of octaves
            CONTRAST_THRESH,  // contrast threshold.
            EDGE_THRESH_2D,  // edge threshold.
            1.6);  // sigma.

    // THEN: This tile is on the boundary, we need to compute SIFT features
    // on the entire section.
    int max_rows = rows / SIFT_D1_SHIFT;
    int max_cols = cols / SIFT_D2_SHIFT;
    n_sub_images = max_rows * max_cols;

    cilk_for (int cur_d1 = 0; cur_d1 < rows; cur_d1 += SIFT_D1_SHIFT) {
      cilk_for (int cur_d2 = 0; cur_d2 < cols; cur_d2 += SIFT_D2_SHIFT) {
        //printf("cur_d2 is %d cur_d1 is %d\n", cur_d2, cur_d1);
        // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
        cv::Mat sub_im = (*this->p_image)(cv::Rect(cur_d2, cur_d1,
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
            (*this->p_kps).push_back(v_kps[i][j]);
        }
    }

  } else {
    p_sift = new cv::xfeatures2d::SIFT_Impl(
            4,  // num_features --- unsupported.
            6,  // number of octaves
            CONTRAST_THRESH,  // contrast threshold.
            EDGE_THRESH_2D,  // edge threshold.
            1.6);  // sigma.

    // ELSE THEN: This tile is in the interior of the MFOV. Only need to
    //     compute features along the boundary.
    n_sub_images = 4;

    // BEGIN TOP SLICE
    {
      // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
      cv::Mat sub_im = (*this->p_image)(cv::Rect(
          0, 0, SIFT_D2_SHIFT_3D, OVERLAP_2D));

      // Mask for subimage
      cv::Mat sum_im_mask = cv::Mat::ones(OVERLAP_2D, SIFT_D2_SHIFT_3D, CV_8UC1);
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
      cv::Mat sub_im = (*this->p_image)(cv::Rect(
          0, OVERLAP_2D, OVERLAP_2D, SIFT_D1_SHIFT_3D-OVERLAP_2D));

      // Mask for subimage
      cv::Mat sum_im_mask = cv::Mat::ones(SIFT_D1_SHIFT_3D-OVERLAP_2D, OVERLAP_2D,
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
      cv::Mat sub_im = (*this->p_image)(cv::Rect(
          SIFT_D2_SHIFT_3D-OVERLAP_2D, OVERLAP_2D, OVERLAP_2D, SIFT_D1_SHIFT_3D-OVERLAP_2D));

      // Mask for subimage
      cv::Mat sum_im_mask = cv::Mat::ones(SIFT_D1_SHIFT_3D-OVERLAP_2D, OVERLAP_2D,
          CV_8UC1);
      int sub_im_id = 2;
      // Detect the SIFT features within the subimage.
      fasttime_t tstart = gettime();

      p_sift->detectAndCompute(sub_im, sum_im_mask, v_kps[sub_im_id],
          m_kps_desc[sub_im_id]);
      fasttime_t tend = gettime();
      totalTime += tdiff(tstart, tend);
      for (int i = 0; i < v_kps[sub_im_id].size(); i++) {
          v_kps[sub_im_id][i].pt.x += SIFT_D2_SHIFT_3D-OVERLAP_2D;  // cur_d2;
          v_kps[sub_im_id][i].pt.y += OVERLAP_2D;  // cur_d1;
      }
    }
    // END RIGHT SLICE

    // BEGIN BOTTOM SLICE
    {
      // Subimage of size SIFT_D1_SHIFT x SHIFT_D2_SHIFT
      cv::Mat sub_im = (*this->p_image)(cv::Rect(
          OVERLAP_2D, SIFT_D1_SHIFT_3D-OVERLAP_2D, SIFT_D2_SHIFT_3D-OVERLAP_2D, OVERLAP_2D));

      // Mask for subimage
      cv::Mat sum_im_mask = cv::Mat::ones(OVERLAP_2D, SIFT_D2_SHIFT_3D-OVERLAP_2D,
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
          v_kps[sub_im_id][i].pt.y += (SIFT_D1_SHIFT_3D-OVERLAP_2D);  // cur_d1;
      }
    }
    // END BOTTOM SLICE

    // Regardless of whether we were on or off MFOV boundary, we concat
    //   the keypoints and their descriptors here.
    for (int i = 0; i < n_sub_images; i++) {
        for (size_t j = 0; j < v_kps[i].size(); j++) {
            (*this->p_kps).push_back(v_kps[i][j]);
        }
    }
  }

  cv::vconcat(m_kps_desc, n_sub_images, *(this->p_kps_desc));
  this->p_image->release();
}



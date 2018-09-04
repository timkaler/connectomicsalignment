#ifndef STACK_HELPERS_H
#define STACK_HELPERS_H
namespace cv {
  static bool computeAffineTFK(const std::vector<cv::Point2f> &srcPoints, const std::vector<cv::Point2f> &dstPoints, Mat &transf)
  {
      // sanity check
      if ((srcPoints.size() < 3) || (srcPoints.size() != dstPoints.size()))
          return false;
  
      // container for output
      transf.create(2, 3, CV_64F);
  
      // fill the matrices
      const int n = (int)srcPoints.size(), m = 3;
      Mat A(n,m,CV_64F), xc(n,1,CV_64F), yc(n,1,CV_64F);
      for(int i=0; i<n; i++)
      {
          double x = srcPoints[i].x, y = srcPoints[i].y;
          double rowI[m] = {x, y, 1};
          Mat(1,m,CV_64F,rowI).copyTo(A.row(i));
          xc.at<double>(i,0) = dstPoints[i].x;
          yc.at<double>(i,0) = dstPoints[i].y;
      }
  
      // solve linear equations (for x and for y)
      Mat aTa, resX, resY;
      mulTransposed(A, aTa, true);
      //solve(aTa, A.t()*xc, resX, DECOMP_CHOLESKY);
      //solve(aTa, A.t()*yc, resY, DECOMP_CHOLESKY);
      solve(aTa, A.t()*xc, resX, DECOMP_SVD);
      solve(aTa, A.t()*yc, resY, DECOMP_SVD);
  
      // store result
      memcpy(transf.ptr<double>(0), resX.data, m*sizeof(double));
      memcpy(transf.ptr<double>(1), resY.data, m*sizeof(double));
  
      return true;
  }
}



static inline std::string matchPadTo(std::string str, const size_t num, const char paddingChar = '0')
{
    // suppress unused function warning.
    (void) matchPadTo;
    if(num > str.size())
        str.insert(0, num - str.size(), paddingChar);
    return str;
}


static cv::Mat colorize(cv::Mat c1, cv::Mat c2, cv::Mat c3, unsigned int channel = 0) {
    std::vector<cv::Mat> channels;
    channels.push_back(c1);
    channels.push_back(c2);
    channels.push_back(c3);
    cv::Mat color;
    cv::merge(channels, color);
    return color;
}


static cv::Mat apply_heatmap_to_grayscale(cv::Mat* gray, cv::Mat* heat_floats, int nrows, int ncols) {
  cv::Mat c1,c2,c3;
  c1.create(nrows, ncols, CV_8UC1);
  c2.create(nrows, ncols, CV_8UC1);
  c3.create(nrows, ncols, CV_8UC1);

  for (int x = 0; x < gray->size().width; x++) {
    for (int y = 0; y < gray->size().height; y++) {
       float g = (1.0*gray->at<unsigned char>(y,x))/255;
       float h = 0.5 + 0.5*(heat_floats->at<float>(y,x)*heat_floats->at<float>(y,x));

       c1.at<unsigned char>(y,x) = (unsigned char) (g*255*(1-h));
       c2.at<unsigned char>(y,x) = (unsigned char) (g*255*(1-h));
       c3.at<unsigned char>(y,x) = (unsigned char) (g*255*((h)));
    }
  }
  return colorize(c1,c2,c3);
}

// Helper method to check if a key point is inside a given bounding
// box.
__attribute__((const))
static bool bbox_contains(float pt_x, float pt_y,
                          int x_start, int x_finish,
                          int y_start, int y_finish) {
  // TRACE_1("  -- pt: (%f, %f)\n", pt.x, pt.y);
  // TRACE_1("  -- bbox: [(%d, %d), (%d, %d)]\n",
  //         x_start, y_start,
  //         x_finish, y_finish);
  return (pt_x >= x_start && pt_x <= x_finish) &&
    (pt_y >= y_start && pt_y <= y_finish);
}

static float Dot(cv::Point2f a, cv::Point2f b) {
  return a.x*b.x + a.y*b.y;
}

static double computeTriangleArea(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {


  double v01x = p2.x - p1.x;
  double v01y = p2.y - p1.y;

  double v02x = p3.x - p1.x;
  double v02y = p3.y - p1.y;

  double area = 0.5 * (v02x * v01y - v01x * v02y);
  return area;
  //double dx,dy;
  //dx = p1.x-p2.x;
  //dy = p1.y-p2.y;
  //double p1p2 = std::sqrt(dx*dx+dy*dy);

  //dx = p1.x-p3.x;
  //dy = p1.y-p3.y;
  //double p1p3 = std::sqrt(dx*dx+dy*dy);

  //dx = p2.x-p3.x;
  //dy = p2.y-p3.y;
  //double p2p3 = std::sqrt(dx*dx+dy*dy);

  //double p = (p1p2+p1p3+p2p3)/2;
  //double area = std::sqrt(p*(p-p1p2)*(p-p1p3)*(p-p2p3));
  //return area;
}

static double c_huber(double value, double target, double sigma, double d_value_dx, double d_value_dy,
               double* d_huber_dx, double* d_huber_dy) {
  double diff, a, b;

  diff = value - target;
  if (std::abs(diff) <= sigma) {
    a = (diff*diff)/2;
    d_huber_dx[0] = diff * d_value_dx;
    d_huber_dy[0] = diff * d_value_dy;
    return a;
  } else {
    b = sigma * (std::abs(diff) - sigma / 2);
    d_huber_dx[0] = sigma * d_value_dx;
    d_huber_dy[0] = sigma * d_value_dy;
    return b;
  }
}

static double c_reglen(double vx, double vy, double d_vx_dx, double d_vy_dy,
                double* d_reglen_dx, double* d_reglen_dy) {
  double sq_len, sqrt_len;
  double small_value = 0.0001;
  sq_len = vx * vx + vy * vy + small_value;
  sqrt_len = std::sqrt(sq_len);
  d_reglen_dx[0] = vx / sqrt_len;
  d_reglen_dy[0] = vy / sqrt_len;
  return sqrt_len;
}


//static double crosslink_mesh_derivs(std::vector<cv::Point2f>* mesh1, std::vector<cv::Point2f>* mesh2,
//                             cv::Point2f* d_cost_d_mesh1, cv::Point2f* d_cost_d_mesh2,
//                             int* indices1, int* indices2, double* barys1, double* barys2,
//                             double all_weight, double sigma) {



static double crosslink_mesh_derivs(std::vector<cv::Point2f>* mesh1,
                             cv::Point2f* d_cost_d_mesh1,
                             int* indices1, double* barys1,
                             double all_weight, double sigma, cv::Point2f dest_p) {
  double px, py, qx, qy;
  int pidx0, pidx1, pidx2;
  int qidx0, qidx1, qidx2;
  double pb0, pb1, pb2;
  double qb0, qb1, qb2;
  double r, h;
  double dr_dx, dr_dy, dh_dx, dh_dy;
  double cost;

  cost = 0;

  pidx0 = indices1[0];
  pidx1 = indices1[1];
  pidx2 = indices1[2];

  pb0 = barys1[0];
  pb1 = barys1[1];
  pb2 = barys1[2];



  px = (*mesh1)[pidx0].x * pb0 +
       (*mesh1)[pidx1].x * pb1 + 
       (*mesh1)[pidx2].x * pb2;

  py = (*mesh1)[pidx0].y * pb0 +
       (*mesh1)[pidx1].y * pb1 +
       (*mesh1)[pidx2].y * pb2;

  qx = dest_p.x;
  qy = dest_p.y;

  r = c_reglen(px-qx, py-qy,1,1,&(dr_dx),&(dr_dy));
  h = c_huber(r, 0, sigma, dr_dx, dr_dy, &(dh_dx), &(dh_dy));

  cost += h * all_weight;
  dh_dx *= all_weight;
  dh_dy *= all_weight;

  // update derivs.
  d_cost_d_mesh1[pidx0].x += (float)1.0*(pb0 * dh_dx);
  d_cost_d_mesh1[pidx1].x += (float)1.0*(pb1 * dh_dx);
  d_cost_d_mesh1[pidx2].x += (float)1.0*(pb2 * dh_dx);

  d_cost_d_mesh1[pidx0].y += (float)(pb0 * dh_dy);
  d_cost_d_mesh1[pidx1].y += (float)(pb1 * dh_dy);
  d_cost_d_mesh1[pidx2].y += (float)(pb2 * dh_dy);

  //d_cost_d_mesh2[qidx0].x -= (float)(qb0 * dh_dx);
  //d_cost_d_mesh2[qidx1].x -= (float)(qb1 * dh_dx);
  //d_cost_d_mesh2[qidx2].x -= (float)(qb2 * dh_dx);

  //d_cost_d_mesh2[qidx0].y -= (float)(qb0 * dh_dy);
  //d_cost_d_mesh2[qidx1].y -= (float)(qb1 * dh_dy);
  //d_cost_d_mesh2[qidx2].y -= (float)(qb2 * dh_dy);

  return cost;
}

static double internal_mesh_derivs(std::vector<cv::Point2f>* mesh, cv::Point2f* d_cost_d_mesh,
                            std::pair<int, int> edge_indices,
                            double rest_length, double all_weight,
                            double sigma) {
  int idx1,idx2;
  double px,py,qx,qy;
  double r,h;
  double dr_dx, dr_dy, dh_dx, dh_dy;
  double cost;

  cost = 0;

  idx1 = edge_indices.first;
  idx2 = edge_indices.second;

  px = (*mesh)[idx1].x;
  py = (*mesh)[idx1].y;

  qx = (*mesh)[idx2].x; 
  qy = (*mesh)[idx2].y;

  r = c_reglen(px-qx, py-qy, 1, 1, &(dr_dx), &(dr_dy));
  h = c_huber(r, rest_length, sigma, dr_dx, dr_dy, &(dh_dx), &(dh_dy));

  cost += h * all_weight;
  dh_dx *= all_weight;
  dh_dy *= all_weight;

  // update derivs.
  d_cost_d_mesh[idx1].x += dh_dx;
  d_cost_d_mesh[idx1].y += dh_dy;
  d_cost_d_mesh[idx2].x -= dh_dx;
  d_cost_d_mesh[idx2].y -= dh_dy;

  return cost;
}

static double area_mesh_derivs(std::vector<cv::Point2f>* mesh, cv::Point2f* d_cost_d_mesh,
                        int* triangle_indices, double rest_area, double all_weight) {
  int idx0, idx1, idx2;
  double v01x, v01y, v02x, v02y, area, r_area;
  double cost, c, dc_da;

  cost = 0;

  idx0 = triangle_indices[0];
  idx1 = triangle_indices[1];
  idx2 = triangle_indices[2];

  v01x = (*mesh)[idx1].x - (*mesh)[idx0].x;
  v01y = (*mesh)[idx1].y - (*mesh)[idx0].y;

  v02x = (*mesh)[idx2].x - (*mesh)[idx0].x;
  v02y = (*mesh)[idx2].y - (*mesh)[idx0].y;

  area = 0.5 * (v02x * v01y - v01x * v02y);
  r_area = rest_area;
  if (area*r_area <= 0) {
    c = INFINITY;
    dc_da = 0;
  } else {
    double tmp = ((area - r_area) / area);
    c = all_weight * (tmp*tmp);
    dc_da = 2 * all_weight * r_area * (area - r_area) / (area*area*area);
  }
  cost += c;

  // update derivs
  d_cost_d_mesh[idx1].x += dc_da * 0.5 * (-1*v02y);
  d_cost_d_mesh[idx1].y += dc_da * 0.5 * (v02x);
  d_cost_d_mesh[idx2].x += dc_da * 0.5 * (v01y);
  d_cost_d_mesh[idx2].y += dc_da * 0.5 * (-1*v01x);

  // sum of negative of above.
  d_cost_d_mesh[idx0].x += dc_da * 0.5 * (v02y - v01y);
  d_cost_d_mesh[idx0].y += dc_da * 0.5 * (v01x - v02x);

  return cost;
}



static void Barycentric(cv::Point2f p, cv::Point2f a, cv::Point2f b, cv::Point2f c,
   float &u, float &v, float &w)
{
    cv::Point2f v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = Dot(v0, v0);
    float d01 = Dot(v0, v1);
    float d11 = Dot(v1, v1);
    float d20 = Dot(v2, v0);
    float d21 = Dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}
#endif

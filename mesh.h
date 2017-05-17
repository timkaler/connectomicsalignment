
std::vector<cv::Point2f>* generate_hex_grid(double* bounding_box, double spacing) {
  double hexheight = spacing;
  double hexwidth = sqrt(3.0) * spacing / 2.0;
  double vertspacing = 0.75 * hexheight;
  double horizspacing = hexwidth;
  int sizex = (int)((bounding_box[1]-bounding_box[0])/horizspacing) + 2; 
  int sizey = (int)((bounding_box[3]-bounding_box[2])/vertspacing) + 2;

  if (sizey % 2 == 0) {
    sizey += 1;
  }

  std::vector<cv::Point2f>* hex_grid = new std::vector<cv::Point2f>();
  for (int i = -2; i < sizex; i++) {
    for (int j = -2; j < sizey; j++) {
      double xpos = i * spacing;
      double ypos = j * spacing;
      if (j % 2 == 1) {
        xpos += spacing * 0.5;
      }
      if (j % 2 == 1 && i == sizex-1) {
        continue;
      }
      hex_grid->push_back(cv::Point2f(xpos+bounding_box[0], ypos+bounding_box[2]));
    }
  }
  return hex_grid;
}

double c_huber(double value, double target, double sigma, double d_value_dx, double d_value_dy,
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

double c_reglen(double vx, double vy, double d_vx_dx, double d_vy_dy,
                double* d_reglen_dx, double* d_reglen_dy) {
  double sq_len, sqrt_len;
  double small_value = 0.0001;
  sq_len = vx * vx + vy * vy + small_value;
  sqrt_len = std::sqrt(sq_len);
  d_reglen_dx[0] = vx / sqrt_len;
  d_reglen_dy[0] = vy / sqrt_len;
  return sqrt_len;
}


double crosslink_mesh_derivs(std::vector<cv::Point2f>* mesh1, std::vector<cv::Point2f>* mesh2,
                             cv::Point2f* d_cost_d_mesh1, cv::Point2f* d_cost_d_mesh2,
                             int* indices1, int* indices2, double* barys1, double* barys2,
                             double all_weight, double sigma) {
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

  qidx0 = indices2[0];
  qidx1 = indices2[1];
  qidx2 = indices2[2];

  qb0 = barys2[0];
  qb1 = barys2[1];
  qb2 = barys2[2];

  px = (*mesh1)[pidx0].x * pb0 +
       (*mesh1)[pidx1].x * pb1 + 
       (*mesh1)[pidx2].x * pb2;

  py = (*mesh1)[pidx0].y * pb0 +
       (*mesh1)[pidx1].y * pb1 + 
       (*mesh1)[pidx2].y * pb2;

  qx = (*mesh2)[qidx0].x * qb0 +
       (*mesh2)[qidx1].x * qb1 + 
       (*mesh2)[qidx2].x * qb2;

  qy = (*mesh2)[qidx0].y * qb0 +
       (*mesh2)[qidx1].y * qb1 + 
       (*mesh2)[qidx2].y * qb2;

  r = c_reglen(px-qx, py-qy,1,1,&(dr_dx),&(dr_dy));
  h = c_huber(r, 0, sigma, dr_dx, dr_dy, &(dh_dx), &(dh_dy));

  cost += h * all_weight;
  dh_dx *= all_weight;
  dh_dy *= all_weight;

  // update derivs.
  d_cost_d_mesh1[pidx0].x += pb0 * dh_dx;
  d_cost_d_mesh1[pidx1].x += pb1 * dh_dx;
  d_cost_d_mesh1[pidx2].x += pb2 * dh_dx;

  d_cost_d_mesh1[pidx0].y += pb0 * dh_dy;
  d_cost_d_mesh1[pidx1].y += pb1 * dh_dy;
  d_cost_d_mesh1[pidx2].y += pb2 * dh_dy;

  d_cost_d_mesh2[qidx0].x -= qb0 * dh_dx;
  d_cost_d_mesh2[qidx1].x -= qb1 * dh_dx;
  d_cost_d_mesh2[qidx2].x -= qb2 * dh_dx;

  d_cost_d_mesh2[qidx0].y -= qb0 * dh_dy;
  d_cost_d_mesh2[qidx1].y -= qb1 * dh_dy;
  d_cost_d_mesh2[qidx2].y -= qb2 * dh_dy;

  return cost;
}

double internal_mesh_derivs(std::vector<cv::Point2f>* mesh, cv::Point2f* d_cost_d_mesh,
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

double area_mesh_derivs(std::vector<cv::Point2f>* mesh, cv::Point2f* d_cost_d_mesh,
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


float Dot(cv::Point2f a, cv::Point2f b) {
  return a.x*b.x + a.y*b.y;
}
void Barycentric(cv::Point2f p, cv::Point2f a, cv::Point2f b, cv::Point2f c,
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

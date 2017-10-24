#ifndef MESH_H
#define MESH_H

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
      //double xpos = i * spacing;
      //double ypos = j * spacing;
      double xpos = i * horizspacing;
      double ypos = j * vertspacing;
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


static double crosslink_mesh_derivs(std::vector<cv::Point2f>* mesh1, std::vector<cv::Point2f>* mesh2,
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
  d_cost_d_mesh1[pidx0].x += (float)1.0*(pb0 * dh_dx);
  d_cost_d_mesh1[pidx1].x += (float)1.0*(pb1 * dh_dx);
  d_cost_d_mesh1[pidx2].x += (float)1.0*(pb2 * dh_dx);

  d_cost_d_mesh1[pidx0].y += (float)(pb0 * dh_dy);
  d_cost_d_mesh1[pidx1].y += (float)(pb1 * dh_dy);
  d_cost_d_mesh1[pidx2].y += (float)(pb2 * dh_dy);

  d_cost_d_mesh2[qidx0].x -= (float)(qb0 * dh_dx);
  d_cost_d_mesh2[qidx1].x -= (float)(qb1 * dh_dx);
  d_cost_d_mesh2[qidx2].x -= (float)(qb2 * dh_dx);

  d_cost_d_mesh2[qidx0].y -= (float)(qb0 * dh_dy);
  d_cost_d_mesh2[qidx1].y -= (float)(qb1 * dh_dy);
  d_cost_d_mesh2[qidx2].y -= (float)(qb2 * dh_dy);

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


static float Dot(cv::Point2f a, cv::Point2f b) {
  return a.x*b.x + a.y*b.y;
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



void construct_triangles(Graph* graph, double hex_spacing) {
  double min_x, max_x, min_y, max_y;
  for (int v = 0; v < graph->num_vertices(); v++) {
     vdata* vd = graph->getVertexData(v);
     if (v==0) {
       min_x = vd->start_x;
       max_x = vd->end_x;
       min_y = vd->start_y;
       max_y = vd->end_y;
     }
     double xvals[2] = {vd->start_x, vd->end_x};
     double yvals[2] = {vd->start_y, vd->end_y};
     for (int i = 0; i < 2; i++) {
       if (xvals[i] < min_x) min_x = xvals[i];
       if (xvals[i] > max_x) max_x = xvals[i];
     }
     for (int i = 0; i < 2; i++) {
       if (yvals[i] < min_y) min_y = yvals[i];
       if (yvals[i] > max_y) max_y = yvals[i];
     }
  }

  double bounding_box[4] = {min_x,max_x,min_y,max_y};
  std::vector<cv::Point2f>* hex_grid = generate_hex_grid(bounding_box, hex_spacing);

  cv::Rect rect(min_x-hex_spacing*2,min_y-hex_spacing*2,max_x-min_x+hex_spacing*4, max_y-min_y + hex_spacing*4); 
  cv::Subdiv2D subdiv(rect);
  subdiv.initDelaunay(rect);
  for (int i = 0; i < hex_grid->size(); i++) {
    cv::Point2f pt = (*hex_grid)[i];
    //printf("hex grid %f, %f\n", pt.x, pt.y);
    //printf("grid values are %f, %f, %f, %f\n",min_x-hex_spacing*2,min_y-hex_spacing*2,max_x-min_x+hex_spacing*4, max_y-min_y + hex_spacing*4); 
    subdiv.insert(pt);
  }

  std::vector<cv::Vec6f> triangle_list;
  subdiv.getTriangleList(triangle_list);

  printf("The number of triangles is %lu\n", triangle_list.size()); 

  //cv::Point2f* d_cost_d_mesh = new cv::Point2f[hex_grid->size()];

  std::vector<tfkTriangle> triangle_list_index;
  for (int i = 0; i < triangle_list.size(); i++) {
    cv::Point2f tpt1 = cv::Point2f(triangle_list[i][0],triangle_list[i][1]);
    cv::Point2f tpt2 = cv::Point2f(triangle_list[i][2],triangle_list[i][3]);
    cv::Point2f tpt3 = cv::Point2f(triangle_list[i][4],triangle_list[i][5]);
    int index1=-1;
    int index2=-1;
    int index3=-1;

    for (int j = 0; j < hex_grid->size(); j++) {
      cv::Point2f pt = (*hex_grid)[j];
      if (std::abs(pt.x-tpt1.x) < 0.01 && std::abs(pt.y-tpt1.y) < 0.01) {
        index1 = j;    
      }

      if (std::abs(pt.x-tpt2.x) < 0.01 && std::abs(pt.y-tpt2.y) < 0.01) {
        index2 = j;    
      }
      
      if (std::abs(pt.x-tpt3.x) < 0.01 && std::abs(pt.y-tpt3.y) < 0.01) {
        index3 = j;    
      }
    }
    //printf("triangle is %f %f %f %f %f %f\n", triangle_list[i][0], triangle_list[i][1], triangle_list[i][2], triangle_list[i][3], triangle_list[i][4], triangle_list[i][5]);
    //printf("points are %f %f %f %f %f %f\n", tpt1.x, tpt1.y, tpt2.x, tpt2.y, tpt3.x, tpt3.y);
    if (!(index1 >= 0 && index2 >= 0 && index3 >=0)) continue;
   // printf("Success\n");
    tfkTriangle tri;
    tri.index1 = index3;
    tri.index2 = index2;
    tri.index3 = index1;
    triangle_list_index.push_back(tri);
  }

  printf("Triangle_list_index length is %lu\n", triangle_list_index.size());

  std::vector<std::pair<int,int> > triangle_edges;
  for (int i = 0; i < triangle_list_index.size(); i++) {
    tfkTriangle tri = triangle_list_index[i];

    if (tri.index1 < tri.index2) {
      triangle_edges.push_back(std::make_pair(tri.index1,tri.index2));
    } else {
      triangle_edges.push_back(std::make_pair(tri.index2,tri.index1));
    }

    if (tri.index2 < tri.index3) {
      triangle_edges.push_back(std::make_pair(tri.index2,tri.index3));
    } else {
      triangle_edges.push_back(std::make_pair(tri.index3,tri.index2));
    }

    if (tri.index1 < tri.index3) {
      triangle_edges.push_back(std::make_pair(tri.index1,tri.index3));
    } else {
      triangle_edges.push_back(std::make_pair(tri.index3,tri.index1));
    }
  } 

  std::vector<std::pair<int,int> > triangle_edges_dedupe;
  std::set<std::pair<int, int> > triangle_edges_set;
  for (int i = 0; i < triangle_edges.size(); i++) {
    if (triangle_edges_set.find(triangle_edges[i]) == triangle_edges_set.end()) {
      triangle_edges_set.insert(triangle_edges[i]);
      triangle_edges_dedupe.push_back(triangle_edges[i]);
    }
  }


  std::vector<std::pair<int, int> >* _triangle_edges = new std::vector<std::pair<int, int> >();
  for (int i = 0; i < triangle_edges_dedupe.size(); i++) {
    _triangle_edges->push_back(triangle_edges_dedupe[i]);
  }
  std::vector<tfkTriangle>* _triangle_list = new std::vector<tfkTriangle>();
  for (int i = 0; i < triangle_list_index.size(); i++) {
    _triangle_list->push_back(triangle_list_index[i]);
  }  

  std::vector<cv::Point2f>* orig_hex_grid = new std::vector<cv::Point2f>();
  for (int i = 0; i < hex_grid->size(); i++) {
    orig_hex_grid->push_back((*hex_grid)[i]);
  }
 
  graph_section_data section_data;

  section_data.triangle_edges = _triangle_edges;
  section_data.mesh_orig = orig_hex_grid;
  section_data.mesh = hex_grid;
  section_data.triangles = _triangle_list;
  section_data.transform = new cv::Mat(3,3,cv::DataType<double>::type);
  section_data.transform->at<double>(0,0) = 1.0;
  section_data.transform->at<double>(0,1) = 0.0;
  section_data.transform->at<double>(0,2) = 0.0;
  section_data.transform->at<double>(1,0) = 0.0;
  section_data.transform->at<double>(1,1) = 1.0;
  section_data.transform->at<double>(1,2) = 0.0;
  section_data.transform->at<double>(2,0) = 0.0;
  section_data.transform->at<double>(2,1) = 0.0;
  section_data.transform->at<double>(2,2) = 1.0;
  section_data.z = graph->getVertexData(0)->z;

  graph_section_data* section_data_ptr = new graph_section_data[1];
  *section_data_ptr = section_data;
  for (int v = 0; v < graph->num_vertices(); v++) {
    graph->getVertexData(v)->section_data = section_data_ptr;

    double min_x = graph->getVertexData(v)->start_x +graph->getVertexData(v)->offset_x - hex_spacing*4;
    double min_y = graph->getVertexData(v)->start_y +graph->getVertexData(v)->offset_y- hex_spacing*4;
    double max_x = graph->getVertexData(v)->end_x+graph->getVertexData(v)->offset_x + hex_spacing*4;
    double max_y = graph->getVertexData(v)->end_y+graph->getVertexData(v)->offset_y + hex_spacing*4;
    
    std::vector<int>* my_mesh_points = new std::vector<int>();
    graph->getVertexData(v)->my_mesh_points = my_mesh_points;
    // get all triangles overlapping.
    for (int i = 0; i < hex_grid->size(); i++) {
      cv::Point2f pt = (*hex_grid)[i];
      if (pt.x < min_x || pt.x > max_x) continue;
      if (pt.y < min_y || pt.y > max_y) continue;
      my_mesh_points->push_back(i);
    }
    //printf("Vertex %d has mesh point count %d\n", v, my_mesh_points->size());
  }

  printf("Now done with setup num edges is before dedupe %lu, after %lu\n", triangle_edges.size(), triangle_edges_dedupe.size());

  

  // need array of edges in triangulation.
  //for (int i = 0; i < triangle_list.size(); i++) {
  //  std::pair<int, int> e1 =  
  //} 


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


#endif // MESH_H

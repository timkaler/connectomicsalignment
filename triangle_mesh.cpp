#include "triangle_mesh.hpp"
#include "range_tree.hpp"

namespace tfk {



Triangle TriangleMesh::find_triangle(cv::Point2f pt) {
  return index->find_triangle(pt);
}

Triangle TriangleMesh::find_triangle_post(cv::Point2f pt) {
  return index_post->find_triangle(pt);
}


void TriangleMesh::build_index_post() {


  Triangle* items = new Triangle[triangles->size()];
  //std::vector<Triangle> items;

  auto new_bbox = bbox;

  for (int i = 0; i < triangles->size(); i++) {
    Triangle t;
    t.index = i;
    t.points[0] = (*mesh)[(*triangles)[i].index1];
    t.points[1] = (*mesh)[(*triangles)[i].index2];
    t.points[2] = (*mesh)[(*triangles)[i].index3];
    items[i] = t;
    for (int j = 0; j < 3; j++) {
      cv::Point2f pt = t.points[j];
      if (pt.x < new_bbox.first.x) new_bbox.first.x = pt.x;
      if (pt.x > new_bbox.second.x) new_bbox.second.x = pt.x;
      if (pt.y < new_bbox.first.y) new_bbox.first.y = pt.y;
      if (pt.y > new_bbox.second.y) new_bbox.second.y = pt.y;
    }
  }

  printf("Post The total size of the triangles before is %zu\n", triangles->size());


  new_bbox.first.x-=0.01;
  new_bbox.first.y-=0.01;
  new_bbox.second.x+=0.01;
  new_bbox.second.y+=0.01;

  this->index_post = new RangeTree(items,triangles->size(), new_bbox);
  printf("index post pointer is %p, my pointer %p\n", this->index_post, this);
  printf("The total size of the tree is %d\n", index->get_total_item_count());

  std::set<int> index_set = index->get_index_set();
  printf("The total size of the index set is %zu\n", index_set.size());
  //exit(0);
  //Triangle tri = index->find_triangle(cv::Point2f(16806.0, 20157.0));
  //if (tri.index==-1) {
  //  printf("Failure!\n");
  //  printf("bbox is %f %f %f %f\n", bbox.first.x, bbox.first.y, bbox.second.x, bbox.second.y);
  //  exit(0);
  //}
}

void TriangleMesh::build_index() {

  bbox = std::make_pair(cv::Point2f(0.0,0.0),cv::Point2f(0.0,0.0));



  Triangle* items = new Triangle[triangles->size()];
  for (int i = 0; i < triangles->size(); i++) {
    Triangle t;
    t.index = i;
    t.points[0] = (*mesh_orig)[(*triangles)[i].index1];
    t.points[1] = (*mesh_orig)[(*triangles)[i].index2];
    t.points[2] = (*mesh_orig)[(*triangles)[i].index3];
    items[i] = t;

    for (int j = 0; j < 3; j++) {
      cv::Point2f pt = t.points[j];
      if (pt.x < bbox.first.x) bbox.first.x = pt.x;
      if (pt.x > bbox.second.x) bbox.second.x = pt.x;
      if (pt.y < bbox.first.y) bbox.first.y = pt.y;
      if (pt.y > bbox.second.y) bbox.second.y = pt.y;
    }

  }

  printf("The total size of the triangles before is %zu\n", triangles->size());


  bbox.first.x-=0.01;
  bbox.first.y-=0.01;
  bbox.second.x+=0.01;
  bbox.second.y+=0.01;

  this->index = new RangeTree(items,triangles->size(), bbox);
  printf("index pointer is %p, my pointer %p\n", this->index, this);
  printf("The total size of the tree is %d\n", index->get_total_item_count());

  std::set<int> index_set = index->get_index_set();
  printf("The total size of the index set is %zu\n", index_set.size());
  //exit(0);
  Triangle tri = index->find_triangle(cv::Point2f(16806.0, 20157.0));
  if (tri.index==-1) {
    printf("Failure!\n");
    printf("bbox is %f %f %f %f\n", bbox.first.x, bbox.first.y, bbox.second.x, bbox.second.y);
    exit(0);
  }
}

TriangleMesh::TriangleMesh(TriangleMeshProto triangleMesh) {


  mesh = new std::vector<cv::Point2f>();
  mesh_orig = new std::vector<cv::Point2f>();

  for (int i = 0; i < triangleMesh.mesh_size(); i++) {
    mesh->push_back(cv::Point2f(triangleMesh.mesh(i).x(),
                                triangleMesh.mesh(i).y()));
    mesh_orig->push_back(cv::Point2f(triangleMesh.mesh_orig(i).x(),
                                     triangleMesh.mesh_orig(i).y()));
  }

  triangle_edges = new std::vector<std::pair<int,int> >();
  for (int i = 0; i < triangleMesh.triangle_edges_size(); i++) {
    triangle_edges->push_back(std::make_pair(triangleMesh.triangle_edges(i).x(),
                                             triangleMesh.triangle_edges(i).y()));
  }

  triangles = new std::vector<tfkTriangle>();
  for (int i = 0; i < triangleMesh.triangles_size(); i++) {
    tfkTriangle tri;
    tri.index1 = triangleMesh.triangles(i).index1();
    tri.index2 = triangleMesh.triangles(i).index2();
    tri.index3 = triangleMesh.triangles(i).index3();
    triangles->push_back(tri);
  }

  printf("building the index\n");
  this->build_index();
  this->build_index_post();
}

TriangleMesh::TriangleMesh(double hex_spacing,
                           std::pair<cv::Point2f, cv::Point2f> _bbox) {

  this->bbox = _bbox;
  bbox.first.x-=hex_spacing*3+0.01;
  bbox.first.y-=hex_spacing*3+0.01;
  bbox.second.x+=hex_spacing*3+0.01;
  bbox.second.y+=hex_spacing*3+0.01;

  double min_x = bbox.first.x;
  double min_y = bbox.first.y;
  double max_x = bbox.second.x;
  double max_y = bbox.second.y;

  double bounding_box[4] = {min_x,max_x,min_y,max_y};
  std::vector<cv::Point2f>* hex_grid = this->generate_hex_grid(bounding_box, hex_spacing);

  cv::Rect rect(min_x-hex_spacing*3,min_y-hex_spacing*3,max_x-min_x+hex_spacing*6, max_y-min_y + hex_spacing*6);


  cv::Subdiv2D subdiv(rect);
  subdiv.initDelaunay(rect);
  for (int i = 0; i < hex_grid->size(); i++) {
    cv::Point2f pt = (*hex_grid)[i];
    subdiv.insert(pt);
  }

  std::vector<cv::Vec6f> triangle_list;
  subdiv.getTriangleList(triangle_list);
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
    if (!(index1 >= 0 && index2 >= 0 && index3 >=0)) continue;
    tfkTriangle tri;
    tri.index1 = index3;
    tri.index2 = index2;
    tri.index3 = index1;
    triangle_list_index.push_back(tri);
  }

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



  //std::vector<tfkTriangle>* _triangle_list = new std::vector<tfkTriangle>();
  std::vector< std::vector<tfkTriangle>* > _triangle_list;
  int nworkers = 1;
  for (int i = 0; i < nworkers; i++) {
    _triangle_list.push_back(new std::vector<tfkTriangle>());
  }



  for (int i = 0; i < triangle_list_index.size(); i++) {
    for (int j = 0; j < nworkers; j++) {
      _triangle_list[j]->push_back(triangle_list_index[i]);
    }
  }


  std::vector<cv::Point2f>* orig_hex_grid = new std::vector<cv::Point2f>();
  for (int i = 0; i < hex_grid->size(); i++) {
    orig_hex_grid->push_back((*hex_grid)[i]);
  }

  this->triangle_edges = _triangle_edges;
  this->mesh_orig = orig_hex_grid;
  this->mesh = hex_grid;
  this->triangles = _triangle_list[0];
  build_index();
}


// begin private methods
std::vector<cv::Point2f>* TriangleMesh::generate_hex_grid(double* bounding_box, double spacing) {
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


void TriangleMesh::expand_bbox(double hex_spacing,
                               std::pair<cv::Point2f, cv::Point2f> _bbox){

  // this->bbox = _bbox;
  double min_x = _bbox.first.x;
  double min_y = _bbox.first.y;
  double max_x = _bbox.second.x;
  double max_y = _bbox.second.y;

  // double min_x_orig = bbox.first.x;
  // double min_y_orig = bbox.first.y;
  // double max_x_orig = bbox.second.x;
  // double max_y_orig = bbox.second.y;
  this->bbox = _bbox;

  double bounding_box[4] = {min_x,max_x,min_y,max_y};

  cv::Rect rect(min_x-hex_spacing*3,min_y-hex_spacing*3,max_x-min_x+hex_spacing*6, max_y-min_y + hex_spacing*6);

  // Set to max possible
  double min_x_orig = bbox.second.x;
  double min_y_orig = bbox.second.y;
  double max_x_orig = bbox.first.x;
  double max_y_orig = bbox.first.y;
  cv::Subdiv2D subdiv(rect);
  subdiv.initDelaunay(rect);
  int prev_mesh_size = (int)(this->mesh->size());
  for (int i = 0; i < prev_mesh_size; i++) {
    cv::Point2f pt = (*this->mesh_orig)[i];
    subdiv.insert(pt);
    if(pt.x < min_x_orig) min_x_orig = pt.x;
    if(pt.y < min_y_orig) min_y_orig = pt.y;
    if(pt.x > max_x_orig) max_x_orig = pt.x;
    if(pt.y > max_y_orig) max_y_orig = pt.y;
  }
  double bounding_box_orig[4] = {min_x_orig,max_x_orig,min_y_orig,max_y_orig};
  this->add_mesh_border(bounding_box, bounding_box_orig, hex_spacing);

  for(int i=prev_mesh_size; i<this->mesh_orig->size(); i++){
    cv::Point2f pt = (*this->mesh_orig)[i];
    subdiv.insert(pt);
  }

  std::vector<cv::Vec6f> triangle_list;
  subdiv.getTriangleList(triangle_list);
  std::vector<tfkTriangle> triangle_list_index;

  for (int i = 0; i < triangle_list.size(); i++) {
    cv::Point2f tpt1 = cv::Point2f(triangle_list[i][0],triangle_list[i][1]);
    cv::Point2f tpt2 = cv::Point2f(triangle_list[i][2],triangle_list[i][3]);
    cv::Point2f tpt3 = cv::Point2f(triangle_list[i][4],triangle_list[i][5]);
    int index1=-1;
    int index2=-1;
    int index3=-1;

    for (int j = 0; j < this->mesh_orig->size(); j++) {
      cv::Point2f pt = (*this->mesh_orig)[j];
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
    if (!(index1 >= 0 && index2 >= 0 && index3 >=0)) continue;
    if (index1<prev_mesh_size && index2<prev_mesh_size && index3<prev_mesh_size) continue;
    tfkTriangle tri;
    tri.index1 = index3;
    tri.index2 = index2;
    tri.index3 = index1;
    triangle_list_index.push_back(tri);
  }

  std::vector<std::pair<int,int> > triangle_edges;
  for (int i = 0; i < triangle_list_index.size(); i++) {
    tfkTriangle tri = triangle_list_index[i];

    if(tri.index1>=prev_mesh_size || tri.index2>=prev_mesh_size){
      if (tri.index1 < tri.index2) {
        triangle_edges.push_back(std::make_pair(tri.index1,tri.index2));
      } else {
        triangle_edges.push_back(std::make_pair(tri.index2,tri.index1));
      }
    }

    if(tri.index3>=prev_mesh_size || tri.index2>=prev_mesh_size){
      if (tri.index2 < tri.index3) {
        triangle_edges.push_back(std::make_pair(tri.index2,tri.index3));
      } else {
        triangle_edges.push_back(std::make_pair(tri.index3,tri.index2));
      }
    }

    if(tri.index3>=prev_mesh_size || tri.index1>=prev_mesh_size){
      if (tri.index1 < tri.index3) {
        triangle_edges.push_back(std::make_pair(tri.index1,tri.index3));
      } else {
        triangle_edges.push_back(std::make_pair(tri.index3,tri.index1));
      }
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

  for (int i = 0; i < triangle_edges_dedupe.size(); i++) {
    this->triangle_edges->push_back(triangle_edges_dedupe[i]);
  }


  for (int i = 0; i < triangle_list_index.size(); i++) {
    this->triangles->push_back(triangle_list_index[i]);
  }

  build_index();
} 


void TriangleMesh::add_mesh_border(double* bounding_box, double* old_bounding_box, double spacing) {
  double hexheight = spacing;
  double hexwidth = sqrt(3.0) * spacing / 2.0;
  double vertspacing = 0.75 * hexheight;
  double horizspacing = hexwidth;
  int sizex = (int)(round((old_bounding_box[1]-old_bounding_box[0])/horizspacing))+1; 
  int sizey = (int)(round((old_bounding_box[3]-old_bounding_box[2])/vertspacing))+1;

  int extend_left = (int)((bounding_box[0] - old_bounding_box[0])/horizspacing)-1;
  int extend_up = (int)((bounding_box[2] - old_bounding_box[2])/vertspacing)-1;
  int extend_right = (int)((bounding_box[1] - old_bounding_box[1])/horizspacing)+1;
  int extend_down = (int)((bounding_box[3] - old_bounding_box[3])/vertspacing)+1;
  //if(abs(bounding_box[0]-old_bounding_box[0])<.1) extend_left = 0;
  //if(abs(bounding_box[2]-old_bounding_box[2])<.1) extend_up = 0;
  //if(abs(bounding_box[1]-old_bounding_box[1])<.1) extend_right = 0;
  //if(abs(bounding_box[3]-old_bounding_box[3])<.1) extend_down = 0;
  if(extend_left>-1) extend_left = -1;
  if(extend_right<1) extend_right = 1;
  if(extend_up>-1) extend_up = -2;
  if(extend_down<1) extend_down = 2;
  if(extend_up % 2 != 0) {
    extend_up-=1;
  }
  if(extend_down % 2 != 0) {
    extend_down+=1;
  }
  printf("top_left: (%d,%d), bottom_right (%d,%d)\n", extend_left, extend_up,extend_right,extend_down);
  double left_pos = old_bounding_box[0];
  double top_pos = old_bounding_box[2];

  // Add the left side of the mesh
  for(int i = extend_left; i<0; i++){
    for (int j=extend_up; j<extend_down+sizey; j++){
      double xpos = i * horizspacing;
      double ypos = j * vertspacing;
      if (j % 2 == 1) {
        xpos += spacing * 0.5;
      }
      this->mesh_orig->push_back(cv::Point2f(xpos+left_pos, ypos+top_pos));
      this->mesh->push_back(cv::Point2f(xpos+left_pos, ypos+top_pos));
    }
  }
  // Add the right side of the mesh
  for(int i = sizex-1; i<extend_right+sizex && extend_right>0; i++){
    for (int j=extend_up; j<extend_down+sizey; j++){
      double xpos = i * horizspacing;
      double ypos = j * vertspacing;
      if (j % 2 == 1) {
        xpos += spacing * 0.5;
      }
      if (j % 2 == 1 && i == extend_right+sizex-1) {
        continue;
      }
      if (j % 2 == 0 && i == sizex-1) {
        continue;
      }
      this->mesh_orig->push_back(cv::Point2f(xpos+left_pos, ypos+top_pos));
      this->mesh->push_back(cv::Point2f(xpos+left_pos, ypos+top_pos));
    }
  }
  // Add top and bottom of the mesh
  for(int i = 0; i<sizex; i++){
    for(int j = extend_up; j<0; j++){
      double xpos = i * horizspacing;
      double ypos = j * vertspacing;
      if (j % 2 == 1) {
        xpos += spacing * 0.5;
      }
      if (j % 2 == 1 && i == sizex-1) {
        continue;
      }
      this->mesh_orig->push_back(cv::Point2f(xpos+left_pos, ypos+top_pos));
      this->mesh->push_back(cv::Point2f(xpos+left_pos, ypos+top_pos));
    }
    for(int j = sizey; j<sizey+extend_down; j++){
      double xpos = i * horizspacing;
      double ypos = j * vertspacing;
      if (j % 2 == 1) {
        xpos += spacing * 0.5;
      }
      if (j % 2 == 1 && i == sizex-1) {
        continue;
      }
      this->mesh_orig->push_back(cv::Point2f(xpos+left_pos, ypos+top_pos));
      this->mesh->push_back(cv::Point2f(xpos+left_pos, ypos+top_pos));
    }
  }
}




// end namespace tfk
}

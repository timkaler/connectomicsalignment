#include "data.hpp"


namespace tfk {

void sample_stack(Stack* stack, int num_samples, int box_size, std::string filename_prefix){
  auto entire_bbox = stack->sections[0]->get_bbox();
  tfk::Render* render = new tfk::Render();
  //srand(time(NULL));
  //srand();
  for (int i=0; i<num_samples; ++i){
    int section_num = rand() % (stack->sections.size()-1);
    Section* section1 = stack->sections[section_num];
    Section* section2 = stack->sections[section_num+1];
    float start_x =(float)(rand() % ((int)entire_bbox.second.x - box_size));
    float start_y =(float)(rand() % ((int)entire_bbox.second.y - box_size));
    float end_x = start_x + (float)box_size;
    float end_y = start_y + (float)box_size;
    auto sample_box = std::make_pair(cv::Point2f(start_x,start_y), cv::Point2f(end_x, end_y));
    render->render(section1, sample_box, tfk::FULL, filename_prefix+"_"+std::to_string(i)+"_"+
    std::to_string(section1->real_section_id)+"_"+std::to_string(start_x)+"_"+std::to_string(start_y)+"_1.tif");
    render->render(section2, sample_box, tfk::FULL, filename_prefix+"_"+std::to_string(i)+"_"+
    std::to_string(section2->real_section_id)+"_"+std::to_string(start_x)+"_"+std::to_string(start_y)+"_2.tif");
  }
}

//TODO:Explain how this works using cross products to detect direction of spin (ccw vs cw) 
bool mesh_overlaps(Stack* stack){
  bool result = false;
  int count = 0;
  for (int i=0; i<stack->sections.size(); i++) {
    Section* section = stack->sections[i];
    for (int i = 0; i < section->triangle_mesh->triangles->size(); i++) {
      tfkTriangle tri = (*(section->triangle_mesh->triangles))[i];
      cv::Point2f p1 = (*(section->triangle_mesh->mesh))[tri.index1];
      cv::Point2f p2 = (*(section->triangle_mesh->mesh))[tri.index2];
      cv::Point2f p3 = (*(section->triangle_mesh->mesh))[tri.index3];
      cv::Point2f op1 = (*(section->triangle_mesh->mesh_orig))[tri.index1];
      cv::Point2f op2 = (*(section->triangle_mesh->mesh_orig))[tri.index2];
      cv::Point2f op3 = (*(section->triangle_mesh->mesh_orig))[tri.index3];
      float cross_p = (p2.y-p1.y)*(p3.x-p1.x) - (p3.y-p1.y)*(p2.x-p1.x);
      float cross_op = (op2.y-op1.y)*(op3.x-op1.x) - (op3.y-op1.y)*(op2.x-op1.x);
      if (cross_p*cross_op < 0){
        count++;
        printf("OVERLAP AT Section: %d | (%f)\n(%f,%f),(%f,%f),(%f, %f)\n(%f,%f),(%f,%f),(%f,%f)\n",
            section->real_section_id,cross_p*cross_op,
            p1.x,p1.y,p2.x,p2.y,p3.x,p3.y,
            op1.x,op1.y,op2.x,op2.y,op3.x,op3.y);
        result = true;
      }
    }
  }
  printf("The mesh overlaps %d times\n", count);
  return result;
}

std::vector<Triangle>* bad_triangles(Section* section){
  std::vector<Triangle>* triangles_list = new std::vector<Triangle>();
  for (int i = 0; i < section->triangle_mesh->triangles->size(); i++) {
    tfkTriangle tri = (*(section->triangle_mesh->triangles))[i];
    cv::Point2f p1 = (*(section->triangle_mesh->mesh))[tri.index1];
    cv::Point2f p2 = (*(section->triangle_mesh->mesh))[tri.index2];
    cv::Point2f p3 = (*(section->triangle_mesh->mesh))[tri.index3];
    cv::Point2f op1 = (*(section->triangle_mesh->mesh_orig))[tri.index1];
    cv::Point2f op2 = (*(section->triangle_mesh->mesh_orig))[tri.index2];
    cv::Point2f op3 = (*(section->triangle_mesh->mesh_orig))[tri.index3];
    float cross_p = (p2.y-p1.y)*(p3.x-p1.x) - (p3.y-p1.y)*(p2.x-p1.x);
    float cross_op = (op2.y-op1.y)*(op3.x-op1.x) - (op3.y-op1.y)*(op2.x-op1.x);
    if (cross_p*cross_op < 0){
      Triangle bad_t;
      bad_t.points[0] = p1;
      bad_t.points[1] = p2;
      bad_t.points[2] = p3;
      triangles_list->push_back(bad_t);
    }
  }
  return triangles_list;
}

void overlay_triangles(Section* section, std::pair<cv::Point2f, cv::Point2f> bbox,
    tfk::Resolution resolution, std::string filename){
  tfk::Render* render = new tfk::Render();
  //cv::Mat img = render->render(section, bbox, resolution);
  cv::Mat img;
  img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  cv::Point2f render_scale = render->get_render_scale(section, resolution);

  //std::pair<cv::Point2f, cv::Point2f> scaled_bbox = render->scale_bbox(bbox, render_scale);
  int input_lower_x = bbox.first.x;
  int input_lower_y = bbox.first.y;
  //int input_upper_x = bbox.second.x;
  //int input_upper_y = bbox.second.y;
  //int nrows = (input_upper_y-input_lower_y)/render_scale.y;
  //int ncols = (input_upper_x-input_lower_x)/render_scale.x;

  std::vector<Triangle>* bad_list = bad_triangles(section);
  printf("Detected %zu Flipped Triangles in Section %d\n",bad_list->size(), section->real_section_id);
  for(int i = 0; i<bad_list->size(); i++){
    Triangle tri = (*bad_list)[i];
    for(int j = 0; j<3; j++){
      tri.points[j].x = (tri.points[j].x-(float)input_lower_x)/render_scale.x;
      tri.points[j].y = (tri.points[j].y-(float)input_lower_y)/render_scale.y;
    }
    for(int j=0; j<3; j++){
      cv::line(img, tri.points[j], tri.points[(j+1)%3],CV_RGB(255,0,0),5);
    }
  }

  cv::imwrite(filename,img);//"t_"+filename, img);
}

void overlay_mesh(Section* section, std::pair<cv::Point2f, cv::Point2f> bbox,
    tfk::Resolution resolution, std::string filename){
  tfk::Render* render = new tfk::Render();
  //cv::Mat img = render->render(section, bbox, resolution);
  cv::Mat img;
  img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  cv::Point2f render_scale = render->get_render_scale(section, resolution);

  //std::pair<cv::Point2f, cv::Point2f> scaled_bbox = render->scale_bbox(bbox, render_scale);
  int input_lower_x = bbox.first.x;
  int input_lower_y = bbox.first.y;
  //int input_upper_x = bbox.second.x;
  //int input_upper_y = bbox.second.y;
  //int nrows = (input_upper_y-input_lower_y)/render_scale.y;
  //int ncols = (input_upper_x-input_lower_x)/render_scale.x;

  std::vector<Triangle>* triangles_list = new std::vector<Triangle>();
  //std::vector<Triangle>* triangles_old = new std::vector<Triangle>();
  std::vector<Triangle>* bad_list = bad_triangles(section);
  printf("Matches try %zu\n", section->section_mesh_matches.size());
  std::vector<tfkMatch>& mesh_matches = section->section_mesh_matches;


  for (int i = 0; i < section->triangle_mesh->triangles->size(); i++) {
    tfkTriangle tri = (*(section->triangle_mesh->triangles))[i];
    Triangle new_t;
    new_t.points[0] = (*(section->triangle_mesh->mesh))[tri.index1];
    new_t.points[1] = (*(section->triangle_mesh->mesh))[tri.index2];
    new_t.points[2] = (*(section->triangle_mesh->mesh))[tri.index3];
    //Triangle old_t;
    //old_t.points[0] = (*(section->triangle_mesh->mesh_orig))[tri.index1];
    //old_t.points[1] = (*(section->triangle_mesh->mesh_orig))[tri.index2];
    //old_t.points[2] = (*(section->triangle_mesh->mesh_orig))[tri.index3];
    triangles_list->push_back(new_t);
    //triangles_old->push_back(old_t);
  }
 
  for(int i = 0; i<triangles_list->size(); i++){
    Triangle tri = (*triangles_list)[i];
    for(int j = 0; j<3; j++){
      tri.points[j].x = (tri.points[j].x-(float)input_lower_x)/render_scale.x;
      tri.points[j].y = (tri.points[j].y-(float)input_lower_y)/render_scale.y;
    }
    auto color = CV_RGB(0,0,255);
    if(i >= section->added_triangles) color = CV_RGB(175,0,175);
    for(int j=0; j<3; j++){
      cv::line(img, tri.points[j], tri.points[(j+1)%3],CV_RGB(0,0,255),5);
    }
  }
  //for(int i = 0; i<triangles_old->size(); i++){
  //  Triangle tri = (*triangles_old)[i];
  //  for(int j = 0; j<3; j++){
  //    tri.points[j].x = (tri.points[j].x-(float)input_lower_x)/render_scale.x;
  //    tri.points[j].y = (tri.points[j].y-(float)input_lower_y)/render_scale.y;
  //  }
  //  for(int j=0; j<3; j++){
  //    cv::line(img, tri.points[j], tri.points[(j+1)%3],CV_RGB(0,0,255),5);
  //  }
  //}

  printf("There are %zu matches\n", mesh_matches.size());
  for(int i = 0; i < mesh_matches.size(); i++){
    Triangle tri;
    tri.points[0] = (*(section->triangle_mesh->mesh))[mesh_matches[i].my_tri.index1];
    tri.points[1] = (*(section->triangle_mesh->mesh))[mesh_matches[i].my_tri.index2];
    tri.points[2] = (*(section->triangle_mesh->mesh))[mesh_matches[i].my_tri.index3];
    for(int j = 0; j<3; j++){
      tri.points[j].x = (tri.points[j].x-(float)input_lower_x)/render_scale.x;
      tri.points[j].y = (tri.points[j].y-(float)input_lower_y)/render_scale.y;
    }
    for(int j=0; j<3; j++){
      cv::line(img, tri.points[j], tri.points[(j+1)%3],CV_RGB(0,255,0),5);
    }
    double* barys = mesh_matches[i].my_barys;
    float px, py;
    px = tri.points[0].x*barys[0] + tri.points[1].x*barys[1] + tri.points[2].x*barys[2];
    py = tri.points[0].y*barys[0] + tri.points[1].y*barys[1] + tri.points[2].y*barys[2];
    cv::Point2f p = cv::Point2f(px,py);
    cv::circle(img,p, 5, CV_RGB(0,255,255),5, 8, 0); 
  }

  for(int i=section->added_points; i<section->triangle_mesh->mesh->size(); i++){
    float px, py;
    px = ((*(section->triangle_mesh->mesh))[i].x-(float)input_lower_x)/render_scale.x;
    py = ((*(section->triangle_mesh->mesh))[i].y-(float)input_lower_y)/render_scale.y;
    cv::Point2f p = cv::Point2f(px,py);
    cv::circle(img,p, 5, CV_RGB(255,0,255),5, 8, 0); 
  }

  for(int i = 0; i<bad_list->size(); i++){
    Triangle tri = (*bad_list)[i];
    for(int j = 0; j<3; j++){
      tri.points[j].x = (tri.points[j].x-(float)input_lower_x)/render_scale.x;
      tri.points[j].y = (tri.points[j].y-(float)input_lower_y)/render_scale.y;
    }
    for(int j=0; j<3; j++){
      cv::line(img, tri.points[j], tri.points[(j+1)%3],CV_RGB(255,0,0),5);
    }
  }

  cv::imwrite(filename,img);//"t_"+filename, img);
}

void overlay_triangles_stack(Stack* stack,
    std::pair<cv::Point2f, cv::Point2f> bbox, tfk::Resolution resolution,
    std::string filename_prefix) {

  cilk_for (int i = 0; i < stack->sections.size(); i++) {
    Section* section = stack->sections[i];
    overlay_mesh(section, bbox, resolution,
           filename_prefix+"_"+std::to_string(section->real_section_id)+".tif"); 
  }
}
// end namespace tfk
}

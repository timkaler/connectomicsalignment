#include "data.hpp"


namespace tfk {

void sample_stack(Stack* stack, int num_samples, int box_size, std::string filename_prefix){
  auto entire_bbox = stack->sections[0]->get_bbox();
  tfk::Render* render = new tfk::Render();
  srand(time(NULL));
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


// end namespace tfk
}

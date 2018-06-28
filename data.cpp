#include "data.hpp"


namespace tfk {

Data::Data() {}

void Data::sample_stack(Stack* stack, int num_samples, int box_size, std::string filename_prefix){
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



// end namespace tfk
}

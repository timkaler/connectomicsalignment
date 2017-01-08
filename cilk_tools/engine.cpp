// Copyright (c) 2013, Tim Kaler - MIT License

#include <vector>
#include "./engine.h"

template<typename VertexType, typename EdgeType>
engine<VertexType, EdgeType>::engine(
    Graph<VertexType, EdgeType>* graph, Scheduler* scheduler) {
  this->graph = graph;
  this->scheduler = scheduler;
}

template<typename VertexType, typename EdgeType>
void engine<VertexType, EdgeType>::run() {
  int iterationCount = 0;
  std::vector<std::vector<Scheduler::update_task>*> subbags =
      scheduler->get_task_bag();
  while (subbags.size() > 0) {
    iterationCount++;
    if (iterationCount%100){
    printf("iteration count is %d\n", iterationCount);
    }
    parallel_process(subbags);
    subbags = scheduler->get_task_bag();
    //if (iterationCount % 100 == 0) {
    //}
  }
  printf("iteration count is %d\n", iterationCount);
}

template<typename VertexType, typename EdgeType>
void engine<VertexType, EdgeType>::process_update_task(
    Scheduler::update_task task) {
  task.update_fun(task.vid, scheduler);
}

template<typename VertexType, typename EdgeType>
void engine<VertexType, EdgeType>::parallel_process(
    std::vector<std::vector<Scheduler::update_task>*> subbags) {
  cilk_for (int _i = 0; _i < subbags.size(); _i++) {
    cilk_for (int _j = 0; _j < subbags[_i]->size(); _j++) {
      this->process_update_task((*(subbags[_i]))[_j]);
    }
  }
}

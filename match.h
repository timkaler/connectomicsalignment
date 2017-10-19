/////////////////////////////////////////////////////////////////////////////////////////
//
//
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef MATCH_H
#define MATCH_H 1

/////////////////////////////////////////////////////////////////////////////////////////
// INCLUDES
/////////////////////////////////////////////////////////////////////////////////////////
#include "common.h"
#include "cilk_tools/Graph.h"
/////////////////////////////////////////////////////////////////////////////////////////
// DEFINES 
/////////////////////////////////////////////////////////////////////////////////////////
const int MIN_FEATURES_NUM = 5;
const int MAX_EPSILON = 10;

/////////////////////////////////////////////////////////////////////////////////////////
// TYPES
/////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////
//void compute_tile_matches(align_data_t *p_align_data);
void compute_alignment_2d(align_data_t *p_align_data, Graph<vdata, edata>* merged_graph);
void compute_alignment_3d(align_data_t *p_align_data, Graph<vdata, edata>* merged_graph, bool construct_tri);

void compute_tile_matches_active_set(align_data_t *p_align_data, int sec_id, std::set<int> active_set, Graph<vdata, edata>* graph);

void set_graph_list(std::vector<Graph<vdata,edata>* > graph_list, bool startEmpty);

void unpack_graph(align_data_t* p_align_data, Graph<vdata,edata>* merged_graph);

Graph<vdata, edata>* pack_graph();

int get_all_close_tiles(int atile_id, section_data_t *p_sec_data, int* indices_to_check);

void match_features(std::vector< cv::DMatch > &matches,
                           cv::Mat &descs1, cv::Mat &descs2,
                           float rod);
cv::Point2f transform_point(vdata* vertex, cv::Point2f point_local);
void updateVertex2DAlignFULLFast(int vid, void* scheduler_void);
void computeError2DAlign(int vid, void* scheduler_void);


template<typename VertexType, typename EdgeType>
class engine {
 private:
  Graph<VertexType, EdgeType>* graph;
  Scheduler* scheduler;
 public:
  engine(Graph<VertexType, EdgeType>* graph, Scheduler* scheduler);
  void run();
  void process_update_task(Scheduler::update_task task);
  void parallel_process(
      std::vector<std::vector<Scheduler::update_task>*> subbags);
};

class Scheduler {
 public:
    struct update_task{
      int vid;
      void (*update_fun)(int, void*);
    };
 private:
    Multibag<update_task>* Q;
    int currentColor;
    int* vertexColors;
    int colorCount;
    int* vertex_task_added;
    int numVertices;
    bool keepGoing;
 public:
    Scheduler(int* vertexColors, int colorCount, int vertexCount);
    void add_task(int vid, void (*update_function)(int, void*));
    void add_task_static(int vid, void (*update_function)(int, void*));
    std::vector<std::vector<update_task>*> get_task_bag();
    void collect_tasks();
    int roundNum;
    void* graph_void;
    bool isStatic;
};


/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL MACROS
/////////////////////////////////////////////////////////////////////////////////////////

#endif // MATCH_H

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

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL MACROS
/////////////////////////////////////////////////////////////////////////////////////////

#endif // MATCH_H

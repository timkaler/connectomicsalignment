
#ifndef ALIGN_H
#define ALIGN_H 1

#include "common.h"

void align_execute(align_data_t *p_align_data);

void param_optimize(align_data_t *p_align_data);

void test_learning(align_data_t *p_align_data);

void testcv();

void fill_match_tiles_task_pdb(align_data_t *p_align_data);
#endif // ALIGN_H

// Copyright 2016 - Supertech Research Group
//   Tim Kaler, Tao B. Schardl, Haoran Xu, Charles E. Leiserson, Alex Matveev.

#ifndef ALIGN_H
#define ALIGN_H 1

#include "./common.h"

void align_execute(align_data_t *p_align_data, bool do_3d, bool do_render);

void param_optimize(align_data_t *p_align_data);

void test_learning(align_data_t *p_align_data);
void train_fsj(align_data_t *p_align_data, int num_train_iters);

void testcv();

void fill_match_tiles_task_pdb(align_data_t *p_align_data);

void testing_corralation_test(align_data_t *p_align_data);
#endif  // ALIGN_H

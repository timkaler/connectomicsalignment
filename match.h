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
void compute_tile_matches(align_data_t *p_align_data, int force_section_id);

/////////////////////////////////////////////////////////////////////////////////////////
// EXTERNAL MACROS
/////////////////////////////////////////////////////////////////////////////////////////

#endif // MATCH_H

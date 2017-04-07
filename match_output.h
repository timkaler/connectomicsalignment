const int INDENT_SPACES = 4;
static void save_tile_matches(size_t num_matches,
                      const char *out_filepath,
                      const tile_data_t *a_data,
                      const tile_data_t *b_data,
                      const std::vector< cv::Point2f > *match_points_a,
                      const std::vector< cv::Point2f > *match_points_b,
                      const std::vector< cv::Point2f > *match_points_a_fixed) {
#ifndef SKIPJSON
  static double totalTime = 0;
  FILE *fp;

  TRACE_1("save_tile_matches: start\n");

  TRACE_1("Writing %s\n", out_filepath);

  fasttime_t tstart = gettime();

  fp = fopen(out_filepath, "wb+");
  // Output prologue
  fprintf(fp, "[\n");
  fprintf(fp, "%*s%s\n", INDENT_SPACES, "", "{");

  // Output correspondence point pairs
  TRACE_1("  -- outputting %lu matches\n", num_matches);
  fprintf(fp, "%*s%s\n", 2*INDENT_SPACES, "",
          "\"correspondencePointPairs\": [");
  for (size_t i = 0; i < num_matches; ++i) {
    fprintf(fp, "%*s%s\n", 3*INDENT_SPACES, "", "{");

    // Emit distance
    fprintf(fp, "%*s%s %f, \n", 4*INDENT_SPACES, "",
            "\"dist_after_ransac\":",
            dist((*match_points_a_fixed)[i], (*match_points_b)[i]));
    // Emit first point
    fprintf(fp, "%*s%s\n", 4*INDENT_SPACES, "",
            "\"p1\": {");

    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "\"l\": [");
    fprintf(fp, "%*s%f, \n", 6*INDENT_SPACES, "",
            (*match_points_a)[i].x);
    fprintf(fp, "%*s%f\n", 6*INDENT_SPACES, "",
            (*match_points_a)[i].y);
    fprintf(fp, "%*s%s, \n", 5*INDENT_SPACES, "",
            "]");

    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "\"w\": [");
    fprintf(fp, "%*s%f, \n", 6*INDENT_SPACES, "",
            (*match_points_a)[i].x + a_data->x_start);
    fprintf(fp, "%*s%f\n", 6*INDENT_SPACES, "",
            (*match_points_a)[i].y + a_data->y_start);
    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "]");

    fprintf(fp, "%*s%s, \n", 4*INDENT_SPACES, "",
            "}");

    // Emit second point
    fprintf(fp, "%*s%s\n", 4*INDENT_SPACES, "",
            "\"p2\": {");

    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "\"l\": [");
    fprintf(fp, "%*s%f, \n", 6*INDENT_SPACES, "",
            (*match_points_b)[i].x);
    fprintf(fp, "%*s%f\n", 6*INDENT_SPACES, "",
            (*match_points_b)[i].y);
    fprintf(fp, "%*s%s, \n", 5*INDENT_SPACES, "",
            "]");

    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "\"w\": [");
    fprintf(fp, "%*s%f, \n", 6*INDENT_SPACES, "",
            (*match_points_b)[i].x + b_data->x_start);
    fprintf(fp, "%*s%f\n", 6*INDENT_SPACES, "",
            (*match_points_b)[i].y + b_data->y_start);
    fprintf(fp, "%*s%s\n", 5*INDENT_SPACES, "",
            "]");

    fprintf(fp, "%*s%s\n", 4*INDENT_SPACES, "",
            "}");

    fprintf(fp, "%*s%s", 3*INDENT_SPACES, "", "}");
    if (i+1 < num_matches)
      fprintf(fp, ", ");
    fprintf(fp, "\n");
  }
  fprintf(fp, "%*s%s, \n", 2*INDENT_SPACES, "",
          "]");

  // Output mipmapLevel
  fprintf(fp, "%*s%s: %d, \n", 2*INDENT_SPACES, "",
          "\"mipmapLevel\"", 0);

  // TODO(TB): Output model
  //
  // TB: This field does not seem to be used by the optimization step.

  // Output input file images
  fprintf(fp, "%*s%s: \"file://%s\", \n", 2*INDENT_SPACES, "",
          "\"url1\"", a_data->filepath);
  fprintf(fp, "%*s%s: \"file://%s\"\n", 2*INDENT_SPACES, "",
          "\"url2\"", b_data->filepath);

  // Output epilogue
  fprintf(fp, "%*s%s\n", INDENT_SPACES, "", "}");
  fprintf(fp, "]");

  fclose(fp);

  fasttime_t tend = gettime();
  totalTime += tdiff(tstart, tend);
  TRACE_1("save_tile_matches cumulative time: %.6lf [sec]\n",
          totalTime);

  TRACE_1("save_tile_matches: finish\n");
#endif
}



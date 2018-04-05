#!/bin/python
import sys
sys.path = ["/efs/home/wheatman/.local/lib/python2.7/site-packages"] + sys.path
import numpy
import pandas
# writes a c program to import all the data into a ParamDB

file_input = "11400_params.csv"
file_output = "make_paramsdb_gen.cpp"

# needed to calculate the accuracy
size_of_group = 1000

params_of_interest = ["num_features","num_octaves","scale"]

df = pandas.read_csv(file_input)

df2 = df.groupby(params_of_interest, as_index=False).agg({"number_correct":"sum", "time":"mean"})


str1 = ""

for row in df2.iterrows():
	row_data = str(row[1:]).replace("\n",",").replace("(","").replace(")","").replace("   "," ").split(",")
	line = """\tMRParams* mrp{0} = new MRParams;
\tmrp{0}->put_int_param("{1}", {2}); 
\tmrp{0}->put_int_param("{3}", {4}); 
\tmrp{0}->put_float_param("{5}", {6});
\tmrp{0}->set_accuracy({7});
\tmrp{0}->set_cost({8});
\tpdb->import_params(mrp{0});
""".format(row_data[5].split()[1], "num_features", row_data[0].split()[1].split(".")[0],
			"num_octaves", row_data[1].split()[1].split(".")[0],
			"scale", row_data[2].split()[1],
			float(row_data[3].split()[1])/size_of_group, row_data[4].split()[1]
			)
	str1+=line

code = """#include "paramdb.hpp"
#include "mrparams.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
namespace tfk {0}
void param_db_import(ParamDB* pdb){0}
{2}
{1}
{1}


""".format("{","}",str1)

with open(file_output, "w") as f:
	f.write(code)
import os
import sys
import AlignData_pb2
import math
import random
from copy import deepcopy
import argparse


parser = argparse.ArgumentParser(description='create a new proto by copying one section many times')
parser.add_argument("orig", help="The original proto")
parser.add_argument("sec_num",help="which section to copy")
parser.add_argument("copies", type=int, help="How many times to copy the data")
args = parser.parse_args()



pre_align_data = AlignData_pb2.AlignDataHierarchy()
copied_align_data = AlignData_pb2.AlignDataHierarchy()
pre_align_data.ParseFromString(open(args.orig).read())
section_number = int(args.sec_num)
section_location = pre_align_data.sec_data_location[section_number]
section = AlignData_pb2.SectionData()
section.ParseFromString(open(section_location).read())
copies = args.copies

output_directory = "copied_section_"+str(section_number)+"_"+str(copies)+"_times"
os.system("mkdir "+output_directory)

copied_align_data.n_sections = copies
for i in range(copies):
  section.section_id = i
  string = output_directory+"/section_"+str(i)+".pbuf"
  f = open(string, 'wb')
  f.write(section.SerializeToString())
  f.close()
  copied_align_data.sec_data_location.append(string)


copied_align_data.base_section=0
copied_align_data.n_sections=copies


f = open(output_directory+"/stack.pbuf", 'wb')
f.write(copied_align_data.SerializeToString())
f.close()





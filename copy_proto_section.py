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
parser.add_argument("copies", type=int, help="How many times to copy the data",)
args = parser.parse_args()

pre_align_data = AlignData_pb2.AlignData()
copied_align_data = AlignData_pb2.AlignData()
pre_align_data.ParseFromString(open(args.orig).read())
section_number = int(args.sec_num)
section = pre_align_data.sec_data[section_number]
copies = args.copies

copied_align_data.n_sections = copies
for i in range(copies):
  new_sec = copied_align_data.sec_data.add()
  new_sec.CopyFrom(section)
  copied_align_data.sec_data[i].section_id = i
copied_align_data.base_section=0


f = open("copied_section_"+str(section_number)+"_"+str(copies)+"_times.pbuf", 'wb')
f.write(copied_align_data.SerializeToString())
f.close()





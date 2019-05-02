import os
import sys
import json    
import AlignData_pb2


align_data = AlignData_pb2.AlignDataHierarchy()

align_data.ParseFromString(open(sys.argv[1]).read())

print align_data

for section in align_data.sec_data_location:
  sec_data = AlignData_pb2.SectionData()
  sec_data.ParseFromString(open(section).read())
  print sec_data

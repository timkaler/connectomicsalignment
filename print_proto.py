import os
import sys
import json    
import AlignData_pb2


align_data = AlignData_pb2.AlignData()
#
align_data.ParseFromString(open(sys.argv[1]).read())

print align_data

import os
import sys
import AlignData_pb2
import math
import random
from copy import deepcopy
import argparse
from statistics import mean

parser = argparse.ArgumentParser(description='makes a new proto with a smaller area from an old proto')
parser.add_argument("orig", help="The original proto which created the data")
parser.add_argument("sec_num",help="which section number to shrink")
parser.add_argument("radius", type=int, help="radius of the new section")
args = parser.parse_args()
x_size = 3128
y_size = 2724


pre_align_data = AlignData_pb2.AlignDataHierarchy()
small_align_data = AlignData_pb2.AlignDataHierarchy()
pre_align_data.ParseFromString(open(args.orig).read())
section_number = int(args.sec_num)
section_location = pre_align_data.sec_data_location[section_number]
section = AlignData_pb2.SectionData()
section.ParseFromString(open(section_location).read())
radius = args.radius

def get_distance(vec1, vec2):
  return math.sqrt((vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2 )

def get_center(tile1):
  tile1_x_start = tile1.x_start + tile1.offset_x
  tile1_x_finish = tile1_x_start + x_size
  tile1_y_start = tile1.y_start + tile1.offset_y
  tile1_y_finish = tile1_y_start + y_size
  return ((tile1_x_start + tile1_x_finish)/2, (tile1_y_start + tile1_y_finish)/2)
print "starting with ",len(section.tiles),"tiles"
f2 = open("shrinker_out.py", "w")
f2.write("#starting positions\n")
min_x = min([get_center(tile)[0] for tile in section.tiles])
min_y = min([get_center(tile)[1] for tile in section.tiles])
f2.write("a = ["+",".join([str(get_center(tile)[0] - min_x) for tile in section.tiles])+"]\n")
f2.write("b = ["+",".join([str(get_center(tile)[1] - min_y) for tile in section.tiles])+"]\n")

middle_x = mean([get_center(tile)[0] for tile in section.tiles])
middle_y = mean([get_center(tile)[1] for tile in section.tiles])
middle = (middle_x, middle_y)

good_tiles = set()
for i, tile in enumerate(section.tiles):
  if get_distance(get_center(tile), middle) < radius:
    good_tiles.add(i)


new_section = AlignData_pb2.SectionData()
new_section.section_id = 0
for i,tile in enumerate(section.tiles):
  if i in good_tiles:
    new_tile = new_section.tiles.add()
    new_tile.tile_id = tile.tile_id
    new_tile.tile_mfov = tile.tile_mfov
    new_tile.tile_index = tile.tile_index
    new_tile.section_id = 0
    new_tile.x_start = tile.x_start
    new_tile.x_finish = tile.x_finish
    new_tile.y_start = tile.y_start
    new_tile.y_finish = tile.y_finish
    new_tile.tile_filepath = tile.tile_filepath



output_directory ="small_proto_hierarchy_with_radius_"+str(radius)
os.system("mkdir "+output_directory)


small_align_data.n_sections = 1
small_align_data.base_section = 0
string = output_directory+"/section_0.pbuf"
new_section.n_tiles = len(new_section.tiles)

f = open(string, 'wb')
f.write(new_section.SerializeToString())
f.close()
small_align_data.sec_data_location.append(string)

print "ending with ",len(new_section.tiles),"tiles"
f2.write("#small proto\n")
f2.write("e = ["+",".join([str(get_center(tile)[0] - min_x) for tile in new_section.tiles])+"]\n")
f2.write("f = ["+",".join([str(get_center(tile)[1] - min_y) for tile in new_section.tiles])+"]\n")
f2.close()



f = open(output_directory+"/stack.pb", 'wb')
f.write(small_align_data.SerializeToString())
f.close()


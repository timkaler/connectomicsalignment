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
parser.add_argument("final",help="the output proto from the 2d alignment")
parser.add_argument("radius", type=int, help="radius of the new section")
args = parser.parse_args()
x_size = 3128
y_size = 2724


pre_align_data = AlignData_pb2.AlignData()
small_align_data = AlignData_pb2.AlignData()
post_align_data = AlignData_pb2.Saved2DAlignmentSection()
pre_align_data.ParseFromString(open(args.orig).read())
section_number = int(args.sec_num)
post_align_data.ParseFromString(open(args.final).read())
section = pre_align_data.sec_data[section_number]
radius = args.radius

if len(section.tiles) != len(post_align_data.tiles):
  print "their are different amounts of data the protos are probably wrong"
  exit()

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
f2.write("#ending positions\n")
min_x2 = min([get_center(tile)[0] for tile in post_align_data.tiles])
min_y2 = min([get_center(tile)[1] for tile in post_align_data.tiles])
f2.write("c = ["+",".join([str(get_center(tile)[0] - min_x2) for tile in post_align_data.tiles])+"]\n")
f2.write("d = ["+",".join([str(get_center(tile)[1] - min_y2) for tile in post_align_data.tiles])+"]\n")


middle_x = mean([get_center(tile)[0] for tile in post_align_data.tiles])
middle_y = mean([get_center(tile)[1] for tile in post_align_data.tiles])
middle = (middle_x, middle_y)

good_tiles = set()
for i, tile in enumerate(post_align_data.tiles):
  if get_distance(get_center(tile), middle) < radius:
    good_tiles.add(i)


new_section = small_align_data.sec_data.add()
new_section.section_id = section.section_id
for i,tile in enumerate(section.tiles):
  if i in good_tiles:
    new_tile = new_section.tiles.add()
    new_tile.tile_id = tile.tile_id
    new_tile.tile_mfov = tile.tile_mfov
    new_tile.tile_index = tile.tile_index
    new_tile.section_id = tile.section_id
    new_tile.x_start = tile.x_start
    new_tile.x_finish = tile.x_finish
    new_tile.y_start = tile.y_start
    new_tile.y_finish = tile.y_finish
    new_tile.tile_filepath = tile.tile_filepath
small_align_data.n_sections = 1
small_align_data.base_section = section.section_id

print "ending with ",len(new_section.tiles),"tiles"
f2.write("#small proto\n")
f2.write("e = ["+",".join([str(get_center(tile)[0] - min_x) for tile in small_align_data.sec_data[0].tiles])+"]\n")
f2.write("f = ["+",".join([str(get_center(tile)[1] - min_y) for tile in small_align_data.sec_data[0].tiles])+"]\n")
f2.close()
f = open("small_proto_with_radius_"+str(radius)+".pb", 'wb')
f.write(small_align_data.SerializeToString())
f.close()





import os
import sys
import AlignData_pb2
import math
import random
from copy import deepcopy
import argparse


parser = argparse.ArgumentParser(description='move the tiles around by small amounts')
parser.add_argument("orig", help="The original proto which created the data")
parser.add_argument("sec_num",help="which section to jiggle")
parser.add_argument("final",help="the output proto from the 2d alignment")
parser.add_argument("jiggle_amount", type=int, help="How much to try and jiggle by", default=10)
parser.add_argument("jiggle_iters", type=int, help="How many times to try and jiggle each tile", default=10)
args = parser.parse_args()
x_size = 3128
y_size = 2724

percent_cutoff_before = .01
percent_cutoff_after = .01
jiggle_amount = args.jiggle_amount
jiggle_iters = args.jiggle_iters


pre_align_data = AlignData_pb2.AlignData()
post_align_data = AlignData_pb2.Saved2DAlignmentSection()
jiggle_align_data = AlignData_pb2.AlignData()
pre_align_data.ParseFromString(open(args.orig).read())
section_number = int(args.sec_num)
post_align_data.ParseFromString(open(args.final).read())
section = pre_align_data.sec_data[section_number]

if len(section.tiles) != len(post_align_data.tiles):
  print "their are different amounts of data the protos are probably wrong"
  exit()

#SI = Max(0, Min(XA2, XB2) - Max(XA1, XB1)) * Max(0, Min(YA2, YB2) - Max(YA1, YB1))
def overlap(tile1, tile2):
  tile1_x_start = tile1.x_start + tile1.offset_x
  tile1_x_finish = tile1_x_start + x_size
  tile1_y_start = tile1.y_start + tile1.offset_y
  tile1_y_finish = tile1_y_start + y_size
  tile2_x_start = tile2.x_start + tile2.offset_x
  tile2_x_finish = tile2_x_start + x_size
  tile2_y_start = tile2.y_start + tile2.offset_y
  tile2_y_finish = tile2_y_start + y_size
  
  si = max(0, min(tile1_x_finish, tile2_x_finish) - max(tile1_x_start, tile2_x_start)) * max(0, min(tile1_y_finish, tile2_y_finish) - max(tile1_y_start, tile2_y_start))
  if si == 0:
    return 0
  su = 2 * x_size * y_size
  percent = float(si)/su
  return percent
def overlap2(tile1, tile2):
  tile1_x_start = tile1.x_start 
  tile1_x_finish = tile1_x_start + x_size
  tile1_y_start = tile1.y_start 
  tile1_y_finish = tile1_y_start + y_size
  tile2_x_start = tile2.x_start 
  tile2_x_finish = tile2_x_start + x_size
  tile2_y_start = tile2.y_start 
  tile2_y_finish = tile2_y_start + y_size
  
  si = max(0, min(tile1_x_finish, tile2_x_finish) - max(tile1_x_start, tile2_x_start)) * max(0, min(tile1_y_finish, tile2_y_finish) - max(tile1_y_start, tile2_y_start))
  if si == 0:
    return 0
  su = 2 * x_size * y_size
  percent = float(si)/su
  return percent

def get_distance(vec1, vec2):
  return math.sqrt((vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2 )

def get_vector(tile1, tile2):
  tile1_x_start = tile1.x_start
  tile1_y_start = tile1.y_start
  tile2_x_start = tile2.x_start
  tile2_y_start = tile2.y_start
  return (tile2_x_start - tile1_x_start, tile2_y_start - tile1_y_start)

overlaps = [0]*1000
tile_overlaps = {}
for i, tile1 in enumerate(post_align_data.tiles):
  tile_overlaps[i] = []
  for j, tile2 in enumerate(post_align_data.tiles):
    if i != j:
      overlap_percent = overlap(tile1, tile2)
      if overlap_percent > percent_cutoff_before:
        overlaps[int(overlap_percent*1000)]+=1
        tile_overlaps[i].append(j)
  if len(tile_overlaps[i]) == 0:
    del tile_overlaps[i]



print "there are ",len(post_align_data.tiles), "tiles"
print sum(overlaps), "pairs overlap"
for i in range(25):
  print i,"percent overlap",overlaps[10*i:10*i+10]


pair_vectors = {}
for tile_id, neighbor_ids in tile_overlaps.iteritems():
  tile1 = section.tiles[tile_id]
  for tile_id_2 in neighbor_ids:
    tile2 = section.tiles[tile_id_2]
    pair_vectors[(tile_id, tile_id_2)] = get_vector(tile1, tile2)



for i in range(jiggle_iters):
  tiles_jiggles = 0
  errors_acc = {}
  for tile_id, neighbor_ids in tile_overlaps.iteritems():
    tile1 = section.tiles[tile_id]
    proposed_tile = deepcopy(tile1)
    jiggle_x = random.randint(-jiggle_amount, jiggle_amount)
    jiggle_y = random.randint(-jiggle_amount, jiggle_amount)
    proposed_tile.x_start += jiggle_x
    proposed_tile.y_start += jiggle_y
    still_overlap = True
    for tile_id_2 in neighbor_ids:
      tile2 = section.tiles[tile_id_2]
      overlap_percent = overlap2(proposed_tile, tile2)
      original_overlap = overlap2(tile1, tile2)
      #print overlap_percent, original_overlap
      if overlap_percent < percent_cutoff_after:
        still_overlap = False
    if still_overlap:
      section.tiles[tile_id].x_start = proposed_tile.x_start
      section.tiles[tile_id].y_start = proposed_tile.y_start
      tiles_jiggles += 1 

  for tile_id, neighbor_ids in tile_overlaps.iteritems():
    tile1 = section.tiles[tile_id]
    for tile_id_2 in neighbor_ids:
      tile2 = section.tiles[tile_id_2]
      errors_acc[(tile_id, tile_id_2)] = get_distance(get_vector(tile1, tile2), pair_vectors[(tile_id, tile_id_2)])

  errors = [0]*1000
  for key,value in errors_acc.iteritems():
    if value > 999:
      value = 999
    errors[int(value)]+=1
  while errors[-1] == 0:
    errors.pop()
  print errors

  print tiles_jiggles, "tiles jiggled"

after_overlaps = [0]*1000
for i, tile1 in enumerate(section.tiles):
  for j, tile2 in enumerate(section.tiles):
    if i != j:
      overlap_percent = overlap2(tile1, tile2)
      if overlap_percent > 0:
        after_overlaps[int(overlap_percent*1000)]+=1

print sum(after_overlaps), "pairs overlap"
for i in range(25):
  print i,"percent overlap",after_overlaps[10*i:10*i+10]

f = open("jiggled_proto_by_"+str(jiggle_amount)+"_"+str(jiggle_iters)+"_times.pb", 'wb')
f.write(pre_align_data.SerializeToString())
f.close()





import os
import sys
import AlignData_pb2
import math


x_size = 3128
y_size = 2724

percent_cutoff = .008

post_align_data = AlignData_pb2.Saved2DAlignmentSection()
post_align_data.ParseFromString(open(sys.argv[1]).read())

section_num = int(sys.argv[2])

start_proto1 = AlignData_pb2.AlignData()
start_proto2 = AlignData_pb2.AlignData()
start_proto1.ParseFromString(open(sys.argv[3]).read())
start_proto2.ParseFromString(open(sys.argv[4]).read())
section1 = start_proto1.sec_data[section_num]
section2 = start_proto2.sec_data[section_num]

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

def get_vector(tile1, tile2):
  tile1_x_start = tile1.x_start + tile1.offset_x
  tile1_y_start = tile1.y_start + tile1.offset_y
  tile2_x_start = tile2.x_start + tile2.offset_x
  tile2_y_start = tile2.y_start + tile2.offset_y
  return (tile2_x_start - tile1_x_start, tile2_y_start - tile1_y_start)
def get_vector2(tile1, tile2):
  tile1_x_start = tile1.x_start
  tile1_y_start = tile1.y_start
  tile2_x_start = tile2.x_start
  tile2_y_start = tile2.y_start
  return (tile2_x_start - tile1_x_start, tile2_y_start - tile1_y_start)


def get_distance(vec1, vec2):
  return math.sqrt((vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2 )

if len(section1.tiles) != len(section2.tiles):
  print "the sections have different lengths, something is wrong"
  exit()


pairs_to_check = {}
for i, tile1 in enumerate(post_align_data.tiles):
  if not tile1.bad_2d_alignment:
    for j, tile2 in enumerate(post_align_data.tiles):
      if not tile2.bad_2d_alignment: 
        if i != j:
          overlap_percent = overlap(tile1, tile2)
          if overlap_percent > percent_cutoff:
            pairs_to_check[(i,j)] = get_vector(tile1, tile2)

print len(pairs_to_check), "pairs overlap"

errors = [0]*1000
for pair, vec in pairs_to_check.iteritems():
      vec1 = get_vector2(section1.tiles[pair[0]], section1.tiles[pair[1]])
      vec2 = get_vector2(section2.tiles[pair[0]], section2.tiles[pair[1]])
      distance = int(get_distance(vec1, vec2))
      if distance >= 1000:
        distance = 999
      errors[distance] +=1
while errors[-1] == 0:
  errors.pop()
print errors

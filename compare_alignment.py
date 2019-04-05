import os
import sys
import AlignData_pb2
import math


x_size = 3128
y_size = 2724

percent_cutoff = .008

post_align_data1 = AlignData_pb2.Saved2DAlignmentSection()
post_align_data2 = AlignData_pb2.Saved2DAlignmentSection()
post_align_data1.ParseFromString(open(sys.argv[1]).read())
post_align_data2.ParseFromString(open(sys.argv[2]).read())

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

def get_vector(tile1, tile2):
  tile1_x_start = tile1.x_start + tile1.offset_x
  tile1_y_start = tile1.y_start + tile1.offset_y
  tile2_x_start = tile2.x_start + tile2.offset_x
  tile2_y_start = tile2.y_start + tile2.offset_y
  return (tile2_x_start - tile1_x_start, tile2_y_start - tile1_y_start)


def get_distance(vec1, vec2):
  return math.sqrt((vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2 )

if len(post_align_data1.tiles) != len(post_align_data2.tiles):
  print "the sections have different lengths, something is wrong"
  exit()

bad_tiles = 0

pairs_to_check = set()
overlaps = [0]*1000
for i, tile1 in enumerate(post_align_data1.tiles):
  if tile1.bad_2d_alignment:
    bad_tiles +=1
  for j, tile2 in enumerate(post_align_data1.tiles):
    if i != j:
      overlap_percent = overlap(tile1, tile2)
      if overlap_percent > percent_cutoff:
        overlaps[int(overlap_percent*1000)]+=1
        pairs_to_check.add((i,j))

overlaps2 = [0]*1000
for i, tile1 in enumerate(post_align_data2.tiles):
  for j, tile2 in enumerate(post_align_data2.tiles):
    if i != j:
      overlap_percent = overlap(tile1, tile2)
      if overlap_percent > percent_cutoff:
        overlaps2[int(overlap_percent*1000)]+=1
        pairs_to_check.add((i,j))


print sum(overlaps), "pairs overlap in the first one"
#for i in range(25):
#  print i,"percent overlap",overlaps[10*i:10*i+10]
print sum(overlaps2), "pairs overlap in the second one"
#for i in range(25):
#  print i,"percent overlap",overlaps2[10*i:10*i+10]

print "in total",len(pairs_to_check),"paris overlap"

errors = [0]*1000
for pair in pairs_to_check:
  if post_align_data1.tiles[pair[0]].bad_2d_alignment == False and post_align_data1.tiles[pair[1]].bad_2d_alignment == False:
    if post_align_data2.tiles[pair[0]].bad_2d_alignment == False and post_align_data2.tiles[pair[1]].bad_2d_alignment == False:
      vec1 = get_vector(post_align_data1.tiles[pair[0]], post_align_data1.tiles[pair[1]])
      vec2 = get_vector(post_align_data2.tiles[pair[0]], post_align_data2.tiles[pair[1]])
      distance = int(get_distance(vec1, vec2))
      if distance >= 1000:
        distance = 999
        #print vec1, vec2
        #print post_align_data1.tiles[pair[0]]
        #print post_align_data1.tiles[pair[1]]
      errors[distance] +=1
while errors[-1] == 0:
  errors.pop()
print errors
print bad_tiles,"bad tiles"

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
#def overlap(tile1, tile2):
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

  

def get_overlapping_tile_pairs(tiles):
  unsorted_list = []
  for i, t in enumerate(tiles):
    unsorted_list.append((t.x_start + t.offset_x, i))

  sorted_list = sorted(unsorted_list)
  pairs_to_check = []
  for i in range(0,len(sorted_list)):
    #tile = sorted_list[i][2]
    overlaps_x = []
    j = i
    while j >= 0 and sorted_list[j][0] + 5000 > sorted_list[i][0]: #sorted_list[j].x_finish + sorted_list[j].offset_x + 10.0 > tile.x_start + tile.offset_x:
      overlaps_x.append(sorted_list[j][1])
      j -= 1
    j = i+1
    while j < len(sorted_list) and sorted_list[j][0] - 5000 < sorted_list[i][0]:
      overlaps_x.append(sorted_list[j][1])
      j += 1

    #print str(i)
    #print "Done with getting potential overlaps " + str(len(overlaps_x))

    tile1 = tiles[sorted_list[i][1]]
    for index in overlaps_x:
      tile2 = tiles[index]
      if index != sorted_list[i][1]:
        overlap_percent = overlap(tile1, tile2)
        if overlap_percent > percent_cutoff:
          pairs_to_check.append((sorted_list[i][1], index)) 
  return pairs_to_check 



pairs_to_check = set()
overlaps = [0]*1000



#for i, tile1 in enumerate(post_align_data1.tiles):
#for i in range(len(post_align_data1.tiles)):
#  tile1 = post_align_data1.tiles[i]
#  #for j, tile2 in enumerate(post_align_data1.tiles):
#  for j in range(len(post_align_data1.tiles)):
#    tile2 = post_align_data1.tiles[j]
#    if i != j:
#      overlap_percent = overlap(tile1, tile2)
#      if overlap_percent > percent_cutoff:
#        overlaps[int(overlap_percent*1000)]+=1
#        pairs_to_check.add((i,j))

items1 = get_overlapping_tile_pairs(post_align_data1.tiles)
items2 = get_overlapping_tile_pairs(post_align_data2.tiles) 

for x in items1:
  pairs_to_check.add(x)
for x in items2:
  pairs_to_check.add(x)

overlaps2 = [0]*1000


#for i, tile1 in enumerate(post_align_data2.tiles):
#  for j, tile2 in enumerate(post_align_data2.tiles):
#for i in range(len(post_align_data2.tiles)):
#  tile1 = post_align_data2.tiles[i]
#  #for j, tile2 in enumerate(post_align_data1.tiles):
#  for j in range(len(post_align_data2.tiles)):
#    tile2 = post_align_data2.tiles[j]
#
#    if i != j:
#      overlap_percent = overlap(tile1, tile2)
#      if overlap_percent > percent_cutoff:
#        overlaps2[int(overlap_percent*1000)]+=1
#        pairs_to_check.add((i,j))


print sum(overlaps), "pairs overlap in the first one"
#for i in range(25):
#  print i,"percent overlap",overlaps[10*i:10*i+10]
print sum(overlaps2), "pairs overlap in the second one"
#for i in range(25):
#  print i,"percent overlap",overlaps2[10*i:10*i+10]

print "in total",len(pairs_to_check),"paris overlap"

errors = [0]*50
for pair in pairs_to_check:
  if post_align_data1.tiles[pair[0]].bad_2d_alignment == False and post_align_data1.tiles[pair[1]].bad_2d_alignment == False:
    if (post_align_data2.tiles[pair[0]].bad_2d_alignment == False and post_align_data2.tiles[pair[1]].bad_2d_alignment == False) or True:
      vec1 = get_vector(post_align_data1.tiles[pair[0]], post_align_data1.tiles[pair[1]])
      vec2 = get_vector(post_align_data2.tiles[pair[0]], post_align_data2.tiles[pair[1]])
      distance = int(get_distance(vec1, vec2))
      if distance >= 50:
        distance = 49
        #print vec1, vec2
        #print post_align_data1.tiles[pair[0]]
        #print post_align_data1.tiles[pair[1]]
      errors[distance] +=1
print errors

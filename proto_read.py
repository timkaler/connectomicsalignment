import os
import sys
import json    
import AlignData_pb2


align_data = AlignData_pb2.AlignData()
#
align_data.ParseFromString(open('data/proto_data21').read())


## removes a section and then renumbers the section ids.
def remove_section(section_id):
  align_data.sec_data.pop(section_id)
  #align_data.sec_data[section_id].CopyFrom(align_data.sec_data[section_id+1])
  #align_data.sec_data.insert(section_id, align_data.sec_data[section_id]) #= align_data.sec_data[section_id+1].copy()#.pop(section_id)

  #for id in range(section_id, len(align_data.sec_data)):
  #  align_data.sec_data[id].section_id -= 1
  #  for tile in align_data.sec_data[id].tiles:
  #    tile.section_id -= 1



def get_bad_sections():
  delete_section_ids = []
  lines = open("badtriangles.txt").readlines()
  for l in lines:
    section_id = int(l.split(' ')[0].strip())#l[l.find("section "):l.find(":")]
    #section_id = int(section_id.replace("section ", "").strip())
    #print section_id
    bad_count = l[l.find("size "):].replace("size ", "").strip()
    bad_count = int(bad_count)
    #print "bad count is " + str(bad_count)
    if bad_count > 200:
      delete_section_ids.append(section_id)
  return delete_section_ids

def find_sections_to_delete(bad_sections):
  if len(bad_sections) == 0:
    print "NO bad sections"
  bad_sections = sorted(bad_sections)[::-1]
  last_section_id = bad_sections[0]
  delete_id = None
  delete_list = []
  for i in range(1, len(bad_sections)):
    if last_section_id != bad_sections[i] + 1:
      if delete_id != None:
        delete_list.append(delete_id)
      delete_id = None
    else:
      delete_id = bad_sections[i]
    last_section_id = bad_sections[i]
  if delete_id != None:
    delete_list.append(delete_id)
  return delete_list


bad_sections = get_bad_sections()
print bad_sections
sections_to_delete = find_sections_to_delete(bad_sections)
print sections_to_delete

##align_data.sec_data.pop(4)
for x in sections_to_delete:
  remove_section(x)
#
with open ('data/proto_data_delete3', 'wb') as f:
  f.write(align_data.SerializeToString())

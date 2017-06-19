import os
import json

for dirname,dirnames,filenames in os.walk('/efs/worm_dataset'):
  section = None
  try:
    section = int(dirname.split('/')[-1])
    print section
    tile_list = []
    i = 0
    for filename in filenames:
      tile = dict()
      start_x = float(filename.split('_')[0])
      start_y = float(filename.split('_')[1])
      end_x = start_x+float(filename.split('_')[2])
      end_y = start_y+float(filename.split('_')[3])
      tile["bbox"] = [start_x,end_x,start_y,end_y]
      tile["height"] = int(filename.split('_')[3])
      tile["width"] = int(filename.split('_')[2])
      tile["maxIntensity"] = 255.0
      tile["mfov"] = 1
      tile["minIntensity"] = 0.0
      tile["mipmapLevels"] = dict()
      tile["mipmapLevels"][0] = dict()
      tile["mipmapLevels"][0]["imageUrl"] = dirname+"/"+filename
      tile["tile_index"] = i+1
      tile["layer"] = section+1
      i+=1
      tile_list.append(tile)
    open("worm_tilespecs/W01_Sec"+str(section)+".json", 'w').write(json.dumps(tile_list))

  except:
    print "Exception"
    pass
  #print filenames


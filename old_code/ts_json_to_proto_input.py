import os
import sys
import json    
import AlignData_pb2

TILES_TO_PROCESS = {
    1 : {
        1: [ i + 1 for i in xrange(0,5) ],  },
        
    2 : {
        1: [ i + 1 for i in xrange(0,5) ],  },
    
}

MAGIC_STR_TOTAL_TILES = 'total-tiles:'
MAGIC_STR_TILE_START = 'tile-start:'
MAGIC_STR_TILE_END = 'tile-end:'

MAGIC_STR_TILE_SECTION = 'tile-section:'
MAGIC_STR_TILE_MFOV = 'tile-mfov:'
MAGIC_STR_TILE_INDEX = 'tile-index:'
MAGIC_STR_TILE_BBOX = 'tile-bbox:'
MAGIC_STR_TILE_FILEPATH = 'tile-filepath:'

def filter_tilespecs(tilespecs_json_dir, tiles_to_process):
    
    json_filenames = os.listdir(tilespecs_json_dir)
    print str(json_filenames) 
    tilespecs_list = []
    for json_filename in json_filenames:
        json_filepath = os.path.join(tilespecs_json_dir, json_filename)
        print 'Read: %s' % (json_filepath,)
        f = open(json_filepath, 'rb')
        tilespecs = json.load(f)
        f.close()
        
        tilespecs_list.append(tilespecs)
        
    filtered_tilespecs = []
    min_x = 100000000.0
    min_y = 100000000.0
    for tilespecs in tilespecs_list:
        for ts in tilespecs:
          bbox = ts['bbox']
          if bbox[0]<min_x:
            min_x = bbox[0]
          if bbox[1]<min_x:
            min_x = bbox[1]
          if bbox[2]<min_y:
            min_y = bbox[2]
          if bbox[3]<min_y:
            min_y = bbox[3]

    for tilespecs in tilespecs_list:  
        for ts in tilespecs:
            ts['bbox'][0] -= min_x
            ts['bbox'][1] -= min_x
            ts['bbox'][2] -= min_y
            ts['bbox'][3] -= min_y
            cur_section = ts['layer']
            cur_mfov = ts['mfov']
            #if cur_mfov != 1:
            #  continue
            cur_index = ts['tile_index']
            #if ts['bbox'][0] > 30000 or ts['bbox'][1] > 30000 or ts['bbox'][2] > 30000 or ts['bbox'][3]>30000:
            #  continue
            #print ts
            #if ((cur_section in tiles_to_process.keys()) and 
            #    (cur_mfov in tiles_to_process[cur_section].keys()) and 
            #    (cur_index in tiles_to_process[cur_section][cur_mfov])):
            if True:
                print 'Add tile: %d %d %d' % (cur_section, cur_mfov, cur_index)
                filtered_tilespecs.append(ts)
    
    return filtered_tilespecs
    
def write_txt_input(output_filename, tilespecs):
    
    #txt_data = ''
    #txt_data += '%s %d\n' % (MAGIC_STR_TOTAL_TILES, len(tilespecs))

    sections = {}
         
    for i, ts in enumerate(tilespecs):
        #tile_data = ''
        #
        #tile_data += '%s [%d]\n' % (MAGIC_STR_TILE_START, i)
        #
        #tile_data += '\t%s %d\n' % (MAGIC_STR_TILE_SECTION, ts['layer'])
        #tile_data += '\t%s %d\n' % (MAGIC_STR_TILE_MFOV, ts['mfov'])
        #tile_data += '\t%s %d\n' % (MAGIC_STR_TILE_INDEX, ts['tile_index'])        

        #(x_start, x_finish, y_start, y_finish) = ts['bbox']
        #tile_data += '\t%s [%d][%d][%d][%d]\n' % (MAGIC_STR_TILE_BBOX, 
        #    x_start, x_finish, y_start, y_finish)
                
        #tile_path = ts["mipmapLevels"]["0"]["imageUrl"]
        #tile_path = tile_path.replace('file://', '')
        #tile_data += '\t%s %s\n' % (MAGIC_STR_TILE_FILEPATH, tile_path)
        
        #tile_data += '%s [%d]\n' % (MAGIC_STR_TILE_END, i)
        
        #txt_data += tile_data
        ts["id"] = i

        if ts['layer'] not in sections:
            sections[ts['layer']] = []
        sections[ts['layer']].append(ts)
    align_data_proto = AlignData_pb2.AlignData()

    sorted_section = sorted(sections.keys())
    the_count = 0
    for i, sec in enumerate(sorted_section):
        print sec
        
        section_proto = align_data_proto.sec_data.add()
        section_proto.section_id = i
        for j, ts in enumerate(sections[sec]):
            the_count += 1
            tile_proto = section_proto.tiles.add()
            tile_proto.tile_id = ts["id"]
            tile_proto.tile_mfov = ts["mfov"]
            tile_proto.tile_index = ts["tile_index"]
            tile_proto.section_id = section_proto.section_id
            tile_proto.x_start,tile_proto.x_finish, tile_proto.y_start, tile_proto.y_finish   = [int(item) for item in ts["bbox"]]
            tile_filepath = ts["mipmapLevels"]["0"]["imageUrl"]
            if j==0:
              print tile_filepath
            tile_proto.tile_filepath = tile_filepath.replace('file://', '')
            
        

    
    print '  - write: %s' % (output_filename,)
    f = open(output_filename, 'wb')
    f.write(align_data_proto.SerializeToString())
    f.close()
    print "The count is " + str(the_count)
    
def execute(tilespec_json_dir, output_filename):
    
    print '- filter_tilespecs:'
    
    tilespecs = filter_tilespecs(tilespec_json_dir, TILES_TO_PROCESS)
    n_tilespecs = len(tilespecs)
    print '  -- n_tilespecs: %d' % (n_tilespecs,)
    
    print '- write txt input:'
    write_txt_input(output_filename, tilespecs)
    

if '__main__' == __name__:
    try:
        prog_name, tilespec_json_dir, output_filename = sys.argv[:3]

    except ValueError, e:
        sys.exit('USAGE: %s [tilespec_json_dir] [output_filename]' % (sys.argv[0],))
    
    execute(tilespec_json_dir, output_filename)

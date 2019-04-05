import os
import sys
import AlignData_pb2

coordinates_file_names = ["image_coordinates.txt", "full_image_coordinates.txt"]


# TODO get correct numbers
image_width = 3128
image_height = 2724





def find_coordinates(stack_directory):
    min_x = 0
    min_y = 0
    coords = {}
    for root, dirs, files in os.walk(stack_directory):
        print root
        for fil in files:
            if fil in coordinates_file_names:
                with open(os.path.join(root,fil), "r") as f:
                    for line in f:
                        loc, x, y, trash =  line.rstrip().split("\t")
                        loc = root + "/" + loc.replace("\\","/")
                        x,y  = (float(x),float(y))
                        if x < min_x:
                            min_x = int(x)-1
                        if y < min_y:
                            min_y = int(y)-1
                        coords[loc] = [int(x),int(y)]
    return coords, min_x, min_y



def create_proto(coordinates, min_x, min_y):
    align_data_proto = AlignData_pb2.AlignData()
    section_protos = {}
    tile_id = 0
    for key,value in coordinates.iteritems():
        filename = key.split("/")[-1]
        try:
            section_id, mfov_id, tile_index, junk = filename.split("_")
            section_id = int(section_id)
            tile_id = int(tile_id)
            mfov_id = int(mfov_id)
            tile_index = int(tile_index)
            if section_id not in section_protos:
                section_protos[section_id] = align_data_proto.sec_data.add()
                section_protos[section_id].section_id = section_id
            tile_proto = section_protos[section_id].tiles.add()
            tile_proto.tile_id = tile_id
            tile_proto.tile_mfov = mfov_id
            tile_proto.tile_index = tile_index
            tile_proto.section_id = section_id
            #TODO check this line
            value[0] -= min_x
            value[1] -= min_y
            tile_proto.x_start = value[0]
            tile_proto.x_finish = value[0]+image_width
            tile_proto.y_start = value[1]
            tile_proto.y_finish = value[1]+image_height
            tile_proto.tile_filepath = key
            tile_id +=1
        except ValueError as e:
            print "bad format", filename
            del section_protos[section_id].tiles[-1]
    for section, proto in section_protos.iteritems():
        print "in section", section,"we have",len(proto.tiles)
        proto.n_tiles = len(proto.tiles)
    align_data_proto.n_sections = len(section_protos)
    align_data_proto.sec_data.sort(key = lambda x: x.section_id)
    align_data_proto.base_section = min(section_protos.keys())
    return align_data_proto

def execute(stack_directory, output_filename):
    
    print 'getting image coordinates'


    
    coordinates, min_x, min_y = find_coordinates(stack_directory)
    print "we got coords for "+str(len(coordinates))+" pictures"


    proto = create_proto(coordinates, min_x, min_y)
    f = open(output_filename, 'wb')
    f.write(proto.SerializeToString())
    f.close()

    

if '__main__' == __name__:
    try:
        prog_name, stack_directory, output_filename = sys.argv

    except ValueError, e:
        print sys.argv
        sys.exit('USAGE: %s [stack_directory]  [output_filename]' % (sys.argv[0],))
    
    execute(stack_directory, output_filename)

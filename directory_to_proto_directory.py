import os
import sys
import AlignData_pb2
import argparse

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



def create_protos(coordinates, min_x, min_y, output_directory):
    align_data_proto = AlignData_pb2.AlignDataHierarchy()
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
                section_protos[section_id] = AlignData_pb2.SectionData()
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
    

    #align_data_proto.sec_data.sort(key = lambda x: x.section_id)
    section_protos_list = list(section_protos.itervalues())
    section_protos_list.sort(key = lambda x: x.section_id)
    for item in section_protos_list:
        string = output_directory+"section_"+str(item.section_id)+".pbuf"
        f = open(string, 'wb')
        f.write(item.SerializeToString())
        f.close()
        align_data_proto.sec_data_location.append(string)


    align_data_proto.n_sections = len(section_protos)
    align_data_proto.base_section = min(section_protos.keys())

    f = open(output_directory+"stack.pbuf", 'wb')
    f.write(align_data_proto.SerializeToString())
    f.close()

def execute(stack_directory, output_directory):
    
    print 'getting image coordinates'


    
    coordinates, min_x, min_y = find_coordinates(stack_directory)
    print "we got coords for "+str(len(coordinates))+" pictures"


    create_protos(coordinates, min_x, min_y, output_directory)

    

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='Create a set of input protos by recursively searching through the given directorys')
    parser.add_argument("data", help="The location to start searching for the data")
    parser.add_argument("output", help="The location to place the input protots once they are completed")
    args = parser.parse_args()
    
    execute(args.data, args.output)

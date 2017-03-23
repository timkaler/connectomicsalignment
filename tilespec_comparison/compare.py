import json
import math
from rh_renderer.models import RigidModel
from rh_renderer.models import AffineModel
import numpy as np
import sys as Sys
import os
import argparse


TEST_OFFSETS = [0.0, 0.0, 0.0]
np.random.seed(0)
def run(old_directory, new_directory):

  filtered_files = []

  for x in os.listdir(old_directory):
    if x.endswith('.json'):
       filtered_files.append(x)

  for sec in sorted(filtered_files):
    print sec
    info = json.loads(open(new_directory+str(sec)).read())
    tiles = dict()
    for x in info:
      tiles[(x["mfov"],x["tile_index"])] = x
    
    info = json.loads(open(old_directory+str(sec)).read())
    tiles2 = dict()
    for x in info:
      tiles2[(x["mfov"],x["tile_index"])] = x
    
    t = 1000000.0
    worst_seen = 0.0
    
    def compute_val_sum(offsets):
      s = 0.0    
      for x in tiles.keys():
         data1 = tiles[x]["transforms"][0]["dataString"].split(" ")
         data2 = tiles2[x]["transforms"][0]["dataString"].split(" ")
         m1 = AffineModel()
         m1.set_from_modelspec(tiles[x]["transforms"][0]["dataString"])
         m2 = AffineModel()
         m2.set_from_modelspec(tiles2[x]["transforms"][0]["dataString"])
         #m1 = RigidModel(np.float64(data1[0])+(np.float64(offsets[0]))+TEST_OFFSETS[0], np.array([np.float64(data1[1])+offsets[1]+TEST_OFFSETS[1],np.float64(data1[2]) + offsets[2] +TEST_OFFSETS[2]])) 
         #m2 = RigidModel(np.float64(data2[0]), np.array([np.float64(data2[1]),np.float64(data2[2])]))
         mr1 = list(m1.apply(np.array([np.float64(1.0),np.float64(1.0)]))[0,])
         mr2 = list(m2.apply(np.array([np.float64(1.0),np.float64(1.0)]))[0,])
         s += round(abs(mr1[0]-mr2[0])) + round(abs(mr1[1]-mr2[1]))
         if round(abs(mr1[0]-mr2[0])) + round(abs(mr1[1]-mr2[1])) > 20.0:
           print "Big error. " + str(x) + " " + str(round(abs(mr1[0]-mr2[0])) + round(abs(mr1[1]-mr2[1])))
           print tiles[x]
           print ""
      return s 
    
    
    def compute_val_max(offsets):
      s = 0.0    
      for x in tiles.keys():
         data1 = tiles[x]["transforms"][0]["dataString"].split(" ")
         data2 = tiles2[x]["transforms"][0]["dataString"].split(" ")
         m1 = AffineModel()
         m1.set_from_modelspec(tiles[x]["transforms"][0]["dataString"])
         m2 = AffineModel()
         m2.set_from_modelspec(tiles2[x]["transforms"][0]["dataString"])
         #m1 = RigidModel(np.float64(data1[0])+(np.float64(offsets[0]))+np.float64(TEST_OFFSETS[0]), np.array([np.float64(data1[1])+offsets[1] + TEST_OFFSETS[1],np.float64(data1[2]) + offsets[2] +TEST_OFFSETS[2]])) 
         #m2 = RigidModel(np.float64(data2[0]), np.array([np.float64(data2[1]),np.float64(data2[2])]))
         #mr1 = list(m1.apply(np.array([0.0,0.0]))[0,])
         #mr2 = list(m2.apply(np.array([0.0,0.0]))[0,])
         mr1 = list(m1.apply(np.array([np.float64(1.0),np.float64(1.0)]))[0,])
         mr2 = list(m2.apply(np.array([np.float64(1.0),np.float64(1.0)]))[0,])
         s = max(s, abs(mr1[0]-mr2[0]))
         s = max(s, abs(mr1[1]-mr2[1]))
      return s 
    
    
    def compute_val(offsets):
      s = np.float64(0.0)    
      for x in tiles.keys():
         data1 = tiles[x]["transforms"][0]["dataString"].split(" ")
         data2 = tiles2[x]["transforms"][0]["dataString"].split(" ")
         #m1 = RigidModel(np.float64(data1[0])+(np.float64(offsets[0]))+np.float64(TEST_OFFSETS[0]), np.array([np.float64(data1[1])+TEST_OFFSETS[1]+offsets[1],np.float64(data1[2]) +TEST_OFFSETS[2]+ offsets[2]])) 
         m1 = AffineModel()
         m1.set_from_modelspec(tiles[x]["transforms"][0]["dataString"])
         m2 = AffineModel()
         m2.set_from_modelspec(tiles2[x]["transforms"][0]["dataString"])
         #mr1 = list(m1.apply(np.array([0.0,0.0]))[0,])
         #mr2 = list(m2.apply(np.array([0.0,0.0]))[0,])
         #print m1.to_modelspec()
         #quit()
         #m2 = RigidModel(np.float64(data2[0]), np.array([np.float64(data2[1]),np.float64(data2[2])]))
         mr1 = list(m1.apply(np.array([np.float64(1.0),np.float64(1.0)]))[0,])
         mr2 = list(m2.apply(np.array([np.float64(1.0),np.float64(1.0)]))[0,])
         s += np.abs((mr1[0]-mr2[0]))**2 + np.abs((mr1[1]-mr2[1]))**2
      return s 
    
    def compute_gradient(offsets, delta):
      grad = [np.float64(0.0),np.float64(0.0),np.float64(0.0)]
      epsilon = delta
      for i in range(0, len(offsets)):
        mul = 1.0
        if i==0:
          mul = 1e-1
        test1 = [offsets[j] for j in range(0,len(offsets))]
        test1[i] = test1[i]+epsilon
    
        test2 = [offsets[j] for j in range(0,len(offsets))]
        test2[i] = test2[i]-epsilon
    
        grad[i] = ((compute_val(test1) - compute_val(test2))*0.5*-1*delta*mul)
      return grad 
    
    offsets = [np.float64(0.0),np.float64(0.0),np.float64(0.0)]
    delta = np.float64(2.0)
    print ""
    best_value = compute_val(offsets)
    bad_iteration_count = 0
    while delta > 1e-10:
      grad = compute_gradient(offsets, delta)
      curr_val = compute_val(offsets)
      next_val = compute_val([offsets[i]+grad[i] for i in range(0,len(grad))])
      if best_value <= next_val+1e-4:
        bad_iteration_count += 1
      else:
        bad_iteration_count = 0
      if bad_iteration_count > 100:
        break
      if best_value <= next_val:
        delta = delta*np.random.uniform(0.8,0.95)
        Sys.stderr.write("\rSum Squared error:"+str(best_value)+"\t Delta-step:" + str(delta))
        continue
      else:
        delta = delta*np.random.uniform(1.0,1.1)
        best_value = next_val
        Sys.stderr.write("\rSum Squared error:"+str(best_value)+"\t Delta-step:" + str(delta))
        offsets = [offsets[i]+grad[i] for i in range(0,len(grad))]
    print ""
  
    print "Original error was " + str(compute_val_max([0.0,0.0,0.0]))
    print "Max error is " + str(compute_val_max(offsets))
    print "original summed error is " + str(compute_val([0.0,0.0,0.0]))
    print offsets
    print "summed error is " + str(compute_val(offsets))
    print "Finding badly aligned tiles" 
    compute_val_sum(offsets);
    print "Done"

###############################
# Driver
###############################
if __name__ == '__main__':
    # Command line parser
    parser = argparse.ArgumentParser(description='Analyzes alignments generated from two sets of tilespecs.')
    parser.add_argument('-g', '--old', type=str, 
                        help='a directory that contains tilespecs for the "ground truth" alignment.')
    parser.add_argument('-n', '--new', type=str, 
                        help='a directory containing the "new" tilespecs to compare against ground truth.')
    args = parser.parse_args() 


    if args.old == None or args.new == None:
       print "Error, must specify both --old and --new tliespec directories."
       quit()
    else:
       run(args.old, args.new)





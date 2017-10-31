from __future__ import print_function
import os
import sys
import json
import AlignData_pb2
from multiprocessing import Pool


import logging
import io

import cv2
import numpy as np


from google.protobuf.json_format import MessageToJson

import pyspark as spark
from pyspark import SparkContext
from pyspark.sql.types import StructType, StructField, IntegralType, StringType
from pyspark.sql import SQLContext, Row
def extract_sift():
    def extract_sift_nested(tile):
        try:
            image_path = tile.filepath
            img = cv2.imread(image_path,cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
            detector = cv2.FeatureDetector_create("SIFT")
            descriptor = cv2.DescriptorExtractor_create("SIFT")
            #print("Detecting keypoints...")
            kp = detector.detect(img)
            #print("Computing descriptions...")
            kp, descs = descriptor.compute(img,kp) 
            kp = [point.pt for point in kp]


            return Row(tile_id=tile.tile_id,
                            x_start=tile.x_start,
                            x_finish=tile.x_finish,
                            y_start=tile.y_start,
                            y_finish=tile.y_finish,
                            kps=kp,
                            desc=descs)
        except Exception, e:
            logging.exception(e)
            return []
    return extract_sift_nested

def overlap():
    def overlap_nested(pair):
        tile_1 = pair[0]
        tile_2 = pair[1]
        if tile_1.x_finish < tile_2.x_start or tile_2.x_finish < tile_2.x_start:
            return False
        if tile_1.y_finish < tile_2.y_start or tile_2.y_finish < tile_2.y_start:
            return False 
        return True
        
    return overlap_nested


def nearest_neighbor():
    def nearest_neighbor_nested(pair):
        tile_1 = pair[0]
        tile_2 = pair[1]
        # FLANN 
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(tile_1.desc,tile_2.desc,k=2)

        # ratio test for matches
        matches_for_pair = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                #just remember the pt locations for the pair
                matches_for_pair.append((tile_1.kps[m.queryIdx],tile_2.kps[n.trainIdx]))

        return Row(tile_1_id=tile_1.tile_id,
                            x_start_1=tile_1.x_start,
                            x_finish_1=tile_1.x_finish,
                            y_start_1=tile_1.y_start,
                            y_finish_1=tile_1.y_finish,
                            tile_2_id=tile_2.tile_id,
                            x_start_2=tile_2.x_start,
                            x_finish_2=tile_2.x_finish,
                            y_start_2=tile_2.y_start,
                            y_finish_2=tile_2.y_finish,
                            kps=matches_for_pair)
    return nearest_neighbor_nested

def simple_ransac(_thresh):
    def simple_ransac_nested(pair):
        best_dx = 0.0
        best_dy=0.0
        maxInliers = 0
        thresh = _thresh
        for point_1, point_2 in pair.kps:
            dx = point_2[0] - point_1[0]
            dy = point_2[1] - point_1[1]
            inliers = 0
            for other_point_1, other_point_2 in pair.kps:
                other_dx = other_point_2[0] - other_point_1[0] - dx
                other_dy = other_point_2[1] - other_point_1[1] - dy
                dist = other_dx*other_dx+other_dy*other_dy
                if dist < thresh*thresh:
                    inliers+=1
            if inliers > maxInliers:
                maxInliers = inliers
                best_dx = dx
                best_dy = dy
        filtered_matches = []
        for point_1, point_2 in pair.kps:
            dx = point_2[0] - point_1[0] - best_dx
            dy = point_2[1] - point_1[1] - best_dy
            dist = dx*dx+dy*dy
            if dist <= thresh*thresh:
                filtered_matches.append((point_1, point_2))
        return Row(tile_1_id=pair.tile_1_id,
                            x_start_1=pair.x_start_1,
                            x_finish_1=pair.x_finish_1,
                            y_start_1=pair.y_start_1,
                            y_finish_1=pair.y_finish_1,
                            tile_2_id=pair.tile_2_id,
                            x_start_2=pair.x_start_2,
                            x_finish_2=pair.x_finish_2,
                            y_start_2=pair.y_start_2,
                            y_finish_2=pair.y_finish_2,
                            matches=filtered_matches)
    return simple_ransac_nested

if __name__ == "__main__":
    align_data = AlignData_pb2.AlignData()
    align_data.ParseFromString(open('data/proto_data').read())
    section = align_data.sec_data[0]

    
    sc = SparkContext(appName="feature_extractor")
    log4jLogger = sc._jvm.org.apache.log4j
    LOGGER = log4jLogger.LogManager.getLogger(__name__)
    LOGGER.info("pyspark script logger initialized")
    sqlContext = SQLContext(sc)


    tiles = sc.parallelize([Row(tile_id=tile.tile_id,
                            x_start=tile.x_start,
                            x_finish=tile.x_finish,
                            y_start=tile.y_start,
                            y_finish=tile.y_finish,
                            filepath=tile.tile_filepath,
                            ) for tile in section.tiles])
    

    features = tiles.map(extract_sift()).filter(lambda x: x != [])
    pairs = features.cartesian(features)
    overlapping_pairs = pairs.filter(overlap())
    matches = overlapping_pairs.map(nearest_neighbor())
    thresh = 5.0
    filtered_matches = matches.map(simple_ransac(thresh)).filter(lambda x: x.matches != [])
    filtered_matches.foreach(print)
    #features = features.map(lambda x: (Row(fileName=x[0], features=x[1].tolist())))
    




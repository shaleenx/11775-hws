#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
import time
from joblib import Parallel, delayed


def get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    keyframe_iterator = get_keyframes(downsampled_video_filename, keyframe_interval)
    feature_vectors = np.matr


    pass

def get_keyframes(surf, downsampled_video_filename, keyframe_interval): 
    "Generator function which returns the next keyframe."         

    # Create video capture object                                 
    video_cap = cv2.VideoCapture(downsampled_video_filename)      
    frame = 0                                                     
    while True:                                                   
        frame += 1                                                
        ret, img = video_cap.read()                               
        if ret is False:                                          
            break                                                 
        if frame % keyframe_interval == 0:
            surf_features = surf.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)[1]
            if surf_features is not None:
            	yield surf_features.reshape((-1,64))
#             yield img
    video_cap.release()

def get_for_one(line, surf):
    start = time.time()
    video_name = line.replace('\n', '')
    downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
    surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')

    if (os.path.isfile(surf_feat_video_filename)) or (not os.path.isfile(downsampled_video_filename)):
        print("skipping", video_name)
        return
    start = time.time()
    keyframe_iterator = get_keyframes(surf, downsampled_video_filename, keyframe_interval)
    try:
        surf_feats = np.vstack(np.array(list(keyframe_iterator)))
        np.savetxt(surf_feat_video_filename, surf_feats, delimiter=",")
        print("Finished", video_name, "in", time.time() - start, "seconds")
    except(ValueError):
        os.system("echo " + video_name + " >> emptyvideos")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))
    print("Using: ", all_video_names, config_file)
    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    hessian_threshold = my_params.get('hessian_threshold')
    surf_features_folderpath = my_params.get('surf_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO Create SURF object
    # surf = cv2.SURF(hessian_threshold,2,3,1)
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    surf.setExtended(False)

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    Parallel(n_jobs=3, prefer="threads")(delayed(get_for_one)(line, surf) for line in fread.readlines())
    # for line in fread.readlines():
    #     start = time.time()
    #     video_name = line.replace('\n', '')
    #     downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
    #     surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')

    #     if (os.path.isfile(surf_feat_video_filename)) or (not os.path.isfile(downsampled_video_filename)):
    #         print("skipping", video_name)
    #         continue

    #     # # Get SURF features for one video
    #     # # get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval)
    #     # keyframe_iterator = get_keyframes(surf, downsampled_video_filename, keyframe_interval)
    #     # try:
    #     #     surf_feats = np.vstack(np.array(list(keyframe_iterator)))
    #     #     np.savetxt(surf_feat_video_filename, surf_feats, delimiter=",")
    #     #     print("Finished", video_name, "in", time.time() - start, "seconds")
    #     # except(ValueError):
    #     #     os.system("echo " + video_name + " > emptyvideos")

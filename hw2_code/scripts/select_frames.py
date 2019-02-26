#!/bin/python
# Randomly select 

import numpy
import os
import sys
import yaml
import time

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} file_list select_ratio output_file".format(sys.argv[0]))
        print("file_list -- the list of video names")
        print("select_ratio -- the ratio of frames to be randomly selected from each audio file")
        print("output_file -- path to save the selected frames (feature vectors)")
        exit(1)

    file_list = sys.argv[1]; output_file = sys.argv[3]
    ratio = float(sys.argv[2])

    config_file = "../config.yaml"                                                                  
    my_params = yaml.load(open(config_file))                                                   
                                                                                               
    # Get parameters from config file                                                          
    surf_features_folderpath = my_params.get('surf_features')                                  

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows
    numpy.random.seed(18877)

    if not os.path.exists(surf_features_folderpath):                                           
        os.mkdir(surf_features_folderpath)                                                     
    
    fread = open(file_list, "r")
    fwrite = open(output_file,"w")
    
    for line in fread.readlines():
    
        start = time.time()
        video_name = line.replace('\n', '')
        surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')
    
        if (not os.path.isfile(surf_feat_video_filename)):
            print(("skipping", video_name))
            continue
        
        # Mind the delimiter -- Check
        array = numpy.genfromtxt(surf_feat_video_filename, delimiter=",")
        numpy.random.shuffle(array)
        select_size = int(array.shape[0] * ratio)
        feat_dim = array.shape[1]
    
        for n in range(select_size):
            line = str(array[n][0])
            for m in range(1, feat_dim):
                line += ',' + str(array[n][m])
            fwrite.write(line + '\n')
        print("Done", video_name, "in", time.time() - start, "seconds")
    fwrite.close()

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import copy
from PIL import Image


def get_keyframes(downsampled_video_filename, keyframe_interval): 
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
            yield img
    video_cap.release()

def get_for_one(line, resnet_model, layer, scaler, normalize, to_tensor):
    pass

def get_vector(image_name): #, resnet_model, scaler, normalize, to_tensor):
    # 1. Load the image with Pillow library
    img = Image.fromarray(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Defining a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    resnet_model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


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
    cnn_features_folderpath = my_params.get('cnn_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # define Model
    resnet_model = models.resnet18(pretrained=True)
    layer = resnet_model._modules.get('avgpool')
    resnet_model.eval()

    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # Check if folder for CNN features exists
    if not os.path.exists(cnn_features_folderpath):
        os.mkdir(cnn_features_folderpath)

    # Loop over all videos (training, val, testing)
    # get CNN features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    #Parallel(n_jobs=3, prefer="threads")(delayed(get_for_one)(line, resnet_model, layer, scaler, normalize, to_tensor) for line in fread.readlines())
    for line in fread.readlines():
        start = time.time()
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        cnn_feat_video_filename = os.path.join(cnn_features_folderpath, video_name + '.cnn')

        if (os.path.isfile(cnn_feat_video_filename)) or (not os.path.isfile(downsampled_video_filename)):
            print("skipping", video_name)
            continue
        start = time.time()
        keyframe_iterator = get_keyframes(downsampled_video_filename, keyframe_interval)
        try:
            cnn_feats = np.vstack([get_vector(img) for img in keyframe_iterator])
            np.savetxt(cnn_feat_video_filename, cnn_feats, delimiter=",")
            print("Finished", video_name, "in", time.time() - start, "seconds")
        except(ValueError):
            os.system("echo " + video_name + " >> cnnemptyvideos")

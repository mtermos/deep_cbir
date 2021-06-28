import sys
import os
import time
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
from torch import Tensor

from torchsummary import summary

import torchvision.models as models
from torchvision import transforms
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import *
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls

from typing import Type, Any, Callable, Union, List, Dict, Optional, cast
from collections import OrderedDict


from evaluate import compute_map_and_print

data_dir = os.getcwd() + "\\data"

def load_db_dict():
    infile = open(data_dir + "\\gnd_oxford5k.pkl",'rb')
    new_dict = pickle.load(infile)
    infile.close()
    global db_dict
    db_dict = new_dict
    return db_dict
    

def import_images():
    img_list = db_dict["imlist"]
    for i in range(len(img_list)):
        img_list[i] = data_dir + "\\jpg\\" + img_list[i] + ".jpg"
    global images_list
    images_list = img_list
    return images_list

class IntResNet(ResNet):
    def __init__(self,output_layer,*args):
        self.output_layer = output_layer
        super().__init__(*args)
        
        self._layers = []
        for l in list(self._modules.keys()):
            self._layers.append(l)
            if l == output_layer:
                break
        self.layers = OrderedDict(zip(self._layers,[getattr(self,l) for l in self._layers]))

    def _forward_impl(self, x):
        for l in self._layers:
            x = self.layers[l](x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
    
    
def new_resnet(
    arch: str,
    outlayer: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> IntResNet:

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }

    model = IntResNet(outlayer, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def initialize_model(model_name):
    if(model_name == "resnet50"):
        new_model = new_resnet(model_name,'avgpool',Bottleneck, [3,4,6,3],True,True)
    new_model = new_model.to('cuda:0')
    print(summary(new_model,input_size=(3, 224, 224)))
    global model
    model = new_model
    return new_model


def extract_dataset_images_features():
    image_features = []
    i = 0
    for image in images_list:
        input_image = Image.open(image)
        i += 1
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        out = model(input_batch)
        out = out.cpu().data.numpy()
        output = np.squeeze(out).ravel()
        image_features.append(output)

        sys.stdout.write("\r{0} done out of 5063".format(i))
        sys.stdout.flush()
    
    global db_features
    db_features = np.array(image_features)
    
    return db_features

def save_db_features():
    file = open("db_features.pkl", "wb")
    pickle.dump(db_features, file)
    file.close()
    
def load_db_features():
    global db_features
    db_features = pickle.load(open("db_features.pkl",'rb'))
    return db_features

def query_image_and_show(number_of_images, image_id, bbx = None):
    
    print("Query Image")
    imgplot = plt.imshow(mpimg.imread(images_list[image_id]))
    plt.show()
    print(images_list[image_id])
    print(image_id)
    print("*********************")
    print("searching...")
    ranked = query_image(image_id, bbx)
    
    closest = ranked[:number_of_images]

    for i in closest:
        imgplot = plt.imshow(mpimg.imread(images_list[i]))
        plt.show()
        print(images_list[i])
        print(i)
        print("*********************")
    
    
def query_image(image_id, bbx = None):
    image_path = images_list[image_id]
    
    input_image = Image.open(image_path)
    
    if bbx is not None:
        input_image = input_image.crop(bbx)
        
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    out = model(input_batch)
    out = out.cpu().data.numpy()
    query = np.squeeze(out).ravel()
    

    dist_vec = np.linalg.norm(db_features - query, axis=1)
    ranked = np.argsort(dist_vec)
    
    return ranked

def evaluate():
    ranks = np.empty(shape=[0, 5063])
    for qid in range(len(db_dict["qidx"])):

        ranked = np.array(query_image(db_dict["qidx"][qid], db_dict["gnd"][qid]["bbx"]))
        ranks = np.append(ranks, [ranked.tolist()], axis=0)
        
    compute_map_and_print("oxford5k", ranks.T, db_dict["gnd"])
        
    
    
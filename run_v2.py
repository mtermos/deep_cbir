import sys
import os
import time
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import timeit
import cv2    
from sklearn.cluster import MeanShift, estimate_bandwidth

from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from arch import GeM, L2Norm
from fastai.layers import Flatten
from evaluate import compute_map_and_print



from cirtorch.networks.imageretrievalnet import init_network
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.datasets.datahelpers import default_loader, imresize

data_dir = os.getcwd() + "\\data"

output_dim = 2048


# making the resolution adaptive?

# image_size = 256
image_size = 512
# image_size = 768
# image_size = 1024

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


class new_resnet50_class(nn.Module):
    def __init__(self, new_model):
        super().__init__()
        self.conv1 = new_model.conv1
        self.bn1 = new_model.bn1
        self.relu = new_model.relu
        self.maxpool = new_model.maxpool
        self.layer1 = new_model.layer1
        self.layer2 = new_model.layer2
        self.layer3 = new_model.layer3
        self.layer4 = new_model.layer4
        self.avgpool = new_model.avgpool
        # self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = self.maxpool(x)

        return x


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class new_resnet101_class(nn.Module):
    def __init__(self, new_model):
        super().__init__()
        self.conv1 = new_model.conv1
        self.bn1 = new_model.bn1
        self.relu = new_model.relu
        self.maxpool = new_model.maxpool
        self.layer1 = new_model.layer1
        self.layer2 = new_model.layer2
        self.layer3 = new_model.layer3
        self.layer4 = new_model.layer4
        self.avgpool = new_model.avgpool

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class new_inception_v3_class(nn.Module):
    def __init__(self, new_model):
        super().__init__()
        self.Conv2d_1a_3x3 = new_model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = new_model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = new_model.Conv2d_2b_3x3
        self.maxpool1 = new_model.maxpool1
        self.Conv2d_3b_1x1 = new_model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = new_model.Conv2d_4a_3x3
        self.maxpool2 = new_model.maxpool2
        self.Mixed_5b = new_model.Mixed_5b
        self.Mixed_5c = new_model.Mixed_5c
        self.Mixed_5d = new_model.Mixed_5d
        self.Mixed_6a = new_model.Mixed_6a
        self.Mixed_6b = new_model.Mixed_6b
        self.Mixed_6c = new_model.Mixed_6c
        self.Mixed_6d = new_model.Mixed_6d
        self.Mixed_6e = new_model.Mixed_6e
        self.AuxLogits = new_model.AuxLogits
        self.Mixed_7a = new_model.Mixed_7a
        self.Mixed_7b = new_model.Mixed_7b
        self.Mixed_7c = new_model.Mixed_7c
        self.avgpool = new_model.avgpool
        self.dropout = new_model.dropout
        
    def forward(self, x: Tensor) -> Tensor:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x
    

class new_densenet121_class(nn.Module):
    def __init__(self, new_model):
        super().__init__()
        self.cnn =  new_model.features
        self.head = nn.Sequential(nn.ReLU(),
                          GeM(3.0),
                          Flatten(),
                          L2Norm())
        
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.head(x)
       
        return x


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    
    
def my_model(model_name):
    if(model_name == "resnet50"):
        cnn_model = models.resnet50(pretrained=True)
        # new_model = nn.Sequential(*list(cnn_model.children())[:-1])
        new_model = new_resnet50_class(cnn_model)

    if(model_name == "resnet101"):
        cnn_model = models.resnet101(pretrained=True)
        new_model = new_resnet101_class(cnn_model)
    
    if(model_name == "inception"):
        cnn_model = models.inception_v3(pretrained=True)
        new_model = new_inception_v3_class(cnn_model)
        
    if(model_name == "densenet121"):
        cnn_model = models.densenet121(pretrained=True)
        new_model = new_densenet121_class(cnn_model)
        
    if(model_name == "citorch"):
        state = torch.load('C:\\Users\\Mortada\\Python\\Image_retrieval\\cnnimageretrieval-pytorch\\data\\networks\\retrievalSfM120k-resnet101-gem-b80fb85.pth')

        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False

        new_model = init_network(net_params)
        

    if(model_name == "trained"):
        state = torch.load("C:\\Users\\Mortada\\Python\\Image_retrieval\\deep_cbir\\export\\retrieval-SfM-120k_resnet50_imsize512\\model_best.pth.tar")

        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False

        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])
        new_model = net

    global model
    model = new_model
    return new_model


    
def extract_dataset_images_features():
    image_features = torch.zeros(len(images_list), output_dim)
    i = 0
    for image in images_list:
        output = extract_vectors(image)

        image_features[i, :] = output.flatten()
        i += 1
        sys.stdout.write("\r{0} done out of 5063".format(i))
        sys.stdout.flush()

    # for i in range(len(images_list)):
    #     image_features.append(test_query_image(i))
    
    global db_features
    db_features = image_features
    
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
    
    start = timeit.default_timer()
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
        
        
    stop = timeit.default_timer()

    print('Time: ', stop - start) 


def extract_vectors(image_path, bbxs=None):
    model.cuda()
    model.eval()

    with torch.no_grad():
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        img = default_loader(image_path)
        imfullsize = max(img.size)

        if bbxs is not None:
            img = img.crop(bbxs)
            img = imresize(img, image_size * max(img.size) / imfullsize)
        else:
            img = imresize(img, image_size)



        #testing meanshift


        # cv_img = segmented_image(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        # plt.imshow(img)
        # plt.show()

        # img = Image.fromarray(cv_img)

        # plt.imshow(img)
        # plt.show()

        #end testing meanshift



        img = transform(img)
        input = img.cuda()

        vecs = model(input.unsqueeze(0)).cpu().data.squeeze()
        
    return vecs

def query_image(image_id, bbx = None):
    image_path = images_list[image_id]
    
    query = extract_vectors(image_path,bbx)

    dist_vec = np.linalg.norm(db_features - query, axis=1)
    ranked = np.argsort(dist_vec)
    
    return ranked

def evaluate():
    start = timeit.default_timer()
    
    ranks = np.empty(shape=[0, 5063])
    for qid in range(len(db_dict["qidx"])):

        ranked = np.array(query_image(db_dict["qidx"][qid], db_dict["gnd"][qid]["bbx"]))
        ranks = np.append(ranks, [ranked.tolist()], axis=0)
        
    compute_map_and_print("oxford5k", ranks.T, db_dict["gnd"])
        
    stop = timeit.default_timer()

    print('Time: ', stop - start)



def test_model(test_model, dataset):
    if test_model is not None:
        global model
        model = test_model

    if dataset != "oxford5k":
        return 0

    load_db_dict()
    import_images()
    extract_dataset_images_features()
    evaluate()


def segmented_image(originImg):
    #Loading original image
    # originImg = cv2.imread(image_path)

    # Shape of original image    
    originShape = originImg.shape


    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities    
    flatImg=np.reshape(originImg, [-1, 3])


    # Estimate bandwidth for meanshift algorithm    
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)    
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)

    # Performing meanshift on flatImg    
    ms.fit(flatImg)

    # (r,g,b) vectors corresponding to the different clusters after meanshift    
    labels=ms.labels_

    # Remaining colors after meanshift    
    cluster_centers = ms.cluster_centers_    

    # Finding and diplaying the number of clusters    
    labels_unique = np.unique(labels)    
    n_clusters_ = len(labels_unique)

    # Displaying segmented image    
    segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]

    return segmentedImg.astype(np.uint8)


# db_dict = load_db_dict()
# model = my_model("resnet50")
# extract_vectors('all_souls_000209.jpg')
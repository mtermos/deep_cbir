from hashlib import new
import sys
import os
import time
import math
from fastai.torch_core import tensor
import numpy as np
# from PIL import Image
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
# image_size = 512
# image_size = 768
image_size = 1024

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

def create_yolo_model(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """Creates a specified YOLOv5 model

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv5 pytorch model
    """
    from pathlib import Path

    from models.yolo import Model, attempt_load
    from utils.general import check_requirements, set_logging
    from utils.google_utils import attempt_download
    from utils.torch_utils import select_device

    file = Path(__file__).absolute()
    check_requirements(requirements=file.parent / 'requirements.txt', exclude=('tensorboard', 'thop', 'opencv-python'))
    set_logging(verbose=verbose)

    save_dir = Path('') if str(name).endswith('.pt') else file.parent
    path = (save_dir / name).with_suffix('.pt')  # checkpoint path
    try:
        device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)

        if pretrained and channels == 3 and classes == 80:
            model = attempt_load(path, map_location=device)  # download/load FP32 model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{name}.yaml'))[0]  # model.yaml path
            model = Model(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                msd = model.state_dict()  # model state_dict
                csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
        if autoshape:
            model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache may be out of date, try `force_reload=True`. See %s for help.' % help_url
        raise Exception(s) from e






model = create_yolo_model(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)  # pretrained
# print(model)

db_dict = load_db_dict()
images_list = import_images()


imgs = images_list[db_dict["qidx"][5]]
image = cv2.imread(imgs)
# results = model(img1)




from torch.cuda import amp
from pathlib import Path, PosixPath
from PIL import Image
import requests
from utils.datasets import exif_transpose, letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from models.common import Detections
from utils.torch_utils import time_sync
from utils.general import xywh2xyxy, box_iou
from utils.plots import plot_one_box

size=640
augment=False
profile=False
conf = 0.25  # NMS confidence threshold
iou = 0.45  # NMS IoU threshold
classes = None  # (optional list) filter by class
max_det = 1000  # maximum number of detections per image


t = [time_sync()]
p = next(model.parameters())  # for device and type
if isinstance(imgs, torch.Tensor):  # torch
    with amp.autocast(enabled=p.device.type != 'cpu'):
        result = model(imgs.to(p.device).type_as(p), augment, profile)  # inference

# Pre-process
n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
shape0, shape1, files = [], [], []  # image and inference shapes, filenames
for i, im in enumerate(imgs):
    f = f'image{i}'  # filename
    if isinstance(im, (str, PosixPath)):  # filename or uri
        im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
        im = np.asarray(exif_transpose(im))
    elif isinstance(im, Image.Image):  # PIL Image
        im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
    files.append(Path(f).with_suffix('.jpg').name)
    if im.shape[0] < 5:  # image in CHW
        im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
    im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
    s = im.shape[:2]  # HWC
    shape0.append(s)  # image shape
    g = (size / max(s))  # gain
    shape1.append([y * g for y in s])
    imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
shape1 = [make_divisible(x, int(1)) for x in np.stack(shape1, 0).max(0)]  # inference shape
x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
t.append(time_sync())

with amp.autocast(enabled=p.device.type != 'cpu'):
    # Inference
    y = model(x, augment, profile)[0]  # forward
    raw_results = y
    t.append(time_sync())
    
    # Post-process
    y = non_max_suppression(y, conf, iou_thres=iou, classes=classes, max_det=max_det)  # NMS
    for i in range(n):
        scale_coords(shape1, y[i][:, :4], shape0[i])

    t.append(time_sync())
    result = Detections(imgs, y, files, t, names = model.names, shape=x.shape)

# result.print()
# result.show()
# print(raw_results[0][0][:4])

candidates = (raw_results[..., 4] > 0.0001).nonzero()

# image = cv2.imread(x[0])

#resizing image
# shape0, shape1, files = [], [], []  # image and inference shapes, filenames
# s = image.shape[:2]  # HWC
# shape0.append(s)  # image shape
# g = (size / max(s))  # gain
# shape1.append([y * g for y in s])
# shape1 = [make_divisible(x, int(1)) for x in np.stack(shape1, 0).max(0)]  # inference shape

# image = letterbox(image, new_shape=shape1, auto=False)[0]

scale_coords(shape1, candidates[:, :4], shape0[i])






# printing
result_test = image

i=0
for cand in candidates:
    box = xywh2xyxy(raw_results[0][cand[1]][:4].unsqueeze(0)).squeeze(0)
    color = colors(i)
    result_test = plot_one_box(box, result_test, color = color)
    i+=1

cv2.imshow('Image', result_test) 
cv2.waitKey()


from utils.plots import colors

boxes = torch.empty((candidates.shape[0],4))
scores = torch.empty(candidates.shape[0])
i=0
for cand in candidates:
    boxes[i] = xywh2xyxy(raw_results[0][cand[1]][:4].unsqueeze(0)).squeeze(0)
    scores[i] = raw_results[0][cand[1]][4]
    i+=1


# merging boxes
iou_thres = 0.45
# update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)

i = 0
final_boxes = []
for i in range(len(boxes)):
    iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
    weights = iou * scores[None]  # box weights
    final_boxes.append(torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True))  # merged boxes


i=0
for box in final_boxes:
    color = colors(i)
    result_test = plot_one_box(box, result_test, color = color)
    i+=1




cv2.imshow('Image', result_test) 
cv2.waitKey()

# result_test.print()
# result_test.show()


# results.print()
# results.show()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
# results.pandas().xyxy[0]
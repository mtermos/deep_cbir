

import torch.nn as nn
import torchvision.models as models

from cirtorch.layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC, Rpool
from cirtorch.layers.normalization import L2N, PowerLaw

class Cbir_network(nn.Module):
    
    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(Cbir_network, self).__init__()
        self.features = nn.Sequential(*features)
        self.pool = pool
        self.norm = L2N()
        self.fc = nn.Linear(2048,2048)
        self.meta = meta
    
    def forward(self, x):
        # x -> features
        o = self.features(x)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        o = self.fc(o)
        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1,0)

    def __repr__(self):
        tmpstr = super(Cbir_network, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(model_name):

    pooling = 'gem'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    dim = 2048

    if model_name == "resnet50":
        cnn_model = models.resnet50(pretrained=True)
        features = list(cnn_model.children())[:-2]
    if model_name == "resnet101":
        cnn_model = models.resnet101(pretrained=True)
        features = list(cnn_model.children())[:-2]
    
    pool = GeM()
    # pool = Rpool(pool)

    whiten = None
    lwhiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture' : model_name, 
        'pooling' : pooling, 
        'mean' : mean, 
        'std' : std,
        'outputdim' : dim,
    }

    # create a generic image retrieval network
    net = Cbir_network(features, lwhiten, pool, whiten, meta)

    return net

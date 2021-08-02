
import math
import os
import time
import shutil
import numpy as np

import torch
import torch.nn as nn

import torchvision.transforms as transforms


from cbir_network import init_network
from run import test_model

from cirtorch.layers.loss import ContrastiveLoss, TripletLoss
from cirtorch.datasets.traindataset import TuplesDataset
from cirtorch.datasets.datahelpers import collate_tuples, cid2filename
from model import SOLAR_LOCAL

export_directory = os.path.join(os.getcwd(), 'export')
# resume = "model_epoch2.pth.tar"
resume = None
image_size = 512
model_name = "resnet50"
training_dataset = "retrieval-SfM-120k"
neg_num = 5
query_size = 2000
pool_size = 20000
batch_size = 5
num_workers = 8
val = 1
test_datasets ='oxford5k'
epochs = 100
test_freq = 1
update_every = 1
print_freq = 10
min_loss = float('inf')

def main():
    global min_loss

    lr = 1e-6
    optimizer_name = "adam"
    checkpoint_directory = "retrieval-SfM-120k_resnet50_imsize512"
    checkpoint_directory = os.path.join(export_directory, checkpoint_directory)

    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # model = init_network(model_name)
    model = SOLAR_LOCAL()

    # move network to gpu
    model.cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = TuplesDataset(
        name= training_dataset,
        mode='train',
        imsize= image_size,
        nnum=neg_num,
        qsize=query_size,
        poolsize=pool_size,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers= num_workers, pin_memory=True, sampler=None,
        drop_last=True, collate_fn=collate_tuples
    )
    if val:
        val_dataset = TuplesDataset(
            name=training_dataset,
            mode='val',
            imsize=image_size,
            nnum=neg_num,
            qsize=float('Inf'),
            poolsize=float('Inf'),
            transform=transform
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            drop_last=True, collate_fn=collate_tuples
        )

    # test(test_datasets, model)

    for epoch in range(start_epoch, epochs):

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        scheduler.step()
        # # debug printing to check if everything ok
        # lr_feat = optimizer.param_groups[0]['lr']
        # lr_pool = optimizer.param_groups[1]['lr']
        # print('>> Features lr: {:.2e}; Pooling lr: {:.2e}'.format(lr_feat, lr_pool))

        # train for one epoch on train set
        loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if val:
            with torch.no_grad():
                loss = validate(val_loader, model, criterion, epoch)

        # evaluate on test datasets every test_freq epochs
        if (epoch + 1) % test_freq == 0:
            with torch.no_grad():
                test_model(model, test_datasets)

        # remember best loss and save checkpoint
        is_best = loss < min_loss
        min_loss = min(loss, min_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'meta': model.meta,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint_directory)




def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for training
    avg_neg_distance = train_loader.dataset.create_epoch_tuples(model)

    # switch to train mode
    model.cuda()
    model.train()
    model.apply(set_batchnorm_eval)

    # zero out gradients
    optimizer.zero_grad()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)


        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple

        for q in range(nq):
            output = torch.zeros(model.meta['outputdim'], ni).cuda()
            for imi in range(ni):

                # compute output vector for image imi
                output[:, imi] = model(input[q][imi].cuda()).squeeze()

            # reducing memory consumption:
            # compute loss for this query tuple only
            # then, do backward pass for one tuple only
            # each backward pass gradients will be accumulated
            # the optimization step is performed for the full batch later
            loss = criterion(output, target[q].cuda())
            losses.update(loss.item())
            loss.backward()

        if (i + 1) % update_every == 0:
            # do one step for multiple batches
            # accumulated gradients are used
            optimizer.step()
            # zero out gradients so we can 
            # accumulate new ones over batches
            optimizer.zero_grad()
            # print('>> Train: [{0}][{1}/{2}]\t'
            #       'Weight update performed'.format(
            #        epoch+1, i+1, len(train_loader)))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % print_freq == 0 or i == 0 or (i+1) == len(train_loader):
            print('>> Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

        
        torch.cuda.empty_cache()    

    return losses.avg


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for validation
    avg_neg_distance = val_loader.dataset.create_epoch_tuples(model)

    # switch to evaluate mode
    model.cuda()
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        output = torch.zeros(model.meta['outputdim'], nq*ni).cuda()

        for q in range(nq):
            for imi in range(ni):

                # compute output vector for image imi of query q
                output[:, q*ni + imi] = model(input[q][imi].cuda()).squeeze()

        # no need to reduce memory consumption (no backward pass):
        # compute loss for the full batch
        loss = criterion(output, torch.cat(target).cuda())

        # record loss
        losses.update(loss.item()/nq, nq)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % print_freq == 0 or i == 0 or (i+1) == len(val_loader):
            print('>> Val: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg
    

def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std:
        # we do training one image at a time
        # so the statistics would not be per batch
        # hence we choose freezing (ie using imagenet statistics)
        m.eval()
        # # freeze parameters:
        # # in fact no need to freeze scale and bias
        # # they can be learned
        # # that is why next two lines are commented
        # for p in m.parameters():
            # p.requires_grad = False
if __name__ == '__main__':
    main()
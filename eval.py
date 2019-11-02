import torch
import torch.nn as nn
import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable
from torch.utils import data
import torchvision
import numpy as np
from dataset import iSLR_Dataset
from model import iSLR_Model
from transforms import *

import argparse
import os
import os.path as osp
import time
from tensorboardX import SummaryWriter

from opts import parser
from viz_utils import attentionmap_visualize

def create_path(path):
    if not osp.exists(path):
        os.makedirs(path)

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

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    
    create_path(args.root_model)

    args.store_name = '_'.join(['eval','iSLR',args.modality,args.arch,\
                                'class'+str(args.num_class)])
    
    # get model 
    model = iSLR_Model(args.num_class,
                        hidden_unit=args.hidden_unit,base_model=args.arch)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model).cuda()
    model_dict = model.state_dict()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # restore model
    if args.val_resume:
        if osp.isfile(args.val_resume):
            checkpoint = torch.load(args.val_resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # restore_param = {k:v for k,v in model_dict.items()}
            # model_dict.update(restore_param)
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {}) (best prec {})"
                  .format(args.evaluate, checkpoint['epoch'], best_prec1)))

        else:
            print(("=> no checkpoint found at '{}'".format(args.val_resume)))
    
    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean,input_std)

    train_loader = torch.utils.data.DataLoader(
        iSLR_Dataset(args.video_root,args.train_file,
            transform=torchvision.transforms.Compose([
                # train_augmentation,
                GroupScale(int(scale_size)),
                GroupCenterCrop(crop_size),
                Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                normalize,
            ])
        ),
        batch_size=args.batch_size,shuffle=True,
        num_workers=args.workers,pin_memory=True,
        # collate_fn=collate
    )

    val_loader = torch.utils.data.DataLoader(
        iSLR_Dataset(args.video_root,args.val_file,
            transform=torchvision.transforms.Compose([
                GroupScale(int(scale_size)),
                GroupCenterCrop(crop_size),
                Stack(roll=(args.arch in ['BNInception','InceptonV3'])),
                ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                normalize,
            ])
        ),
        batch_size=args.batch_size,shuffle=False,
        num_workers=args.workers,pin_memory=True,
        # collate_fn=collate
    )

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # get writer
    # global writer
    # writer = SummaryWriter(logdir='runs/'+args.store_name)

    prec1 = validate(val_loader, model, criterion, 0)

    # for epoch in range(args.start_epoch, args.epochs):
    #     adjust_learning_rate(optimizer, epoch , args.lr_steps)


    #     # train for one epoch
    #     train(train_loader, model, criterion, optimizer, epoch)

    #     # evaluate on validation set
    #     if (epoch) % args.eval_freq == 0 or epoch == args.epochs-1:
    #         prec1 = validate(val_loader, model, criterion, epoch // args.eval_freq)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)
        input.require_grad = True
        input_var = input
        target.require_grad = True
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()
        # print(list(model.module.base_model.conv1_7x7_s2.parameters())[0][0,:,:,:])
        # print(list(model.module.lstm.parameters())[0].mean())
        # print(list(model.module.lstm.parameters())[0].std())

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
                # print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                    'Prec@5 {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5, 
                        lr=optimizer.param_groups[-1]['lr']))
            print(output)

        # writer.add_scalar('train/loss', losses.avg, epoch*len(train_loader)+i)
        # writer.add_scalar('train/acc', top1.avg, epoch*len(train_loader)+i)



def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)
        input.require_grad = True
        input_var = input
        target.require_grad = True
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # plot attention map on input image
        attention_map = model.module.attention_map
        attentionmap_visualize(input, attention_map)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f}% ({top1.avg:.3f}%)\t'
                  'Prec@5 {top5.val:.3f}% ({top5.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print(output)

        # writer.add_scalar('val/acc', top1.avg, epoch*len(val_loader)+i)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (args.root_model, args.store_name),
            '%s/%s_best.pth.tar' % (args.root_model, args.store_name))

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # return the k largest elements of the given input Tensor
    # along the given dimension. dim = 1
    # pred is the indices
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def collate(batch):
    images_tensor = []
    targets = []
    len_list = []
    max_length = 0
    for sample in batch:
        images = sample[0]
        target = torch.LongTensor([sample[1]])
        length = images.size(0)
        if length > max_length:
            max_length = length
        images_tensor.append(images)
        targets.append(target)
        len_list.append(length)
    # fill zeros
    filled_images_tensor = []
    for images, length in zip(images_tensor, len_list):
        pad = torch.zeros(max_length-length, images.size(1), images.size(2))
        filled_images = torch.cat([images, pad],0)
        filled_images_tensor.append(filled_images)
    images_tensor = filled_images_tensor
    return torch.stack(images_tensor, 0), torch.cat(targets, 0)

if __name__=="__main__":
    main()

        
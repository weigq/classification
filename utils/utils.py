#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import os
import errno

import torch

__all__ = ['save_checkpoint', 'update_lr', 'AverageMeter']


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def update_lr(optimizer, epoch, lr, gamma, lr_step):
    if epoch % lr_step:
        lr = lr * gamma ** (epoch / lr_step)
        print(lr)
        return lr
    else:
        lr = lr * gamma ** (epoch / lr_step)
        for param in optimizer.param_groups:
            param['lr'] = lr
        print lr
        return lr


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class AverageMeter(object):
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.reset()

    def reset(self):
        pass

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

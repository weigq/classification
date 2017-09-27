#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import os

import torch

__all__ = ['save_checkpoint', 'adjust_learning_rate']


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, gamma, schedule):
    global state
    if epoch in schedule:
        state['lr'] *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

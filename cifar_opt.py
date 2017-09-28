#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

__all__ = ['TrainOpt']


class TrainOpt:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument('--dataset',        type=str, default='cifar10')
        self.parser.add_argument('-j', '--workers',  type=int, default=4)
        self.parser.add_argument('--data_path',      type=str, default='./data')

        self.parser.add_argument('--epochs',         type=int, default=300)
        self.parser.add_argument('--start_epoch',    type=int, default=0)
        self.parser.add_argument('--train_batch',    type=int, default=128)
        self.parser.add_argument('--test_batch',     type=int, default=100)
        self.parser.add_argument('--lr',             type=float, default=0.1)
        self.parser.add_argument('--drop',           type=float, default=0)
        self.parser.add_argument('--lr_step',        type=int, default=60)
        self.parser.add_argument('--gamma',          type=float, default=0.1, help='rate of lr decay')
        self.parser.add_argument('--lr_decay',       type=float, default=0.1, help='rate of lr decay')

        self.parser.add_argument('--momentum',       type=float, default=0.9)
        self.parser.add_argument('--weight_decay',   type=float, default=5e-4)
        self.parser.add_argument('--checkpoint',     type=str, default='checkpoint')
        self.parser.add_argument('--resume',         type=str, default='')

        self.parser.add_argument('--arch',           type=str, default='resnet20')
        self.parser.add_argument('--depth',          type=int, default=29)
        self.parser.add_argument('--cardinality',    type=int, default=8, help='Model cardinality (group).')
        self.parser.add_argument('--widen_factor',   type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
        self.parser.add_argument('--growthRate',     type=int, default=12, help='Growth rate for DenseNet.')
        self.parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')

        self.parser.add_argument('--manualSeed',     type=int, help='manual seed')
        self.parser.add_argument('--evaluate',       dest='evaluate', action='store_true')

        self.parser.add_argument('--gpu_id',         type=str, default='0')

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        return self.opt


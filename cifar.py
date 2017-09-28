
from __future__ import print_function

import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, accuracy, savefig

from utils.utils import update_lr, save_checkpoint, mkdir, AverageMeter

from cifar_opt import TrainOpt

#
# state = {k: v for k, v in opt._get_kwargs()}
#
# # Validate dataset
# assert opt.dataset == 'cifar10' or opt.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
#
# # Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
# use_gpu = torch.cuda.is_available()
#
# # Random seed
# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)
# if use_gpu:
#     torch.cuda.manual_seed_all(opt.manualSeed)


def main(opt):
    best_acc = 0

    # GPU/CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    use_gpu = torch.cuda.is_available()
    # Random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if use_gpu:
        torch.cuda.manual_seed_all(opt.manualSeed)

    if not opt.evaluate:
        is_train = True
        print("=========================================")
        print("           training procedure            ")
        print("=========================================")
    else:
        is_train = False
        print("=========================================")
        print("           testing procedure             ")
        print("=========================================")

    if opt.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    # create checkpoint
    if not os.path.isdir(opt.checkpoint):
        mkdir(opt.checkpoint)

    # load dataset
    print('>>>>> load dataset {}'.format(opt.dataset))
    if is_train:
        transform_train = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                             )
        train_data = dataloader(root=opt.data_path, train=True, download=True, transform=transform_train)
        train_loader = data.DataLoader(train_data, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers)

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                        )
    test_data = dataloader(root=opt.data_path, train=False, download=False, transform=transform_test)
    test_loader = data.DataLoader(test_data, batch_size=opt.test_batch, shuffle=False, num_workers=opt.workers)

    # load model
    print(">>>>> creating model '{}'".format(opt.arch))
    if opt.arch.startswith('resnext'):
        model = models.__dict__[opt.arch](
                    cardinality=opt.cardinality,
                    num_classes=num_classes,
                    depth=opt.depth,
                    widen_factor=opt.widen_factor,
                    dropRate=opt.drop,
                )
    elif opt.arch.startswith('densenet'):
        model = models.__dict__[opt.arch](
                    num_classes=num_classes,
                    depth=opt.depth,
                    growthRate=opt.growthRate,
                    compressionRate=opt.compressionRate,
                    dropRate=opt.drop,
                )        
    elif opt.arch.startswith('wrn'):
        model = models.__dict__[opt.arch](
                    num_classes=num_classes,
                    depth=opt.depth,
                    widen_factor=opt.widen_factor,
                    dropRate=opt.drop,
                )
    elif opt.arch.endswith('resnet'):
        model = models.__dict__[opt.arch](
                    num_classes=num_classes,
                    depth=opt.depth,
                )
    else:
        model = models.__dict__[opt.arch](num_classes=num_classes)

    # multi-GPU
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: {0:.2f}M'.format(sum(p.numel() for p in model.parameters())/1000000.0))

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    start_epoch = opt.start_epoch

    # Resume
    title = 'cifar-10-' + opt.arch
    if opt.resume:
        print('>>>>> Resuming from checkpoint..')
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        opt.checkpoint = os.path.dirname(opt.resume)
        checkpoint = torch.load(opt.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if not is_train:
        print('>>>>> Evaluation only')
        test_loss, test_acc = test(test_loader, model, criterion, use_gpu)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    for epoch in range(start_epoch, opt.epochs):
        lr = update_lr(optimizer, epoch, opt.lr, opt.lr_decay, opt.lr_step)

        print('\nEpoch: {}/{} | {:.6}'.format(epoch + 1, opt.epochs, lr))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, use_gpu)
        test_loss, test_acc = test(test_loader, model, criterion, use_gpu)

        logger.append([lr, train_loss, test_loss, train_acc, test_acc],
                      ['int', 'float', 'float', 'float', 'float'])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=opt.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(opt.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(train_loader, model, criterion, optimizer, use_gpu):
    batch_time = AverageMeter()  # every batch time
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    model.train()
    for idx, (inputs, targets) in enumerate(train_loader):
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(batch=idx + 1,
                                                                                        size=len(train_loader),
                                                                                        bt=batch_time.avg,
                                                                                        total=bar.elapsed_td,
                                                                                        eta=bar.eta_td,
                                                                                        loss=losses.avg,
                                                                                        top1=top1.avg,
                                                                                        top5=top5.avg
                                                                                        )
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


def test(test_loader, model, criterion, use_gpu):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(test_loader))
    for idx, (inputs, targets) in enumerate(test_loader):
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(batch=idx + 1,
                                                                                        size=len(test_loader),
                                                                                        bt=batch_time.avg,
                                                                                        total=bar.elapsed_td,
                                                                                        eta=bar.eta_td,
                                                                                        loss=losses.avg,
                                                                                        top1=top1.avg,
                                                                                        top5=top5.avg
                                                                                        )
        bar.next()
    bar.finish()
    return losses.avg, top1.avg


if __name__ == '__main__':
    options = TrainOpt().parse()
    main(options)

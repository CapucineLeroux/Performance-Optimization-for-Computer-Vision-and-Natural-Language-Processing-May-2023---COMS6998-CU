'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.profiler import profile, record_function, ProfilerActivity

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse

from models import *
from utils import progress_bar
from collections import OrderedDict

import time


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--num_workers', default=2, type=int, help='number of dataloader workers')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG11')

# Uncomment to test cpu train/test times
# device = 'cpu'

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_vgg11.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, save=True, cpu=False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if not cpu:
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    if save:
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_vg11.pth')
            best_acc = acc


def print_model_size(path_name):
    print(f'Model size: {os.path.getsize(path_name) / 1e6} MB')


def trace_handler(p):
    output = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    print(output)


def profile_training():
    print('Profiler for training')
    with profile(
        schedule=torch.profiler.schedule(
            skip_first=10,
            wait=5,
            warmup=3,
            active=3,
            repeat=2
        ),
        on_trace_ready=trace_handler,
        with_stack=True
    ) as prof:
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            prof.step()


def profile_inference(cuda=True):
    print('Profiler for inference')
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            net(torch.randn(2,3,32,32))

    print(prof.key_averages().table(sort_by="cuda_time_total" if cuda else "cpu_time_total", row_limit=10))


def profile_memory():
    print('Profiler for memory consumption')
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        net(torch.randn(2,3,32,32))

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


def time_measurement():
    """
    C2: Time measurement of code
    """
    epoch_training_time = list()
    # run 5 epochs

    total_train_start = time.perf_counter()  # total training time C2.3
    for epoch in range(start_epoch, start_epoch + 5):

        epoch_train_start = time.perf_counter()  # training time per epoch, C2.2
        train(epoch)
        epoch_train_end = time.perf_counter()
        epoch_training_time.append(epoch_train_end - epoch_train_start)

        test(epoch)
        scheduler.step()

    total_train_end = time.perf_counter()


    print(f'Training times for each epoch (5): {epoch_training_time}')
    print(f'Total training time: {total_train_end - total_train_start}')
    print(f'Number of data loader workers: {args.num_workers}')


# quantize vgg11 model to show memory optimization
def quantize_model():
    global net
    # load pretrained model
    checkpoint = torch.load('checkpoint/ckpt_vg11.pth')
    net.load_state_dict(checkpoint['net'])

    print('Evaluating trained model (not quantized)')
    test('test', save=False)

    print_model_size('checkpoint/ckpt_vg11.pth')

    # quantize model must do on CPU since quantization does not support cuda
    state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        new_k = k[7:]
        state_dict[new_k] = v
    
    net_quantized = VGG('VGG11', q=True)
    net_quantized.to('cpu')
    net_quantized.load_state_dict(state_dict)

    net_quantized.qconfig = torch.quantization.default_qconfig
    net_quantized = torch.quantization.prepare(net_quantized)

    net = net_quantized
    print('Preparing model for quantization')
    test('test', save=False, cpu=True)

    net = torch.quantization.convert(net)
    torch.save(net.state_dict(), './checkpoint/vg11_quant.pth')

    print('Evaluating 8-bit quantized model')
    test('test', save=False, cpu=True)

    print_model_size('checkpoint/vg11_quant.pth')


# script model for faster cpu inference
def torchscript_model_CPU():
    global net
    # load quantized model
    net = VGG('VGG11', q=True)
    net.qconfig = torch.quantization.default_qconfig
    net = torch.quantization.prepare(net)
    net = torch.quantization.convert(net)
    net.load_state_dict(torch.load('checkpoint/vg11_quant.pth'))
    net.to('cpu')

    # get CPU model inference time 
    print('Running model inference for non-traced model (not calibrated)...')
    start = time.perf_counter()
    test('test', save=False, cpu=True)
    print(f'Quantized model test batch inference time (CPU): {time.perf_counter() - start} sec\n')

    # torchscript model
    print('Converting model to torchscript...')
    net = torch.jit.trace(net, torch.randn(2,3,32,32))

    # get new inference time
    print('Running model inference for traced model...')
    start = time.perf_counter()
    test('test', save=False, cpu=True)
    print(f'Torchscripted quantized model test batch inference time (CPU): {time.perf_counter() - start} sec\n')

    torch.jit.save(net, 'checkpoint/vg11_traced_cpu.pth')


def torchscript_model_GPU():
    global net
    net = VGG('VGG11')
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

        checkpoint = torch.load('checkpoint/ckpt_vg11.pth')
        net.load_state_dict(checkpoint['net'])
        
        profile_inference()

        start = time.perf_counter()
        test('test', save=False)
        print(f'Parallel model test batch inference time (GPU): {time.perf_counter() - start} sec\n')

        print('Converting model to torchscript...')
        net = torch.jit.trace(net, torch.randn(2,3,32,32))

        start = time.perf_counter()
        test('test', save=False)
        print(f'Traced parallel model test batch inference time (GPU): {time.perf_counter() - start} sec\n')

        # cannot export due to torch.nn.parallel scatter function, if export is needed must remove and do manually
        # torch.jit.save(net, 'checkpoint/vg11_traced_gpu.pth')
    
    else:
        print('Cannot run torchscript_model_GPU() function because cuda is not available')


if __name__ == '__main__':
    # profile training with different number of workers to see data loader times, best is 4
    # can compare to num_workers=1 to see difference
    profile_training()
    time_measurement()

    profile_memory()
    quantize_model()
    profile_memory()

    profile_inference(cuda=False)
    torchscript_model_CPU()
    profile_inference(cuda=False)
    
    torchscript_model_GPU()
    profile_inference()
    



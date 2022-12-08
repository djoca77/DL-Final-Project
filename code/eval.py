from __future__ import print_function


import sys
import os
import argparse

#import utils #additional preprocessing code to look into
import timeit

from preprocess import get_data_CIFAR



import tensorflow as tf

#Possible arguments
parser = argparse.ArgumentParser(description='Following arguments are used for the script')
parser.add_argument('--shared_rank', default=16, type=int, help='Number of shared base)')
parser.add_argument('--unique_rank', default=1, type=int, help='Number of unique base')
parser.add_argument('--batch_size', default=256, type=int, help='Batch_size')
parser.add_argument('--visible_device', default="0", help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default=None, help='Path of a pretrained model file')
parser.add_argument('--dataset_path', default="./data/", help='A path to dataset directory')
parser.add_argument('--model', default="ResNet56_DoubleShared", help='ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202, ResNet56_DoubleShared, ResNet32_DoubleShared, ResNet56_SingleShared, ResNet32_SingleShared, ResNet56_SharedOnly, ResNet32_SharedOnly, ResNet56_NonShared, ResNet32_NonShared')
args = parser.parse_args()

import model
dic_model = {
    'ResNet56_NonShared':model.ResNet56_NonShared}
    
if args.model not in dic_model:
    print("The model is currently not supported")
    sys.exit()

#testloader = utils.get_testdata('CIFAR10',args.dataset_path,batch_size=args.batch_size,download=True)
testloader = get_data_CIFAR('test')
#args.visible_device sets which cuda devices to be used
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=args.visible_device
device='cuda'

if 'DoubleShared' in args.model or 'SingleShared' in args.model:
    net = dic_model[args.model](args.shared_rank, args.unique_rank)
elif 'SharedOnly' in args.model:
    net = dic_model[args.model](args.shared_rank)
elif 'NonShared' in args.model:
    net = dic_model[args.model](args.unique_rank)
else:
    net = dic_model[args.model]()
    
net = net.to(device)

#Eval for models
def evaluation():
    net.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with tf.stop_gradient: #torch.no_grad()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            label_e = targets.view(targets.size(0), -1).expand_as(pred)
            correct = pred.eq(label_e).float()

            correct_top5 += correct[:, :5].sum()
            correct_top1 += correct[:, :1].sum()
            
            total += targets.size(0)
            
    # Save checkpoint.
    acc_top1 = 100.*correct_top1/total
    acc_top5 = 100.*correct_top5/total

    print("Eval_Acc_top1 = %.3f" % acc_top1)
    print("Eval_Acc_top5 = %.3f" % acc_top5)
        
if args.pretrained != None:
    checkpoint = tf.keras.models.load_model(args.pretrained)
    net.load_state_dict(checkpoint['net_state_dict'])
    
evaluation()
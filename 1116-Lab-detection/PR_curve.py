from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='subt', choices=['VOC', 'COCO', 'wamv','subt'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=subt_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default="weights/subt.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def eval():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'wamv':

        cfg = wamv
        dataset = wamvDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'subt':

        cfg = subt
        dataset = subtDetection(root=args.dataset_root,image_sets=[('test')],
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))


    ssd_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()



    net.eval()
    # loss counters

    testset = subtDetection(subt_ROOT, [('train')], None, subtAnnotationTransform())
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    imgs_list = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    result_list = [0,0,0,0] # TP TN FP FN
    all_data = 0
    conf_thres = 0
    iou_thres = 0.5

    print("\nPerforming object detection:")
    scale = torch.Tensor((640,480)).repeat(2)
    for batch_i in range(len(testset.ids)):
        image = testset.pull_image(batch_i)
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))

        if torch.cuda.is_available():
            xx = xx.cuda()
        with torch.no_grad():
            detections = net(xx).data

        # Save image and detections
        # imgs_list.extend(imgs)
        # img_detections.extend(detections)
        targets = testset.pull_anno(batch_i)[1]

        for label in targets:
            all_data += 1
            pt_label = label[0:4]

            mask_label = np.zeros((640,480), np.float64)
            mask_label[int(pt_label[0]):int(pt_label[2]),int(pt_label[1]):int(pt_label[3])] = 1
            _bool = False
            for i in range(detections.size(1)):
                j = 0

                while detections[0,i,j,0] > conf_thres:
                    score = detections[0,i,j,0]
                    pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                    mask_predict = np.zeros((640,480), np.float64)
                    mask_predict[int(pt[0]):int(pt[2]),int(pt[1]):int(pt[3])] = 1

                    _and = sum(sum(np.logical_and(mask_label, mask_predict)))
                    _or = sum(sum(np.logical_or(mask_label, mask_predict)))
                    _iou = float(_and/_or)
                    # print (_iou)
                    if score >= conf_thres :
                        if (i-1) == label[4] and not _bool and _iou >= iou_thres:
                            _bool = True
                            result_list[0] += 1
                        elif (i-1) != label[4] or _iou <= iou_thres:
                            print (i-1, label[4], _iou)
                            result_list[2] += 1
                    j+=1

                
            if not _bool:
                result_list[1] += 1
                
    print (result_list)
    print (all_data)
    print ("Recall : ", float(result_list[0]/all_data))
    print ("Precision : ", float(result_list[0]/(result_list[0] + result_list[2])))



if __name__ == '__main__':
    eval()

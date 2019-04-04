#!/usr/bin/env python
#
#   Copyright EAVISE
#   Perform image detection with Yolo network
#

import os
import argparse
import logging
import cv2
import torch
from torchvision import transforms as tf
import brambox.boxes as bbb
from brambox.boxes.util import visual as vis
import vedanet as vn
import logging as log
import numpy as np
import glob
import json
from cfg_parser import getConfig

# log = logging.getLogger('lightnet.detect')


# Functions
def create_network():
    """ Create the vedanet network """
    net = vn.models.Yolov3(num_classes=CLASSES, weights_file=args.weight, train_flag=2, test_args={'conf_thresh': CONF_THRESH, 'labels': LABELS, 'network_size': NETWORK_SIZE, 'nms_thresh': NMS_THRESH})
    net.eval()
    net = net.to(DEVICE)
    return net


def detect(net, img_path):
    """ Perform a detection """
    # Load image
    img = cv2.imread(img_path)
    out = None

    if img is not None:
        im_h,im_w = img.shape[:2]
        img_tf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tf = vn.data.transform.Letterbox.apply(img_tf, dimension=NETWORK_SIZE)
        img_tf = tf.ToTensor()(img_tf)
        img_tf.unsqueeze_(0)
        img_tf = img_tf.to(DEVICE)
    # Run detector
        with torch.no_grad():
            out = net(img_tf)
            out = vn.data.transform.ReverseLetterbox.apply(out[0], NETWORK_SIZE, (im_w, im_h)) # Resize bb to true image dimensions
    return img, out


# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an image through the lightnet yolo network')
    parser.add_argument('-weight', help='Path to weight file')
    parser.add_argument('-all_dirs', help='Path to image directories', nargs='*')
    parser.add_argument('-image_dir', help='Path to image dir, to be used if all_dirs is not specified', nargs='?')
    parser.add_argument('-output_dir', help='Path to output dir, current_dir to be used if no output is specified', nargs='?')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    args = parser.parse_args()

    print("Initializing Annotation Pipeline ... \o/ \o/ ...")
    tanns = []

    # Parse Arguments
    if args.cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            DEVICE = torch.device('cuda')
        else:
            log.error('CUDA not available')
    
    # Parameters
    cfgs_root = 'cfgs'
    cur_cfg = getConfig(cfgs_root, 'Yolov3')
    NETWORK_SIZE = cur_cfg['detect']['input_shape']
    LABELS = cur_cfg['labels']
    CLASSES = len(LABELS)
    CONF_THRESH = cur_cfg['detect']['conf_thresh']
    NMS_THRESH = cur_cfg['detect']['nms_thresh']

    # Network
    network = create_network()

    # Detection
    if args.all_dirs:
        if len(args.all_dirs) > 0:
            pass
    
    elif len(args.image_dir) > 0:
        if not os.path.isdir(args.image_dir):
            raise ValueError('image_dir must be a directory')
        else:
            exts = ['*.jpg', '*.png', '*.jpeg']
            imgs = [f for ext in exts for f in glob.glob(os.path.join(args.image_dir, ext))] 

            for img_name in imgs:
                img_name = os.path.join(args.image_dir, img_name)
                log.info(img_name)
                anns = []
                
                _, output = detect(network, img_name)

                if output:          
                    for detection in output[0]:
                        d = {}
                        d['category'] = detection.class_label
                        d['box2d'] = {'x1': detection.x_top_left,
                                    'y1': detection.y_top_left,
                                    'x2': detection.x_top_left-1 + detection.width,
                                    'y2': detection.y_top_left-1 + detection.height}
                        d['manualAttributes'] = False
                        d['manual'] = False

                        if detection.confidence >= CONF_THRESH:
                            anns.append(d)
                    tanns.append((img_name, anns))

            if args.output_dir:
                # Save tanns
                print(f"Saving Annotations to {args.output_dir}")
                annotations_file = os.path.join(args.output_dir, 'annotations.json')
            else:
                print(f"Saving Annotations to {os.getcwd()}")
                annotations_file = os.path.join(os.getcwd(), 'annotations.json')
            
            with open(annotations_file, 'w+') as fn:
                json.dump(tanns, fn)

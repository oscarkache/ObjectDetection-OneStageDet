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
from cfg_parser import getConfig

# log = logging.getLogger('lightnet.detect')

NETWORK_SIZE = [416, 416]
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    log.debug('CUDA enabled')
    DEVICE = torch.device('cuda')
else:
    log.error('CUDA not available')


# Functions
def create_network():
    """ Create the vedanet network """
    net = vn.models.Yolov3(num_classes=CLASSES, weights_file=args.weight, train_flag=2, test_args={'conf_thresh': CONF_THRESH, 'labels': LABELS, 'network_size': NETWORK_SIZE, 'nms_thresh': NMS_THRESH})
    net.eval()
    net = net.to(DEVICE)

    return net


def detect(net, img_path, network_size = NETWORK_SIZE, device = DEVICE):
    """ Perform a detection """
    # Load image
    img = cv2.imread(img_path)
    out = None

    if img is not None:
        im_h,im_w = img.shape[:2]
        img_tf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tf = vn.data.transform.Letterbox.apply(img_tf, dimension=network_size)
        img_tf = tf.ToTensor()(img_tf)
        img_tf.unsqueeze_(0)
        img_tf = img_tf.to(device)





    # Run detector
        with torch.no_grad():
            out = net(img_tf)
            out = vn.data.transform.ReverseLetterbox.apply(out[0], network_size, (im_w, im_h)) # Resize bb to true image dimensions
    return img, out


class Yolov3Annotator():
    def __init__(self, weights_file, cfgs_root = 'cfgs'):
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            self.device = torch.device('cuda')
        else:
            log.error('CUDA not available')
        
        # Parameters
        self.cur_cfg = getConfig(cfgs_root, 'Yolov3')
        self.network_size = self.cur_cfg['detect']['input_shape']
        self.labels = self.cur_cfg['labels']
        self.num_classes = len(self.labels)
        self.conf_thresh = self.cur_cfg['detect']['conf_thresh']
        self.nms_thresh = self.cur_cfg['detect']['nms_thresh']
        self.weights_file = weights_file

        # Network
        self.network = self.create_yolo()


    # Functions
    def create_yolo(self):
        """ Create the vedanet network """
        net = vn.models.Yolov3(num_classes=self.num_classes, weights_file=self.weights_file, train_flag=2, 
                                test_args={'conf_thresh': self.conf_thresh, 'labels': self.labels, 
                                'network_size': self.network_size, 'nms_thresh': self.nms_thresh})
        net.eval()
        net = net.to(self.device)

        return net


    def annotate(self, image_dir, cfgs_root = 'cfgs', detection_threshold = 0.25):
        print("Initializing Annotation Pipeline ... \o/ \o/ ...")
        tanns = []
        if not os.path.isdir(image_dir):
            raise ValueError('image_dir must be a directory')
        else:
            exts = ['*.jpg', '*.png', '*.jpeg']
            imgs = [f for ext in exts for f in glob.glob(os.path.join(image_dir, ext))] 

            for img_name in imgs:
                anns = []
                
                _, output = detect(self.network, img_name, self.network_size, device = self.device)

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

                        if detection.confidence >= detection_threshold:
                            anns.append(d)
                    tanns.append((img_name, anns))
            return tanns

# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an image through the lightnet yolo network')
    parser.add_argument('-weight', help='Path to weight file')
    parser.add_argument('-image', help='Path to image file(s)', nargs='*')
    parser.add_argument('-image_dir', help='Path to image dir, to be used if no image is specified', nargs='?')
    parser.add_argument('-output_dir', help='Path to output dir, current_dir to be used if no output is specified', nargs='?')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-s', '--save', action='store_true', help='Save image in stead of displaying it')
    parser.add_argument('-l', '--label', action='store_true', help='Print labels and scores on the image')
    args = parser.parse_args()

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
    if args.image:
        if len(args.image) > 0:
            for img_name in args.image:
                log.info(img_name)
                image, output = detect(network, img_name)
                if output:
                    image = vis.draw_boxes(image, output[0], show_labels=args.label)
                    if args.save:
                        if args.output_dir:
                            cv2.imwrite( f'{os.path.join(args.output_dir, os.path.splitext(os.path.split(img_name)[-1])[0])}_detections.png', image)
                        else:
                            cv2.imwrite(f'{os.path.splitext(os.path.split(img_name)[-1])[0]}_detections.png', image)
                    else:
                        cv2.imshow('image', image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
    elif len(args.image_dir) > 0:
        images = os.listdir(args.image_dir)
        for img_name in images:
            img_name = os.path.join(args.image_dir, img_name)
            log.info(img_name)
            image, output = detect(network, img_name)
            if output:
                image = vis.draw_boxes(image, output[0], show_labels=args.label)
                if args.save:
                    if args.output_dir:
                        cv2.imwrite( f'{os.path.join(args.output_dir, os.path.splitext(os.path.split(img_name)[-1])[0])}_detections.png', image)
                    else:
                        cv2.imwrite(f'{os.path.splitext(os.path.split(img_name)[-1])[0]}_detections.png', image)
                else:
                    cv2.imshow('image', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    else:
        while True:
            try:
                img_path = input('Enter image path: ')    
            except (KeyboardInterrupt, EOFError):
                print('')
                break
        
            if not os.path.isfile(img_path):
                log.error(f'\'{img_path}\' is not a valid path')
                break

            image, output = detect(network, img_path)
            image = vis.draw_boxes(image, output[0], show_labels=args.label)
            if args.save:
                cv2.imwrite('detections.png', image)
            else:
                cv2.imshow('image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
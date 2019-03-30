import os
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
from .. import loss
from .yolo_abc import YoloABC
from ..network import backbone
from ..network import head

import vedanet as vn

__all__ = ['Yolov3']


class Yolov3(YoloABC):
    def __init__(self, num_classes=20, weights_file=None, input_channels=3,
                 #anchors=[(6,10), (14,24), (27,43), (32,97), (56,64), (92,108), (73,175), (141,178),  (144,291)], #bdd
                 anchors=[(6,8), (14,21), (25,39), (33,91), (54,57), (90,101), (71,170), (139,170),  (141,283)],
                 anchors_mask=[(6,7,8), (3,4,5), (0,1,2)], train_flag=1, clear=False, test_args=None):
        """ Network initialisation """
        super().__init__()

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.nloss = len(self.anchors_mask)
        
        self.test_args = test_args

        self.loss = None
        self.postprocess = None

        num_anchors_list = [len(x) for x in anchors_mask]
        in_channels_list = [512, 256, 128]

        self.backbone = backbone.Darknet53()
        self.head = head.Yolov3(num_classes, in_channels_list, num_anchors_list)

        if weights_file is not None:
            self.load_weights(weights_file, clear)
        else:
            self.init_weights(slope=0.1)



    def _forward(self, x):
        middle_feats = self.backbone(x)
        features = self.head(middle_feats)
        loss_fn = loss.YoloLoss
        
        self.compose(x, features, loss_fn)

        return features

    def modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.

        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self

        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential, backbone.Darknet53,
                backbone.Darknet53.custom_layers, head.Yolov3, head.Yolov3.custom_layers)):
                yield from self.modules_recurse(module)
            else:
                yield module

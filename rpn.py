from torch import nn
import numpy as np
from utils import calculate_iou_scores

class AnchorGenerator():
    def __init__(self, pixel, scales, aspect_ratios, anchor_num=9, stride=16):
        self.pixel = pixel
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.anchor_num = anchor_num
        self.stride = stride
    
    def xywh2xyxy(self, input):
        input_xyxy = input.copy()
        input_xyxy[...,0] = input[..., 0] - input[..., 2]/2
        input_xyxy[...,1] = input[..., 1] - input[..., 3]/2
        input_xyxy[...,2] = input[..., 0] + input[..., 2]/2
        input_xyxy[...,3] = input[..., 1] + input[..., 2]/2
        return input_xyxy
    
    def generate_anchor(self, height, width):
        anchor_x = np.arange(int(self.stride/2), int(width-self.stride/2), self.stride)
        anchor_y = np.arange(int(self.stride/2), int(height-self.stride/2), self.stride)
        x_coord, y_coord = np.meshgrid(anchor_x, anchor_y, indexing='xy')
        anchors = np.stack([x_coord, y_coord], axis=2)
        anchors = np.repeat(anchors[:,:,None,:], repeats=self.anchor_num, axis=2)
        return anchors
    
    def filter_out_anchors(self, anchors, gt_bbox_xyxy, height, width):
        anchors_xyxy = self.xywh2xyxy(anchors)        
        anchors_xyxy = anchors_xyxy.reshape(-1, 4)
        iou_scores = calculate_iou_scores(anchors_xyxy, gt_bbox_xyxy)
        anchors_w_highest_iou = np.zeros(len(anchors_xyxy))
        negative_label = np.where(np.sum(iou_scores < 0.3, axis=1) == len(gt_bbox_xyxy))[0]
        np.put_along_axis(anchors_w_highest_iou, negative_label, -1, axis=0)
        highest_iou_idx = np.argmax(iou_scores, axis=0)
        highest_iou_idx = np.concatenate([highest_iou_idx, np.where(iou_scores > 0.7)[0]], axis=0)
        np.put_along_axis(anchors_w_highest_iou, highest_iou_idx, 1, axis=0)
        
        # filter out boxes near the boundaries
        filter_boxes = np.sum((anchors[...,:2] - anchors[...,2:4] / 2) < 0, axis=3)
        filter_boxes += (anchors[:,:,:,0] + anchors[:,:,:,2] / 2 > width).astype(filter_boxes.dtype)
        filter_boxes += (anchors[:,:,:,1] + anchors[:,:,:,3] / 2 > height).astype(filter_boxes.dtype)
        boundaries = np.where((filter_boxes != 0).reshape(anchors_w_highest_iou.shape))[0]
        np.put_along_axis(anchors_w_highest_iou, boundaries, 0, axis=0)
        
        anchors = np.concatenate([anchors, anchors_w_highest_iou.reshape(*anchors.shape[:-1], -1)], axis=-1)
        return anchors      
    
    def generate_anchor_boxes(self, height, width, gt_bbox_xyxy):
        scales_and_ratios = np.stack(np.meshgrid(self.scales, self.aspect_ratios), axis=2).reshape(-1, 2)
        scales_and_ratios[:,0] *= self.pixel
        anchors = self.generate_anchor(height, width)
        anchor_widths = scales_and_ratios[:,0] * scales_and_ratios[:,1]
        anchor_heights = scales_and_ratios[:,0] / scales_and_ratios[:,1]
        anchor_whs = np.stack([anchor_widths, anchor_heights], axis=1)
        anchor_whs = np.tile(anchor_whs, (*anchors.shape[:2], 1, 1))
        anchors = np.concatenate([anchors, anchor_whs], axis=3)
        anchors = self.filter_out_anchors(anchors, gt_bbox_xyxy, height, width)
        return anchors        
        
    
    

class RPN(nn.Module):

    def __init__(self, in_channels=512, anchor_num=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, 2*anchor_num, kernel_size=1, stride=1)
        self.reg_layer = nn.Conv2d(in_channels, 4*anchor_num, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv(input)
        cls_logit = self.cls_layer(x)
        cls_score = self.softmax(cls_logit)
        reg_output = self.reg_layer(x)
        return cls_score, reg_output
    
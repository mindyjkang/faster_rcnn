import os
import cv2
import numpy as np

import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(self, datadir, datatype, img_size=600):
        super(Dataset, self).__init__()
        self.datadir = datadir
        self.datatype = datatype
        self.img_size = img_size
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.img_size, interpolation = InterpolationMode.BILINEAR),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.annfile = os.path.join(self.datadir, 'annotations', f'instances_{self.datatype}.json')
        self.coco = COCO(self.annfile)       
        self.img_keys = list(self.coco.imgs.keys())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.names = [cat['name'] for cat in self.cats]
        self.names = list(set(self.names))
        self.cls_ids = [cat['id'] for cat in self.cats]
        self.cls_id_dics = {
            cls_id: i for i, cls_id in enumerate(self.cls_ids)
        }        
        
    def __len__(self):
        return len(self.coco.imgs)
    
    def __getitem__(self, index):
        img_id = self.img_keys[index]
        coco_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.datadir, self.datatype, coco_info['file_name'])

        img = cv2.imread(img_path)
        org_img = img.copy()
        
        # preprocess
        img = torch.tensor(img.transpose((2, 1, 0)), dtype=torch.float32)
        img = self.transforms(img)
        img = img.unsqueeze(0)

        # get gt bboxes
        annIds = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(annIds)
        gt_bbox = np.array([[self.cls_id_dics[ann['category_id']], *ann['bbox']] for ann in anns])
        gt_bbox_xyxy = gt_bbox.copy()
        gt_bbox_xyxy[:, 3] += gt_bbox_xyxy[:, 1]
        gt_bbox_xyxy[:, 4] += gt_bbox_xyxy[:, 2]

        return img, gt_bbox, org_img, gt_bbox_xyxy

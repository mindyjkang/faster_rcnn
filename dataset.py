import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoDataset(Dataset):
    def __init__(self, datadir, datatype):
        super(Dataset, self).__init__()
        self.datadir = datadir
        self.datatype = datatype
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
        annIds = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(annIds)
        gt_bbox = np.array([[self.cls_id_dics[ann['category_id']], *ann['bbox']] for ann in anns])
        return img, gt_bbox

import os
import glob
import cv2
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils import DataLoader, Dataset


class DataModule(DataLoader):
    def __init__(self, cfg):
        self.data_path = cfg['data_path']
        self.test_path = cfg['test_path']
        self.batch_size = cfg['batch_size']
        self.num_workers = cfg['num_workers']
        self.trn_names, self.val_names = self.gen_splits(
            cfg['random_seed'], cfg['val_ratio'], cfg['stratify'])
        
    def gen_splits(self, random_seed, val_ratio, stratify):
        all_img_names = list(glob.glob(f'{self.data_path}/*/*'))
        if stratify:
            label_list = [name.split('/') for name in all_img_names]
        else:
            label_list = None
        trn_names, val_names = train_test_split(
            all_img_names, test_size=val_ratio, random_state=random_seed,
            stratify=label_list
            )
        return trn_names, val_names
    
    def train_dataloader(self):
        trn_dataset = ImageLoader(self.data_path, self.trn_names, 'train')
        trn_loader = DataLoader(trn_dataset, self.batch_size, shuffle=True,
                                num_workers=self.num_workers)
        return trn_loader
    
    def val_dataloader(self):
        val_dataset = ImageLoader(self.data_path, self.val_names, 'val')
        val_loader = DataLoader(val_dataset, self.batch_size, shuffle=False,
                                num_workers=self.num_workers)
        return val_loader
    
    def test_dataloader(self):
        test_list = list(glob.glob(f'{self.test_path}/*'))
        test_dataset = ImageLoader(self.test_path, test_list, 'test')
        test_loader = DataLoader(test_dataset, 1, shuffle=False,
                                 num_workers=self.num_workers)
        return test_loader


class ImageLoader(Dataset):
    def __init__(self, data_path, img_list, set_type):
        self.data_path = data_path
        self.img_list = img_list
        self.set_type = set_type
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, self.img_list[idx])
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
import os
import glob
import cv2
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils import DataLoader, Dataset


class DataModule(pl.LightningDataModule):
    """
    Lightning module that abstracts the creation of dataloaders for
    train, val and test sets. By passing the configuration file to a
    single class, the creation of train-val splits and dataloaders can
    be handled.
    """
    def __init__(self, cfg):
        self.data_path = cfg['data_path']
        self.test_path = cfg['test_path']
        self.batch_size = cfg['batch_size']
        self.num_workers = cfg['num_workers']
        self.trn_names, self.val_names = self.gen_splits(
            cfg['random_seed'], cfg['val_ratio'], cfg['stratify'])
        self.test_labels_avl = cfg['test_labels_avl']

    def gen_splits(self, random_seed, val_ratio, stratify):
        """
        Obtain train-validation splits.
        Args:
            random_seed (int): For reproducability
            val_ratio (float): Percentage of val data, between 0-1
            stratify (bool): Whether to do stratified sampling based on labels
        Returns:
            trn_names (list[str]): list of image names in train set
            val_names (list[str]): list of images names in val set
        """
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
        """ Creates and returns train set dataloader """
        trn_dataset = ImageLoader(self.data_path, self.trn_names, 'train')
        trn_loader = DataLoader(trn_dataset, self.batch_size, shuffle=True,
                                num_workers=self.num_workers)
        return trn_loader
    
    def val_dataloader(self):
        """ Creates and returns val set dataloader """
        val_dataset = ImageLoader(self.data_path, self.val_names, 'val')
        val_loader = DataLoader(val_dataset, self.batch_size, shuffle=False,
                                num_workers=self.num_workers)
        return val_loader
    
    def test_dataloader(self):
        """ Creates and returns test set dataloader """
        test_list = list(glob.glob(f'{self.test_path}/*'))
        test_dataset = ImageLoader(self.test_path, test_list, 'test',
                                   self.test_labels_avl)
        test_loader = DataLoader(test_dataset, 1, shuffle=False,
                                 num_workers=self.num_workers)
        return test_loader


class ImageLoader(Dataset):
    """
    Data loader for loading images one at a time
    """
    def __init__(self, data_path, img_list, set_type, test_labels_avl=False):
        """
        Args:
            data_path (str): path to dataset
            img_list (list[str]): list of images in the selected set
            set_type (str): 'train', 'val' or 'test'
            test_labels_avl (bool): whether ground truth is available for test
                set.
        """
        self.data_path = data_path
        self.img_list = img_list
        self.set_type = set_type
        self.test_labels_avl = test_labels_avl
        
    def __len__(self):
        """ Returns length of dataset """
        return len(self.img_list)
    
    def __getitem__(self, idx):
        """
        Load each image from given path and do following -
        1. BGR to RGB (since opencv used)
        2. Normalization to ImageNet Mean/Std
        3. Augmentation if set
        4. Extract and return label (unless not available in test)
        PyTorch handles the conversion to Tensor and CUDA
        """
        img_name = os.path.join(self.data_path, self.img_list[idx])
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.test_label_avl and self.set_type == 'test':
            return img
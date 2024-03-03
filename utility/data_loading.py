import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image


class LoadDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.ids = [filename.stem for filename in self.images_dir.glob('*.png' or '*.jpg')]

    def __len__(self):
        return len(self.ids)
            
    @classmethod
    def load(cls, filename, is_mask = True):
        img = Image.open(filename)
        img_ndarray = np.asarray(img)
        if is_mask == False:
            img_ndarray = img_ndarray[np.newaxis, ...] if img_ndarray.ndim == 2 else img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255
        return img_ndarray 

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))
        mask_file = list(self.masks_dir.glob(name + '.*'))
        
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
                
        img = self.load(img_file[0], is_mask = False)
        mask = self.load(mask_file[0])
        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
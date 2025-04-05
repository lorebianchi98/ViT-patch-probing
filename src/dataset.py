import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image
from tqdm import tqdm
class PascalVOCWrapper(Dataset):
    """
    PyTorch Dataset wrapper for the PascalVOCDataset.
    """

    def __init__(self, data_root, datapath, image_transforms, resize_dim=518, patch_size=14, **kwargs):
        self.data = {} 
        
        # finding ids of the set 
        with open(os.path.join(data_root, datapath), 'r') as file:
            image_ids = file.readlines()
        image_ids = [line.strip() for line in image_ids]
        n_images = len(image_ids)
        
        for idx, image_id in tqdm(enumerate(image_ids), total=n_images):
            if idx >= n_images:
                break
            pil_img = Image.open(os.path.join(data_root, 'JPEGImages', f'{image_id}.jpg'))
            img = (image_transforms(pil_img))
            # getting segmentation map
            seg_map = np.array(Image.open(os.path.join(data_root, 'SegmentationClass', f'{image_id}.png')))
            # treating undefined labelled pixels as background
            seg_map[seg_map == 255] = 0 
            # Add a batch and channel dimension (required by F.interpolate)
            seg_map_tensor = torch.tensor(seg_map).unsqueeze(0).unsqueeze(0) 

            # Resize using nearest-neighbor interpolation
            new_size = (resize_dim, resize_dim)
            resized_seg_map = F.interpolate(seg_map_tensor, size=new_size, mode='nearest').squeeze(0).squeeze(0)
            
            n_patch_per_side = resize_dim // patch_size
            # Reshape the tensor into blocks of shape (n_patch_per_side, patch_size, n_patch_per_side, patch_size)
            # This groups pixels into patch_size x patch_siz blocks
            blocks = resized_seg_map.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)

            # The shape of 'blocks' will be (n_patch_per_side, n_patch_per_side, patch_size, patch_size)
            # Flatten the last two dimensions (patch_size x patch_size) into one for easier processing
            blocks = blocks.contiguous().view(n_patch_per_side, n_patch_per_side, -1)  # Shape: (n_patch_per_side, n_patch_per_side, patch_size**2)
            
            # keeping the most frequent element for each grid            
            reduced_grid, _ = torch.mode(blocks, dim=2)
            
            # storing image info
            self.data[idx] = {}
            self.data[idx]['img'] = img
            self.data[idx]['grid_segmentation_map'] = reduced_grid
            self.data[idx]['metadata'] = {
                'image_id': image_id,
                'original_shape': pil_img.size
            }
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index not in self.data:
            raise IndexError(f"Index {index} out of range.")
        
        return {
            'img': self.data[index]['img'],
            'grid_segmentation_map': self.data[index]['grid_segmentation_map'],
            'metadata': self.data[index]['metadata']
        }
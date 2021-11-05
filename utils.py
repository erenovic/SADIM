from torch.utils.data import Dataset
import torch

from PIL import Image
from natsort import natsorted
import glob

from torch.nn.functional import interpolate

# For images1024x1024

class SingleResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=512):
        self.frames = natsorted(glob.glob(path + "*.png"))

        self.resolution = resolution
        self.transform = transform
        self.length = len(self.frames)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.frames[index]
        img = Image.open(img_path)
        img = self.transform(img)
        img = torch.unsqueeze(img, 0)
        img = interpolate(img, size=(self.resolution, self.resolution), 
                          mode="bilinear", align_corners=False)
        img = torch.squeeze(img, 0)

        return img

# For thumbnails128x128

#class SingleResolutionDataset(Dataset):
#    def __init__(self, path, transform, resolution=128):
#        self.frames = natsorted(glob.glob(path + "*.*"))
#
#        self.resolution = resolution
#        self.transform = transform
#        self.length = len(self.frames)
#
#    def __len__(self):
#        return self.length
#
#    def __getitem__(self, index):
#        img_path = self.frames[index]
#        img = Image.open(img_path)
#        img = self.transform(img)
#
#        return img
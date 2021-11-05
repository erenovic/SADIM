
from torch.utils.data import Dataset

from PIL import Image
from natsort import natsorted
import glob


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=128):
        self.frames = natsorted(glob.glob(path + "*.*"))

        self.resolution = resolution
        self.transform = transform
        self.length = len(self.frames)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.frames[index]
        img = Image.open(img_path)
        img = self.transform(img)

        return img


import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import make_grid

from model.encoder import Encoder
from model.generator import Generator
# from utils import SingleResolutionDataset


# -----------------------------Dataset for test------------------------------------------------
from torch.utils.data import Dataset

from PIL import Image
from natsort import natsorted
import glob

from torch.nn.functional import interpolate

# For images1024x1024

class SingleResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=512, ext="png"):
        self.frames = natsorted(glob.glob(path + "*." + ext))

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
# ----------------------------------------------------------------------------------------


def disp_img(tensor, im_dim, name):
    dpi=80
    two_img = 255 * make_grid(tensor, nrow=2, normalize=True, value_range=(-1, 1))
    img = np.array(two_img.detach().cpu()).astype(np.uint8).transpose(1, 2, 0)
    h, w, c = img.shape
    figsize = w / float(dpi), h / float(dpi)
    fig = plt.figure(figsize=figsize)
    plt.imshow(img, aspect=1)
    fig.savefig(name)
    
    
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                     inplace=True)])
im_dim = 512
device = torch.device("cuda")
pretrained14 = torch.load("/scratch/users/ecetin17/Swapping-2-GPU/checkpoint/230000.pt")
pretrained34 = torch.load("/scratch/users/ecetin17/Swapping-dif-patch-size/checkpoint/190000.pt")


dataset = SingleResolutionDataset("/datasets/CelebAMask-HQ/CelebA-HQ-img/", transform, im_dim, "jpg")
sampler = RandomSampler(dataset)
dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)


encoder14 = Encoder(32).to(device)
generator14 = Generator(32).to(device)
msg_e = encoder14.load_state_dict(pretrained14["e_ema"])
msg_g = generator14.load_state_dict(pretrained14["g_ema"])

print(msg_e)
print(msg_g)

encoder34 = Encoder(32).to(device)
generator34 = Generator(32).to(device)
msg_e = encoder34.load_state_dict(pretrained34["e_ema"])
msg_g = generator34.load_state_dict(pretrained34["g_ema"])

print(msg_e)
print(msg_g)

encoder14.requires_grad = False
generator14.requires_grad = False

encoder14 = encoder14.eval()
generator14 = generator14.eval()

encoder34.requires_grad = False
generator34.requires_grad = False

encoder34 = encoder34.eval()
generator34 = generator34.eval()

imgs = next(iter(dataloader)).to(device)
img1, img2 = imgs.chunk(2, dim=0)

struct1_14, text1_14 = encoder14(img1)
struct2_14, text2_14 = encoder14(img2)

struct1_34, text1_34 = encoder34(img1)
struct2_34, text2_34 = encoder34(img2)

disp_img(torch.cat((img1, img2), dim=0), im_dim, "original.png")

exact1_14 = generator14(struct1_14, text1_14)
exact2_14 = generator14(struct2_14, text2_14)

disp_img(torch.cat((exact1_14, exact2_14), dim=0), im_dim, "exact_1_4.png")

exact1_34 = generator34(struct1_34, text1_34)
exact2_34 = generator34(struct2_34, text2_34)

disp_img(torch.cat((exact1_34, exact2_34), dim=0), im_dim, "exact_3_4.png")

text_interp_14 = [torch.lerp(text1_14, text2_14, weight.to(device)) for weight in torch.linspace(0, 2, 5)]
img_interp_14 = [generator14(struct1_14, text) for text in text_interp_14]

disp_img(torch.cat(img_interp_14, dim=3), im_dim, "traverse_1_4.png")

text_interp_34 = [torch.lerp(text1_34, text2_34, weight.to(device)) for weight in torch.linspace(0, 2, 5)]
img_interp_34 = [generator34(struct1_34, text) for text in text_interp_34]

disp_img(torch.cat(img_interp_34, dim=3), im_dim, "traverse_3_4.png")









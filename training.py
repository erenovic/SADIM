
########################## REFERENCES #############################

# https://github.com/rosinality/swapping-autoencoder-pytorch

###################################################################

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# I have some problems in this script !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import torch
from torch import autograd
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.distributed import get_rank
from torchvision import transforms
from torchvision.utils import save_image

import random
import logging
import argparse
import os
import wandb

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from model.encoder import Encoder
from model.discriminator import Discriminator
from model.generator import Generator
from model.cooccur import CooccurDiscriminator

from stylegan2.distributed import synchronize, get_rank, reduce_loss_dict
from utils import SingleResolutionDataset


def data_sampler(dataset, shuffle, distributed):
    """Building the DataSampler for the sampling in the DataLoader"""
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return RandomSampler(dataset)

    else:
        return SequentialSampler(dataset)


def requires_grad(model, flag=True):
    """Turn requires_grad on to save the gradient graph"""
    for p in model.parameters():
        p.requires_grad = flag


def set_grad_none(model, targets):
    """Set the gradients of chosen modules in the model to None"""
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def accumulate(model1, model2, decay=0.999):
    """
    I believe we use this to prevent rapid fluctuations
    during the training of GAN. We don't allow large updates
    but onlu make slight updates in the parameters wrt. the
    decay parameter
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    """Wrapper function for the DataLoader"""
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    """
    Standard adversarial loss function used on real,
    reconstructed and hybrid images
    """
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    """
    Calculate the R1 loss only on the real images to
    ensure that the discriminator cannot create a non-zero
    gradient orthogonal to the data manifold without suffering
    """
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    """
    Loss function for the generator to fool discriminator,
    we use non-saturating version to prevent saturation and
    update the encoder and the generator
    """
    loss = F.softplus(-fake_pred).mean()

    return loss


def patchify(image, n_crops, min_size=1/8, max_size=1/4):
    """
    Return n_crops random patches from the image where
    patches are stacked in the batch dimension

    The size of each patch is between 1/8th and 1/4th of
    the image. At the end, each image is interpolated using
    bilinear interpolation to the same target size
    """

    b, c, h, w = image.shape
    target_h = int(h * max_size)
    target_w = int(w * max_size)

    crop_size_vars = torch.rand(n_crops)
    crop_sizes = crop_size_vars * (max_size - min_size) + min_size

    crop_h = (crop_sizes * h).type(torch.int64).tolist()
    crop_w = (crop_sizes * w).type(torch.int64).tolist()

    cropped_images = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(0, h - c_h)
        c_x = random.randrange(0, w - c_w)

        cropped_image = image[:, :, c_y: c_y + c_h, c_x: c_x + c_w]
        interp_cropped_image = F.interpolate(cropped_image,
                                             size= (target_h, target_w),
                                             mode="bilinear",
                                             align_corners=False)
        cropped_images.append(interp_cropped_image)

    return torch.stack(cropped_images, dim=1).view(-1, c, target_h, target_w)


def train(args, dataloader, encoder, generator, discriminator, cooccur,
          g_optim, d_optim, e_ema, g_ema, device):
    """
    1. Update the discriminator and the cooccurence modules on
    generated images using d loss + cooccurrence loss,
    2. If applicable, update the discriminator on the real
    images using r1 loss,
    3. Update the encoder and generator using
    reconstruction loss + gan loss + cooccurrence loss
    """

    # Get the dataloader with the wrapper,
    loader = sample_data(loader=dataloader)

    # Set the progression bar if we are using a non-distributed setup ??????????????
    pbar = range(args.iter)

    # if get_rank() == 0:
    #     pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # The loss values are initialized to 0
    d_loss_val = 0
    r1_loss = torch.zeros((1), device=args.device)
    g_loss_val = 0
    loss_dict = {}

    # Rename the modules of the model since we might use
    # a distributed setup for faster training
    if args.distributed:
        e_module = encoder.module
        g_module = generator.module
        d_module = discriminator.module
        c_module = cooccur.module

    else:
        e_module = encoder
        g_module = generator
        d_module = discriminator
        c_module = cooccur

    # ?????????????????????????????????????????????????????????????
    # I could not understand accum !!!!!!!!!!!!!!!!!!!!!
    accum = 0.5 ** (32 / (10 * 1000))
    # ?????????????????????????????????????????????????????????????

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            logging.info("Training Completed.")
            break

        # Get the random batch of images, N images
        real_img = next(loader)
        real_img = real_img.to(device)

        # N/2 real_img1, N/2 real_img 2
        real_img1, real_img2 = real_img.chunk(2, dim=0)

        # At each step, we first train the discriminator and
        # the cooccurrance discriminator
        requires_grad(encoder, False)
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(cooccur, True)

        # Get the latent representations through the encoder
        structure1, texture1 = encoder(real_img1)
        _, texture2 = encoder(real_img2)

        # Produce the fake images through the generator
        fake_img1 = generator(structure1, texture1)
        fake_img2 = generator(structure1, texture2)

        # Get the discriminator scores for
        # hybrid images (stack in batch dimension) & real images
        # Calculate the adversarial loss using scores
        fake_pred = discriminator(torch.cat((fake_img1, fake_img2), dim=0))
        real_pred = discriminator(real_img)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        # Get the patches from fake and real images to
        # calculate the co-occurrance loss using
        # co-occurrance discriminator and adversarial
        # loss function. Train it once with the
        # fake patches and once with the real patches
        fake_patches = patchify(fake_img2, args.n_crop)
        real_patches = patchify(real_img2, args.n_crop)
        ref_patches = patchify(real_img2, args.n_crop * args.ref_crop)

        fake_patch_pred, ref_input = cooccur(fake_patches, ref_patches,
                                            ref_batch=args.ref_crop)
        real_patch_pred, _ = cooccur(real_patches, ref_input=ref_input)
        cooccur_loss = d_logistic_loss(real_patch_pred, fake_patch_pred)

        # Register losses and scores into loss_dict
        loss_dict["d"] = d_loss
        loss_dict["cooccur"] = cooccur_loss
        loss_dict["real_score"] = real_pred.mean()
        fake_pred1, fake_pred2 = fake_pred.chunk(2, dim=0)
        loss_dict["fake_score"] = fake_pred1.mean()
        loss_dict["hybrid_score"] = fake_pred2.mean()

        # Update weights of cooccurrance discriminator and
        # discriminator
        d_optim.zero_grad()
        (d_loss + cooccur_loss).backward()
        d_optim.step()

        # Perform R1 regularization at every args.d_reg_every step
        # not to miss Nash equilibrium, we use the real images for
        # regularization
        # Update both discriminator and the cooccurrance discriminator
        if i % args.d_reg_every == 0:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            real_patches.requires_grad = True
            real_patch_pred, _ = cooccur(real_patches, ref_patches,
                                         ref_batch=args.ref_crop)
            cooccur_r1_loss = d_r1_loss(real_patch_pred, real_patches)

            d_optim.zero_grad()

            # Sum both R1 losses of discriminator and the
            # coocccurrance discriminator
            r1_loss_sum = args.r1 / 2 * r1_loss * args.d_reg_every
            r1_loss_sum += args.cooccur_r1 / 2 * cooccur_r1_loss * args.d_reg_every

            # ??????????????????????????????????????????????????????????????????
            r1_loss_sum += 0 * real_pred[0, 0] + 0 * real_patch_pred[0, 0]
            # ??????????????????????????????????????????????????????????????????

            r1_loss_sum.backward()

            d_optim.step()

        # Register R1 regularization losses
        loss_dict["r1"] = r1_loss
        loss_dict["cooccur_r1"] = cooccur_r1_loss

        # Now update the encoder and generator using the
        # reconstruction losses (both L1 and GAN)
        requires_grad(encoder, True)
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(cooccur, False)

        # We don't use the real images in training
        # of the encoder & the generator
        real_img.requires_grad = False

        # Again, encoding the pair of initial images
        structure1, texture1 = encoder(real_img1)
        _, texture2 = encoder(real_img2)

        fake_img1 = generator(structure1, texture1)
        fake_img2 = generator(structure1, texture2)

        # Targeting exact reconstruction of image 1
        recon_loss = F.l1_loss(fake_img1, real_img1)

        # Loss for hybrid image is non-saturating generator loss
        fake_pred = discriminator(torch.cat((fake_img1, fake_img2), dim=0))
        g_loss = g_nonsaturating_loss(fake_pred)

        fake_patches = patchify(fake_img2, args.n_crop)
        ref_patches = patchify(real_img2, args.n_crop * args.ref_crop)
        fake_patch_pred, _ = cooccur(fake_patches, ref_patches, ref_batch=args.ref_crop)
        g_cooccur_loss = g_nonsaturating_loss(fake_patch_pred)

        loss_dict["recon"] = recon_loss
        loss_dict["g"] = g_loss
        loss_dict["g_cooccur"] = g_cooccur_loss

        g_optim.zero_grad()
        (recon_loss + g_loss + g_cooccur_loss).backward()
        g_optim.step()

        # ????????????????????????????????????????????????????????????????????????
        # I dont understand why we reduce the loss?
        accumulate(e_ema, e_module, accum)
        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)
        # ????????????????????????????????????????????????????????????????????????

        # Get the reduced loss values in float form instead of tensors
        d_loss_val = loss_reduced["d"].mean().item()
        cooccur_val = loss_reduced["cooccur"].mean().item()
        recon_val = loss_reduced["recon"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        g_cooccur_val = loss_reduced["g_cooccur"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        cooccur_r1_val = loss_reduced["cooccur_r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        hybrid_score_val = loss_reduced["hybrid_score"].mean().item()

        if get_rank() == 0:
            # Progression bar shows those values
            # pbar.set_description(
            #     (
            #         f"d: {d_loss_val:.4f}; c: {cooccur_val:.4f} g: {g_loss_val:.4f}; "
            #         f"g_cooccur: {g_cooccur_val:.4f}; recon: {recon_val:.4f}; r1: {r1_val:.4f}; "
            #         f"r1_cooccur: {cooccur_r1_val:.4f}"
            #     )
            # )

            # Wandb records the given values
            if wandb and args.wandb and i % 100 == 0:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Cooccur": cooccur_val,
                        "Recon": recon_val,
                        "Generator Cooccur": g_cooccur_val,
                        "R1": r1_val,
                        "Cooccur R1": cooccur_r1_val,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Hybrid Score": hybrid_score_val,
                    },
                    step=i,
                )


            # At every 100th iteration, we save a minibatch of samples.
            # It has N / 2 rows and 2 columns, we tell it to normalize it
            # in [0, 1] and the range was initially [-1, 1] since we
            # normalized images initially with mean 0.5 and scale 0.5
            if i % 1000 == 0:
                with torch.no_grad():
                    e_ema.eval()
                    g_ema.eval()

                    structure1, texture1 = e_ema(real_img1)
                    _, texture2 = e_ema(real_img2)

                    fake_img1 = g_ema(structure1, texture1)
                    fake_img2 = g_ema(structure1, texture2)

                    sample = torch.cat((fake_img1, fake_img2), 0)

                    save_image(sample, f"sample/{str(i).zfill(6)}.png",
                               nrow=int(sample.shape[0] ** 0.5),
                               normalize=True, value_range=(-1, 1))

            # Save the model at every 10,000th iteration
            if i % 10000 == 0:
                torch.save(
                    {
                        "e": e_module.state_dict(),
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "cooccur": c_module.state_dict(),
                        "e_ema": e_ema.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"checkpoint/{str(i).zfill(6)}.pt",
                )

if __name__ == "__main__":

    # Train using GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Increases performance in PyTorch
    torch.backends.cudnn.benchmark = True

    # Argument parser
    parser = argparse.ArgumentParser()

    # Hyperparameters, paths and settings are given
    # prior the training and validation
    parser.add_argument("--path", type=str, default="/datasets/ffhq/images1024x1024/")      # Dataset paths
    parser.add_argument("--iter", type=int, default=800000)             # # of total iterations
    parser.add_argument("--batch", type=int, default=8)                # Batch size
    parser.add_argument("--size", type=int, default=512)                # Initial image sizes
    parser.add_argument("--r1", type=float, default=10)                 # Parameter for R1 regularization
    parser.add_argument("--cooccur_r1", type=float, default=1)          # Parameter for R1 regularization
    parser.add_argument("--ref_crop", type=int, default=4)              # # of ref patches for each fake or real patch
    parser.add_argument("--n_crop", type=int, default=8)                # # of patches for cooccurrance discriminator
    parser.add_argument("--d_reg_every", type=int, default=16)          # R1 regularization in specified interval
    parser.add_argument("--ckpt", type=str, default=None)               # Checkpoint path if we want to load else None
    parser.add_argument("--lr", type=float, default=0.002)              # Learning rate
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--wandb", action="store_true", default=True)   # Use wandb to follow training
    parser.add_argument("--local_rank", type=int, default=0)            # Our local process ID (rank) is 0
    parser.add_argument("--num_workers", type=int, default=3)           # number of workers

    parser.add_argument("--latent", type=int, default=512)              # Latent channel size
    parser.add_argument("--n_mlp", type=int, default=8)                 # Dense layer output size
    parser.add_argument("--start_iter", type=int, default=0)            # Start iterations from 0
    
    logging.basicConfig(filename= "swapping.log", level=logging.INFO)

    # Builds the Namespace which we can use to reach the required parameters
    args = parser.parse_args()
    
    args.device = device

    # Get the number of GPUs available
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    # ????????????????????????????????????????????????????????????????????????????????
    # If we are able to use distribution to increase speed, we set the local
    # process ID (rank) to be 0. We use nccl backend because we perform
    # distributed GPU training. "env://" specifies how to initialize the process
    # group (not quite sure what that means)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    # ????????????????????????????????????????????????????????????????????????????????

    # Initialize encoder, generator, discriminator and cooccurrance discriminator
    encoder = Encoder(args.channel).to(device)
    generator = Generator(args.channel).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    cooccur = CooccurDiscriminator(args.channel, args.size).to(device)

    # ????????????????????????????????????????????????????????????????????????????????
    # We build a second encoder and generator and save our models on them
    # in order to use accumulate function to prevent fluctuations during
    # training with decay value. Initially, we use accumulate with decay=0
    # to equalize e_ema with encoder and g_ema with generator. That makes
    # their parameters equal
    e_ema = Encoder(args.channel).to(device)
    g_ema = Generator(args.channel).to(device)
    e_ema.eval()
    g_ema.eval()
    accumulate(e_ema, encoder, 0)
    accumulate(g_ema, generator, 0)
    # ????????????????????????????????????????????????????????????????????????????????


    # ???????????????????????????????????????????????????????????????????????????
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    # Build the optimizers, not sure what the betas or d_reg_ratio do
    g_optim = optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=args.lr,
        betas=(0, 0.99),
    )
    d_optim = optim.Adam(
        list(discriminator.parameters()) + list(cooccur.parameters()),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    # ???????????????????????????????????????????????????????????????????????????

    # If we have a checkpoint path, load it
    if args.ckpt is not None:
        logging.info("load model: " + args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        encoder.load_state_dict(ckpt["e"])
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        cooccur.load_state_dict(ckpt["cooccur"])
        e_ema.load_state_dict(ckpt["e_ema"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        cooccur = nn.parallel.DistributedDataParallel(
            cooccur,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                         inplace=True)])

    # datasets = []

    # Load multiple datasets using MultiResolutionDataset and use random sampling
    # for path in args.path:
    dataset = SingleResolutionDataset(args.path, transform, args.size)
    # datasets.append(dataset)

    # concat_datasets = ConcatDataset(datasets)
    sampler = data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, drop_last=True, num_workers=args.num_workers)

    # Start recording with wandb
    if (get_rank() == 0) and (wandb is not None) and (args.wandb):
        logging.info("Wandb started...")
        wandb.init(name="Swapping Autoencoder", project="Swapping Autoencoder for Deep Image Manipulation")

    train(args, loader, encoder, generator, discriminator, cooccur,
          g_optim, d_optim, e_ema, g_ema, device)
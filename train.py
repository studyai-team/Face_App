import argparse
from datetime import datetime
import os

# import matplotlib
# matplotlib.use("Agg")

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset_class import VidDataSet, collate_fn
from models import *
# from torchsummary import summary

import torch
import cv2
import numpy as np
from networks import generate_landmarks
from tqdm import tqdm

from loss import VGGLoss, VGGFace

t_start = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_path", type=str, default="/home/sato/D/unzippedFaces", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=5e-5, help="adam: learning rate of generator")
parser.add_argument("--lr_d", type=float, default=5e-5, help="adam: learning rate of discriminator")
parser.add_argument("--lr_e", type=float, default=5e-5, help="adam: learning rate of embedder")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--hr_shape", type=int, default=256, help="training image size 256 or 512")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="batch interval between model checkpoints")
parser.add_argument("--warmup_epochs", type=int, default=0, help="number of epochs with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=1e-2, help="adversarial loss weight")
parser.add_argument("--save_images", default='images', help="where to store images")
parser.add_argument("--save_models", default='saved_models', help="where to save models")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.save_images, exist_ok=True)
os.makedirs(opt.save_models, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))
else:
    device = torch.device('cpu')

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)
embedder = Embedder().to(device)

# # Losses
Loss_L1 = torch.nn.L1Loss().to(device)
Loss_VGG19 = VGGLoss(loss_type="vgg54").to(device)
Loss_VGGFace = VGGFace(model_path="resnet50_ft_weight.pkl").to(device)
Loss_adv = torch.nn.MSELoss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(opt.save_models + "/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load(opt.save_models + "/discriminator_%d.pth" % opt.epoch))
    embedder.load_state_dict(torch.load(opt.save_models + "/embedder_%d.pth" % opt.epoch))

# Optimizers (Learning parameter is different from the original paper)
optimizer_G = torch.optim.Adam(list(generator.parameters()) + list(embedder.parameters()), lr=opt.lr_g, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataset = VidDataSet(size=256, data_path=opt.dataset_path, device=device)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    D_loss = 0
    G_loss = 0

    # pbar = tqdm(total=len(dataloader))
    # dataset is too large, we will skip every 1000 iter
    pbar = tqdm(total=1000)

    for i, data in enumerate(dataloader):
        with torch.set_grad_enabled(True):
            # Configure model input
            source_image = data["source_image"].type(Tensor)
            target_image = data["target_image"].type(Tensor)
            source_landmark = data["source_landmark"].type(Tensor)
            target_landmark = data["target_landmark"].type(Tensor)

            # ------------------
            #  Train Generators
            # ------------------

            # optimizer_E.zero_grad()
            optimizer_G.zero_grad()
            output_1, output_2, output_3, output_4, output_5 = embedder(target_landmark)
            generated_image = generator(source_image, output_1, output_2, output_3, output_4, output_5)

            # Measure loss
            loss_l1 = Loss_L1(target_image, generated_image)
            loss_vgg = Loss_VGG19(target_image, generated_image)
            loss_id = Loss_VGGFace(target_image, generated_image)

            # Extract validity predictions from discriminator
            pred_fake = discriminator(generated_image, target_landmark)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = Loss_adv(pred_fake, torch.ones_like(pred_fake))

            # Total generator loss
            loss_G = loss_GAN + 20 * loss_l1 + 2 * loss_vgg + 0.2 * loss_id

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(source_image, source_landmark)
            pred_fake = discriminator(generated_image.detach(), target_landmark)

            # Total loss
            loss_D = Loss_adv(pred_real, torch.ones_like(pred_real)) + Loss_adv(pred_fake, torch.zeros_like(pred_fake))

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            D_loss += loss_D.item()
            G_loss += loss_G.item()
            pbar.update(1)

            # dataset is too large, we will skip every 1000 iter
            if i >= 1000:
                break

    avg_D_loss = D_loss / len(dataloader)
    avg_G_loss = G_loss / len(dataloader)

    print(
        'Epoch:{1}/{2} D_loss:{3} G_loss:{4} time:{0}'.format(
            datetime.now() - t_start, epoch + 1, opt.n_epochs, avg_D_loss,
            avg_G_loss))
    if (epoch + 1) % opt.sample_interval == 0:
        # Save example results
        img_grid = torch.cat((source_image, target_image, generated_image, target_landmark), -1)
        save_image(img_grid, opt.save_images + "/epoch-{}.png".format(epoch + 1), nrow=1, normalize=False)
    if (epoch + 1) % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(embedder.state_dict(), opt.save_models + "/embedder_{}.pth".format(epoch + 1))
        torch.save(generator.state_dict(), opt.save_models + "/generator_{}.pth".format(epoch + 1))
        torch.save(discriminator.state_dict(), opt.save_models + "/discriminator_{}.pth".format(epoch + 1))
    pbar.close()

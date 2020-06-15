import argparse
import os
import numpy as np
import math
import itertools
import sys
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from dataset import *

import torch.nn as nn
import torch.nn.functional as F
import torch

class Log_Writer(object):
    def __init__(self, file_folder, file_name, computer):
        # cur_time = time.localtime(time.time())
        # file_name = "00训练输出log_" + computer + "_{:0>4d}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}"\
        #     .format(cur_time.tm_year, cur_time.tm_mon, cur_time.tm_mday,
        #             cur_time.tm_hour, cur_time.tm_min, cur_time.tm_sec)

        self.file_path = os.path.join(file_folder, file_name)

        # 新建一个空白文件 并写入当前时间
        f = open(self.file_path, mode='w')
        # f.write(str(cur_time))
        # f.write('\n')
        f.close()

    def addline(self, string_to_write):
        f = open(self.file_path, mode='a')
        f.writelines(string_to_write)
        f.write('\n')
        f.close()

    def print_and_addline(self, string_to_write):
        print(string_to_write)
        self.addline(string_to_write)

def psnr(target, ref, scale):
    # target:目标图像  ref:参考图像  scale:尺寸大小
    # assume RGB image
    # target_data = np.array(target)
    # target_data = target_data[scale:-scale, scale:-scale, :]
    #
    # ref_data = np.array(ref)
    # ref_data = ref_data[scale:-scale, scale:-scale, :]
    diff = ref - target
    # diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataloader = DataLoader(
    ImageDataset("E:\\Tsinghua\\研究生课程\\模式识别\\experiments\\exp3\\ESRGAN\\images\\training", hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
testdataloader = DataLoader(
    ImageDataset("E:\\Tsinghua\\研究生课程\\模式识别\\experiments\\exp3\\ESRGAN\\images\\test", hr_shape=hr_shape),
    batch_size=1,
    shuffle=True,
    num_workers=0,
)
# ----------
#  Training
# ----------
loss_G_writer = Log_Writer("E:\Tsinghua\研究生课程\模式识别\experiments\exp3\ESRGAN", "loss_G", "CoLab")
loss_D_writer = Log_Writer("E:\Tsinghua\研究生课程\模式识别\experiments\exp3\ESRGAN", "loss_D", "CoLab")
PSNR_writer = Log_Writer("E:\Tsinghua\研究生课程\模式识别\experiments\exp3\ESRGAN", "PSNR", "CoLab")
Mean_PSNR_writer = Log_Writer("E:\Tsinghua\研究生课程\模式识别\experiments\exp3\ESRGAN", "Mean_PSNR", "CoLab")

best_PSNR = 20;
Mean_PSNR = 0;
for epoch in range(opt.epoch, opt.n_epochs):
    generator.train()
    discriminator.train()
    for i, imgs in enumerate(dataloader):

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
            )
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
        loss_G_writer.print_and_addline("{} {} {}".format(epoch, i, loss_G))
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D_writer.print_and_addline("{} {} {}".format(epoch, i, loss_D))
        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_content.item(),
                loss_GAN.item(),
                loss_pixel.item(),
            )
        )

        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
            save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)

    generator.eval()
    Mean_PSNR_list = []
    for i, imgs in enumerate(testdataloader):
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))
        with torch.no_grad():
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)
            sr_image = denormalize(gen_hr).squeeze(0).cpu().numpy()
            hr_image = denormalize(imgs_hr).squeeze(0).cpu().numpy()
            # print(sr_image.shape)
            # print(hr_image.shape)
        #PSNR
        sr_image = sr_image.transpose(1, 2, 0)
        hr_image = hr_image.transpose(1, 2, 0)
        PSNR = psnr(sr_image, hr_image, 255)
        PSNR_writer.print_and_addline("{} {} {}".format(epoch, i, PSNR))
        Mean_PSNR_list.append(PSNR)
        # print(PSNR)
    Mean_PSNR = np.mean(np.array(Mean_PSNR_list))
    Mean_PSNR_writer.print_and_addline("{} {}".format(epoch, Mean_PSNR))

    if Mean_PSNR > best_PSNR:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator.pth")
        torch.save(discriminator.state_dict(), "saved_models/discriminator.pth")
        torch.save(optimizer_D.state_dict(), "saved_models/optimizer_D.pth")
        torch.save(optimizer_G.state_dict(), "saved_models/optimizer_G.pth")
        best_PSNR = Mean_PSNR
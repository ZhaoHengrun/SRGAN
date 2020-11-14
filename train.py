# encoding: utf-8

import argparse
import os
import sys
import time

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import visdom
import numpy as np

from models import Generator, Discriminator, FeatureExtractor
from dataset import get_training_set

start_time = time.perf_counter()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use the chosen gpu
# torch.cuda.set_device(2)  # use the chosen gpu
vis = visdom.Visdom(env='SRGAN')

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')  # default = 16
parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')  # default = 2
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--cuda', default='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='',
                    help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints/', help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass

min_total_loss = 99999999
save_flag = False

train_set = get_training_set()

dataloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)

generator = Generator(16, opt.upSampling)
# generator = nn.DataParallel(generator)
if opt.generatorWeights != '':
    generator.load_state_dict(torch.load(opt.generatorWeights))
# print(generator)

discriminator = Discriminator()
# discriminator = nn.DataParallel(discriminator)
if opt.discriminatorWeights != '':
    discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
# print(discriminator)

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
# print(feature_extractor)

content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

ones_const = torch.ones(opt.batchSize, 1)

if opt.cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()
    ones_const = ones_const.cuda()

optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)

print('Generator pre-training')
for epoch_p in range(2):
    mean_generator_content_loss = 0.0

    for i, data in enumerate(dataloader):
        low_res_real = data[0]
        high_res_real = data[1]

        if opt.cuda:
            high_res_real = high_res_real.cuda()
            high_res_fake = generator(low_res_real.cuda())
            high_res_fake = high_res_fake.cuda()
        else:
            high_res_real = high_res_real
            high_res_fake = generator(low_res_real)
        # ######## Train generator #########
        generator.zero_grad()

        generator_content_loss = content_criterion(high_res_fake, high_res_real)
        mean_generator_content_loss += generator_content_loss.item()

        generator_content_loss.backward()
        optim_generator.step()
        # ######## Status and display #########
        sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (
            epoch_p, 2, i, len(dataloader), generator_content_loss.item()))
    sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f\n' % (
        epoch_p, 2, i, len(dataloader), mean_generator_content_loss / len(dataloader)))

torch.save(generator.state_dict(), '%s/generator_pretrain.pth' % opt.out)

optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR * 0.1)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR * 0.1)

print('SRGAN training')
for epoch in range(opt.nEpochs):
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    for i, data in enumerate(dataloader):
        # Generate data
        low_res_real = data[0]
        high_res_real = data[1]

        if opt.cuda:
            high_res_real = high_res_real.cuda()
            high_res_fake = generator(low_res_real.cuda())
            high_res_fake = high_res_fake.cuda()
            target_real = torch.rand(opt.batchSize, 1) * 0.5 + 0.7

            target_real = target_real.cuda()
            target_fake = torch.rand(opt.batchSize, 1) * 0.3
            target_fake = target_fake.cuda()
        else:
            high_res_fake = generator(low_res_real)
            target_real = torch.rand(opt.batchSize, 1) * 0.5 + 0.7
            target_fake = torch.rand(opt.batchSize, 1) * 0.3
        # ######## Train discriminator #########
        discriminator.zero_grad()

        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real)
        discriminator_loss = discriminator_loss + adversarial_criterion(
            discriminator(high_res_fake.data), target_fake)

        mean_discriminator_loss += discriminator_loss.item()

        discriminator_loss.backward()
        optim_discriminator.step()
        # ######## Train generator #########
        generator.zero_grad()

        real_features = feature_extractor(high_res_real).data
        fake_features = feature_extractor(high_res_fake)
        # content loss
        generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006 * content_criterion(
            fake_features, real_features)
        mean_generator_content_loss += generator_content_loss.item()
        # adversarial loss
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
        mean_generator_adversarial_loss += generator_adversarial_loss.item()
        # total loss
        generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
        mean_generator_total_loss += generator_total_loss.item()

        generator_total_loss.backward()
        optim_generator.step()
        # ######## Status and display #########
        sys.stdout.write(
            '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (
                epoch, opt.nEpochs, i, len(dataloader),
                discriminator_loss.item(), generator_content_loss.item(), generator_adversarial_loss.item(),
                generator_total_loss.item()))

        vis.line(Y=np.array([generator_total_loss.item()]), X=np.array([i]),
                 win='generator_total_loss',
                 opts=dict(title='generator_total_loss'),
                 update='append'
                 )
        vis.line(np.array([generator_content_loss.item()]), np.array([i])
                 , win='generator_content_loss',
                 opts=dict(title='generator_content_loss'),
                 update='append')
        vis.line(np.array([generator_adversarial_loss.item()]), np.array([i])
                 , win='generator_adversarial_loss',
                 opts=dict(title='generator_adversarial_loss'),
                 update='append')

    mean_discriminator_loss = mean_discriminator_loss / len(dataloader)
    mean_generator_content_loss = mean_generator_content_loss / len(dataloader)
    mean_generator_adversarial_loss = mean_generator_adversarial_loss / len(dataloader)
    mean_generator_total_loss = mean_generator_total_loss / len(dataloader)

    sys.stdout.write(
        '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (
            epoch, opt.nEpochs, i, len(dataloader),
            mean_discriminator_loss, mean_generator_content_loss,
            mean_generator_adversarial_loss, mean_generator_total_loss))

    # if mean_generator_total_loss < min_total_loss:
    #     min_total_loss = mean_generator_total_loss
    # torch.save(generator.state_dict(),
    #            '{}g_minloss{:.8f}epoch{}.pth'.format(opt.out, mean_generator_total_loss, epoch))
    #
    # torch.save(generator.state_dict(), '{}g_epoch_{}.pth'.format(opt.out, epoch))
    # torch.save(discriminator.state_dict(), '{}d_epoch_{}.pth'.format(opt.out, epoch))

    vis.line(Y=np.array([mean_generator_total_loss]), X=np.array([epoch]),
             win='mean_generator_total_loss',
             opts=dict(title='mean_generator_total_loss'),
             update='append'
             )
    vis.line(np.array([mean_generator_content_loss]), np.array([epoch])
             , win='mean_generator_content_loss',
             opts=dict(title='mean_generator_content_loss'),
             update='append')
    vis.line(np.array([mean_generator_adversarial_loss]), np.array([epoch])
             , win='mean_generator_adversarial_loss',
             opts=dict(title='mean_generator_adversarial_loss'),
             update='append')

end_time = time.perf_counter()
print("Running time: ", (end_time - start_time) / 60)

# ssh -L 8097:127.0.0.1:8097 zhaohengrun@192.168.1.229

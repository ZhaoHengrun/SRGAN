from __future__ import print_function
import argparse
import numpy as np

from dataset import *
from models import Generator
import torch

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, default='datasets/test/test.png', help='input image to use')
parser.add_argument('--input_LR_path', type=str, default='datasets/test/set14/LR/', help='input path to use')
parser.add_argument('--input_HR_path', type=str, default='datasets/test/set14/HR/', help='input path to use')
parser.add_argument('--model', type=str, default='checkpoints/g.pth', help='model file to use')
parser.add_argument('--output_path', default='results/', type=str, help='where to save the output image')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--calc_psnr', default=True, action='store_true', help='calc psnr')
opt = parser.parse_args()

print(opt)


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


# def calc_psnr(img1, img2):
#     criterion = nn.MSELoss()
#     mse = criterion(img1, img2)
#     return 10 * log10(1 / mse.item())


loader = transforms.Compose([
    transforms.ToTensor()])
unnormalize = transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])

path = opt.input_LR_path
path_HR = opt.input_HR_path
calc_flag = opt.calc_psnr

crop_size = 256
scale = 4
image_nums = len([lists for lists in listdir(path) if is_image_file('{}/{}'.format(path, lists))])
print(image_nums)
psnr_avg = 0
psnr_avg_bicubic = 0
generator = Generator(16, 4)

for i in listdir(path):
    if is_image_file(i):
        with torch.no_grad():
            img_name = i.split('.')
            img_num = img_name[0]
            img_original = Image.open('{}{}'.format(path_HR, i))

            if calc_flag is True:
                img_original_ybr = img_original.convert('YCbCr')
                img_original_y, _, _ = img_original_ybr.split()

            img_LR = Image.open('{}{}'.format(path, i))

            if len(np.array(img_LR).shape) != 3:
                img_LR = img_LR.convert('RGB')

            # img_to_tensor = ToTensor()
            img_to_tensor = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])

            input = img_to_tensor(img_LR)
            input = torch.unsqueeze(input, dim=0).float()

            # model = torch.load(opt.model, map_location='cuda:0')
            # generator = nn.DataParallel(generator, device_ids=range(1))
            generator.load_state_dict(torch.load(opt.model, map_location='cuda:0'))

            if opt.cuda:
                generator = generator.cuda()
                input = input.cuda()

            out = generator(input)

            out = out.cpu()
            im_h = unnormalize(out.data[0])
            # im_h = out.data[0].numpy().astype(np.float32)
            im_h = im_h.numpy().astype(np.float32)
            im_h = im_h * 255.
            im_h = np.clip(im_h, 0., 255.)
            im_h = im_h.transpose(1, 2, 0)
            im_h_pil = Image.fromarray(im_h.astype(np.uint8))
            if calc_flag is True:
                im_h_pil_ybr = im_h_pil.convert('YCbCr')
                im_h_pil_y, _, _ = im_h_pil_ybr.split()

                # fig = plt.figure()
                # plt.imshow(im_h.astype(np.uint8))
                # plt.title('EDSR')
                # plt.show()

                psnr_val = calc_psnr(loader(im_h_pil_y), loader(img_original_y))
                psnr_avg += psnr_val
                print(psnr_val)

            im_h_pil.save('{}output/{}.png'.format(opt.output_path, img_num))
            img_original.save('{}gt/{}.png'.format(opt.output_path, img_num))
            print('image {} saved to {}'.format(img_num, opt.output_path))
if calc_flag is True:
    psnr_avg = psnr_avg / image_nums
    print('psnr_avg', psnr_avg)

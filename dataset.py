import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
from torchvision import transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def get_training_set():
    root_dir = 'datasets/train/'
    LR_dir = join(root_dir, "LR")
    HR_dir = join(root_dir, "HR")

    return DatasetFromFolder(LR_dir, HR_dir)


def get_test_set():
    root_dir = 'datasets/test/set14/'
    LR_dir = join(root_dir, "LR")
    HR_dir = join(root_dir, "HR")

    return DatasetFromFolder(LR_dir, HR_dir)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir_1, image_dir_2, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames_1 = [join(image_dir_1, x) for x in listdir(image_dir_1) if is_image_file(x)]
        self.image_filenames_2 = [join(image_dir_2, x) for x in listdir(image_dir_2) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.input_transform = transforms.Compose([
            transforms.CenterCrop(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            # transforms.Lambda(lambda x: x.repeat(1, 1, 1)),
        ])
        self.target_transform = transforms.Compose([
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_input = Image.open(self.image_filenames_1[index])
        img_target = Image.open(self.image_filenames_2[index])

        img_input = self.input_transform(img_input)
        img_target = self.target_transform(img_target)
        # if self.input_transform:
        #     input = self.input_transform(input)
        # if self.target_transform:
        #     target = self.target_transform(target)
        return img_input, img_target

    def __len__(self):
        return len(self.image_filenames_1)

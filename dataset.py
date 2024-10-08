import numpy as np
# import scipy.misc as reader
import imageio as reader
import os
import scipy.io as sio
from PIL import Image
from torchvision import transforms
# from config import INPUT_SIZE
INPUT_SIZE = (448, 448)


class CustomDataset():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train

        variants = []
        with open(os.path.join(root, 'data/variants.txt'), 'r', encoding='utf-16') as f:
            for line in f:
                variants.append(line.strip())

        if self.is_train:
            self.train_file_list = []
            self.train_label = []
            with open(os.path.join(root, 'data/images_variant_trainval.txt'), 'r') as f:
                for line in f:
                    line = line.strip()
                    path, variant = line.split()
                    self.train_file_list.append(
                        os.path.join(root, 'data', path))
                    self.train_label.append(variants.index(variant))
        else:
            self.test_file_list = []
            self.test_label = []
            with open(os.path.join(root, 'data/images_variant_test.txt'), 'r') as f:
                for line in f:
                    line = line.strip()
                    path, variant = line.split()
                    self.test_file_list.append(
                        os.path.join(root, 'data', path))
                    self.test_label.append(variants.index(variant))

    def __getitem__(self, index):
        if self.is_train:
            img_path = self.train_file_list[index]
            img, target = reader.imread(
                img_path)[:-20, ...], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            # img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            # img = transforms.RandomCrop(INPUT_SIZE)(img)
            # img = transforms.RandomHorizontalFlip()(img)
            img = transforms.Resize((550, 550))(img)
            img = transforms.RandomCrop(INPUT_SIZE, padding=8)(img)
            # img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        else:
            img_path = self.test_file_list[index]
            img, target = reader.imread(
                img_path)[:-20, ...], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
                # img = img[y1:y2, x1:x2, :]
            img = Image.fromarray(img, mode='RGB')
            # img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(INPUT_SIZE)(img)
            # img = transforms.ToTensor()(img)
            # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            img = transforms.Resize((550, 550))(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':
    dataset = CustomDataset(root='./')
    print(len(dataset.train_file_list))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
        break

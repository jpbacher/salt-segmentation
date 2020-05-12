from os.path import join
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


class SaltDataset(Dataset):

    def __init__(self, root_path, file_ids, is_test=False):
        self.root_path = root_path
        self.file_ids = file_ids
        self.is_test = is_test

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]

        img_folder = join(self.root_path, 'images')
        img_path = join(img_folder, file_id + '.png')

        mask_folder = join(self.root_path, 'masks')
        mask_path = join(mask_folder, file_id + '.png')

        image = get_image(img_path)

        if self.is_test:
            return (image,)
        else:
            mask = get_image(mask_path, mask=True)
            return image, mask


def get_image(path, mask=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    if h % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - h % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad
    if w % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - w % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(
        img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)
    if mask:
        img = img[:, :, 0:1] // 255
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(img)
    else:
        img = img / 255.
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(img)

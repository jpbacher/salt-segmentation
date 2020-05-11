from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class SaltDataset(Dataset):

    def __init__(self, root_path):
        self.root_path = root_path
        self.depths_df = pd.read_csv(join(root_path, 'train.csv'))
        self.file_list = list(self.depths_df['id'].values)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_id = self.file_list[idx]

        img_folder = join(self.root_path, 'images')
        img_path = join(img_folder, file_id + '.png')

        mask_folder = join(self.root_path, 'masks')
        mask_path = join(mask_folder, file_id + '.png')

        image = plt.imread(img_path)
        mask = plt.imread(mask_path)

        return image, mask

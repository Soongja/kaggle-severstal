import os
import math
import cv2
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset

from utils.rle import rle2mask


class SteelDataset_Seg(Dataset):
    def __init__(self, config, split, transform=None):
        self.config = config
        self.split = split
        self.transform = transform

        self.train_df = pd.read_csv(self.config.TRAIN_DF)
        # if self.config.PSEUDO_TRAIN_DF:
        #     pseudo_train_df = pd.read_csv(self.config.PSEUDO_TRAIN_DF)
        #     self.train_df = pd.concat([self.train_df, pseudo_train_df])

        fold_df = pd.read_csv(self.config.FOLD_DF)
        if self.config.PSEUDO_FOLD_DF:
            pseudo_fold_df = pd.read_csv(self.config.PSEUDO_FOLD_DF)
            fold_df = pd.concat([fold_df, pseudo_fold_df])
            print('[*] pseudo labels combined.')

        if self.config.TRAIN_ALL:
            if self.split == 'train':
                self.fold_df = fold_df
            elif self.split == 'val':
                self.fold_df = fold_df.loc[fold_df['split'] == self.split].reset_index(drop=True)
        else:
            self.fold_df = fold_df.loc[fold_df['split'] == self.split].reset_index(drop=True)

        if self.config.DEBUG:
            self.fold_df = self.fold_df[:40]
        print(self.split, 'set:', len(self.fold_df))

        self.labels = self.fold_df['ClassIds'].values

    def __len__(self):
        return len(self.fold_df)

    def __getitem__(self, idx):
        ImageId = self.fold_df["ImageId"][idx]
        image = cv2.imread(os.path.join(self.config.DATA_DIR, ImageId), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((256, 1600, 4), dtype=np.uint8) # 256, 1600으로 하드코딩 하고 나중에 리사이징 하는게 맞지 않을까.
        EncodedPixels = self.train_df.loc[self.train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) == ImageId]['EncodedPixels'].values

        if len(EncodedPixels) > 0:
            for i in range(4):
                if str(EncodedPixels[i]) != 'nan':
                    # mask_c = (i+1) * rle2mask(EncodedPixels[i])
                    # mask_c = cv2.resize(mask_c, (self.config.SEG.IMG_W, self.config.SEG.IMG_H))
                    # mask[0] += mask_c

                    mask[:,:,i] = rle2mask(EncodedPixels[i])

        # mask의 값은 0과 1!!!!
        mask = mask * 255 # albu 넣을 때 1로 넣어도 되는지 아직 모름

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H))
        # mask = cv2.resize(mask, (self.config.DATA.IMG_W, self.config.DATA.IMG_H), interpolation=cv2.INTER_NEAREST)

        image = (image - 128.) / 128.
        mask = mask / 255.

        image = torch.from_numpy(image).permute((2, 0, 1)).float()
        mask = torch.from_numpy(mask).permute((2, 0, 1)).float()

        return ImageId, image, mask


class SteelDataset_Cls(Dataset):
    def __init__(self, config, split, transform=None):
        self.config = config
        self.split = split
        self.transform = transform

        fold_df = pd.read_csv(self.config.FOLD_DF)
        if self.config.PSEUDO_FOLD_DF:
            pseudo_fold_df = pd.read_csv(self.config.PSEUDO_FOLD_DF)
            fold_df = pd.concat([fold_df, pseudo_fold_df])
            print('[*] pseudo labels combined.')

        if self.config.TRAIN_ALL:
            if self.split == 'train':
                self.fold_df = fold_df
            elif self.split == 'val':
                self.fold_df = fold_df.loc[fold_df['split'] == self.split].reset_index(drop=True)
        else:
            self.fold_df = fold_df.loc[fold_df['split'] == self.split].reset_index(drop=True)

        if self.config.DEBUG:
            self.fold_df = self.fold_df[:100]
        print(self.split, 'set:', len(self.fold_df))

        self.labels = self.fold_df['ClassIds'].values
        self.onehot_labels = self.convert_to_onehot(self.labels)

    def __len__(self):
        return len(self.fold_df)

    def __getitem__(self, idx):
        ImageId = self.fold_df["ImageId"][idx]
        image = cv2.imread(os.path.join(self.config.DATA_DIR, ImageId), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H))

        if self.transform is not None:
            image = self.transform(image)

        image = (image - 128.) / 128.

        # new
        label = np.array([0, 0, 0, 0])
        classids = self.fold_df['ClassIds'][idx]
        if not pd.isnull(classids):
            classids = list(str(int(classids)))  # ['1', '3']
            for classid in classids:
                label[int(classid) - 1] = 1

        # label = np.array([int(self.fold_df['hasMask'][idx])])

        image = torch.from_numpy(image).permute((2, 0, 1)).float()
        label = torch.from_numpy(label).float()

        return ImageId, image, label


    def convert_to_onehot(self, labels):
        onehot_labels = []
        for idx, classids in enumerate(labels):
            label = [0, 0, 0, 0]
            if not pd.isnull(classids):
                classids = list(str(int(classids)))  # ['1', '3']
                for classid in classids:
                    label[int(classid) - 1] = 1

            onehot_labels.append(label)
        return np.asarray(onehot_labels)

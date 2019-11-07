import pandas as pd
import numpy as np
import cv2
import os


def rle2mask(mask_rle, shape=(1600, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


os.makedirs('train_masks', exist_ok=True)

df = pd.read_csv('train.csv')

for idx, row in df.iterrows():
    rle = row['EncodedPixels']

    fname = row['ImageId_ClassId']
    fname = fname.split('.')[0] + '_' + fname.split('_')[1] + '.jpg'

    if not pd.isna(rle):
        mask = rle2mask(rle) * 255
        cv2.imwrite(os.path.join('train_masks', fname), mask)
    else:
        cv2.imwrite(os.path.join('train_masks', fname), np.zeros((256,1600), np.uint8))

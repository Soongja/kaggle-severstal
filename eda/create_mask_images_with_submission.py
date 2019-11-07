import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


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


name = '0.4'

os.makedirs(name, exist_ok=True)

df = pd.read_csv('submission_' + name + '.csv')
print(df)

for idx in tqdm(range(len(df) // 4)):
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    id = df.loc[4*idx]['ImageId_ClassId'].split('.')[0]
    EncodedPixels = df.loc[4*idx:4*(idx+1)-1]['EncodedPixels'].values
    # print(EncodedPixels)

    if len(EncodedPixels) > 0:
        for i in range(4):
            if str(EncodedPixels[i]) != 'nan':
                mask[:,:,i] = rle2mask(EncodedPixels[i])

    mask = mask * 255
    # print(mask)
    cv2.imwrite('%s/%s_1.png' % (name, id), mask[:,:,0])
    cv2.imwrite('%s/%s_2.png' % (name, id), mask[:,:,1])
    cv2.imwrite('%s/%s_3.png' % (name, id), mask[:,:,2])
    cv2.imwrite('%s/%s_4.png' % (name, id), mask[:,:,3])
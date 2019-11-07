import os
import cv2
import pandas as pd
import numpy as np
from utils.rle import rle2mask
import matplotlib.pyplot as plt
import random

#######
pred_csv = 'submissions/submission_190918a.csv'
# pred_csv = 'val_pred.csv'

img_dir = 'data/test_images'

pred_img_dir = 'visualize/submission_190918a'
# pred_img_dir = 'val_preds'
os.makedirs(pred_img_dir, exist_ok=True)
#######


df = pd.read_csv(pred_csv)

if len(os.listdir(pred_img_dir)) == 0:
    for i in range(len(df)):
        if not pd.isna(df['EncodedPixels'][i]):
            pred = rle2mask(df['EncodedPixels'][i]) * 255
            fname = df['ImageId_ClassId'][i].split('.')[0] + df['ImageId_ClassId'][i][-2:] + '.png'
            cv2.imwrite(os.path.join(pred_img_dir, fname), pred)


pred_names = os.listdir(pred_img_dir)
print(pred_names)
img_names = [f[:9]+'.jpg' for f in pred_names]
img_names = np.unique(img_names)

random.shuffle(img_names)

total_len = len(img_names)
print(total_len)

for step in range((total_len // 10) + 1):

    fig = plt.figure(figsize=(20, 100))
    columns = 2
    rows = 5
    for i in range(10):
        ax = fig.add_subplot(rows, columns, i+1)
        idx = step*10 + i

        img = cv2.imread(os.path.join(img_dir, img_names[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_name = [pred_name for pred_name in pred_names if img_names[idx].split('.')[0] in pred_name][0]
        mask = cv2.imread(os.path.join(pred_img_dir, mask_name), 0)
        mask = np.uint8(mask / 255)

        img[mask == 1, 0] = 255

        # cv2.imshow('img', img)
        # cv2.waitKey()

        ax.title.set_text(mask_name)
        plt.imshow(img)
    plt.show()

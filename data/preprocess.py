import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv('train.csv')
train_df['ImageId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
# print(train_df.shape)
# print(train_df)

mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
mask_count_df.sort_values('hasMask', ascending=False, inplace=True)
# print(mask_count_df.shape)
print(mask_count_df.head())

class_ids = []
for id in tqdm(mask_count_df['ImageId'].values):
    class_ids.append("".join((train_df.loc[(train_df['ImageId'] == id) & (train_df['hasMask'] == True), 'ClassId'].values)))
mask_count_df['ClassIds'] = class_ids
print(mask_count_df.head())

non_missing_mask_count_df = mask_count_df[mask_count_df['hasMask'] > 0].reset_index(drop=True)
# print(non_missing_mask_count_df.shape)
# print(non_missing_mask_count_df.head())

# 아직 정리 안 함
missing_nonmissing_df = mask_count_df
missing_nonmissing_df['hasMask'] = (missing_nonmissing_df['hasMask'] > 0).astype(int)
print(missing_nonmissing_df.head())


####################################################################
# 걍 첫 processing
apply = 0
if apply:
    train_df.to_csv('train_processed.csv', index=False)

####################################################################
# segmentation

apply = 1
if apply:
    # 둘 중 하나 선택
    seg_df = mask_count_df 
    # seg_df = non_missing_mask_count_df

    # kfold for non_missing_mask_count_df
    x = seg_df['ImageId'].values
    y = seg_df['ClassIds'].values
    # print(y)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
    skf.get_n_splits(x, y)

    seg_df['split'] = 'split'
    for fold, (train_index, val_index) in enumerate(skf.split(x, y)):
        if fold == 9:
            print(fold, len(train_index), len(val_index))
            seg_df['split'].iloc[train_index] = 'train'
            seg_df['split'].iloc[val_index] = 'val'
    # print(non_missing_mask_count_df)

    seg_df.to_csv('folds/seg_10fold_9.csv', index=False)

####################################################################
# classification(정리 안 함)

apply = 0
if apply:
    x = missing_nonmissing_df['ImageId'].values
    y = missing_nonmissing_df['hasMask'].values

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
    skf.get_n_splits(x, y)
    missing_nonmissing_df['split'] = 'split'

    for fold, (train_index, val_index) in enumerate(skf.split(x, y)):
        if fold == 0:
            print(fold, len(train_index), len(val_index))
            missing_nonmissing_df['split'].iloc[train_index] = 'train'
            missing_nonmissing_df['split'].iloc[val_index] = 'val'
    # print(missing_nonmissing_df)

    missing_nonmissing_df.to_csv('classification_10fold_0.csv', index=False)

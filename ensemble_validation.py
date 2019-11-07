import os
import random
import cv2
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet34

# final 때 바꿔야 할 애들
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
from segmentation_models_pytorch import Unet, Linknet, FPN, PSPNet
from albumentations import Compose, Rotate, RandomSizedCrop, RandomBrightnessContrast


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

########################################################################################################################


def get_model(config):
    if config.TASK == 'seg':
        model_architecture = config.ARCHITECTURE
        model_encoder = config.ENCODER

        # activation은 eval 모드일 때 적용해 주는 거라 train 때에는 직접 sigmoid 쳐야한다.
        model = globals().get(model_architecture)(model_encoder, classes=4, activation='sigmoid')

        print('architecture:', model_architecture, 'encoder:', model_encoder)

    elif config.TASK == 'cls':
        model_name = config.MODEL
        f = globals().get(model_name)

        if model_name.startswith('resnet'):
            model = f(pretrained=False)
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 1)

        elif model_name.startswith('efficient'):
            model = EfficientNet.from_name(model_name, override_params={'num_classes': 1})

            if config.ADD_FC:
                in_features = model._fc.in_features
                new_fc = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.BatchNorm1d(256, eps=0.001, momentum=0.010000000000000009, affine=True,
                                   track_running_stats=True),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 1))

                model._fc = new_fc
                print('new fc added')

        else:
            model = f(num_classes=1000, pretrained=None)
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = model.last_linear.in_features
            model.last_linear = nn.Linear(in_features, 1)

        print('model name:', model_name)

    return model


########################################################################################################################

class SteelDataset(Dataset):
    def __init__(self, config, fnames=None, transform=None):
        self.config = config
        self.transform = transform

        sample_submission = pd.read_csv(self.config.SAMPLE_SUBMISSION)
        self.sample_submission = sample_submission.loc[sample_submission['split'] == 'val'].reset_index(drop=True)

        # self.ImageIds = np.unique(self.sample_submission['ImageId_ClassId'].apply(lambda x: x.split('_')[0]).values)
        self.ImageIds = self.sample_submission['ImageId'].values

        if fnames is not None:
            self.ImageIds = fnames

        print('len dataset: %s' % len(self.ImageIds))

    def __len__(self):
        return len(self.ImageIds)

    def __getitem__(self, idx):
        ImageId = self.ImageIds[idx]
        image = cv2.imread(os.path.join(self.config.DATA_DIR, ImageId), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        # image = cv2.resize(image, (self.config.DATA.IMG_W, self.config.DATA.IMG_H))

        image = (image - 128.) / 128.
        image = torch.from_numpy(image).permute((2, 0, 1)).float()

        return image


def get_dataloader(config, fnames=None, transform=None):
    dataset = SteelDataset(config, fnames, transform)

    dataloader = DataLoader(dataset,
                             shuffle=False,
                             batch_size=config.BATCH_SIZE,
                             num_workers=config.NUM_WORKERS,
                             pin_memory=True)

    return dataloader


########################################################################################################################


class HFlip:
    def __call__(self, image):
        return image[:,::-1]


class VFlip:
    def __call__(self, image):
        return image[::-1]


########################################################################################################################


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


########################################################################################################################


def inference(model, dataloader):
    model.eval()

    output = []
    with torch.no_grad():
        start = time.time()
        for i, images in enumerate(dataloader):
            images = images.cuda()
            logits = model(images)
            logits = F.sigmoid(logits)

            preds = logits.detach().cpu().numpy()

            # for idx in range(len(ids)):
            #     output[ids[idx]] = preds[idx]
            output.append(preds)

            del images, logits, preds
            torch.cuda.empty_cache()

            end = time.time()
            if i % 10 == 0:
                print('[%2d/%2d] time: %.2f' % (i, len(dataloader), end - start))

    output = np.concatenate(output, axis=0)

    print('inference finished. shape:', output.shape)
    return output


def run(config, fnames=None):
    model = get_model(config).cuda()
    checkpoint = torch.load(config.CHECKPOINT)

    state_dict_old = checkpoint['state_dict']
    state_dict = OrderedDict()
    # delete 'module.' because it is saved from DataParallel module
    for key in state_dict_old.keys():
        if key.startswith('module.'):
            state_dict[key[7:]] = state_dict_old[key]
        else:
            state_dict[key] = state_dict_old[key]
    model.load_state_dict(state_dict)

    ####################################################################################################
    test_loader = get_dataloader(config, fnames=fnames, transform=None)
    out = inference(model, test_loader)  # cls: (1801, 1) / seg: (N, 4, 256, 1600)

    # TTA
    ####################################################################################################
    print('----- HFlip TTA -----')
    test_loader = get_dataloader(config, fnames = fnames, transform=HFlip())
    out_hflip = inference(model, test_loader)
    ####################################################################################################
    print('----- VFlip TTA -----')
    test_loader = get_dataloader(config, fnames = fnames, transform=VFlip())
    out_vflip = inference(model, test_loader)
    ####################################################################################################
    print('----- HFlip + VFlip TTA -----')
    test_loader = get_dataloader(config, fnames = fnames, transform=transforms.Compose([HFlip(),
                                                                                       VFlip()]))
    out_flip = inference(model, test_loader)
    ####################################################################################################

    if config.TASK == 'seg':
        out_hflip = out_hflip[:,:,:,::-1]
        out_vflip = out_vflip[:,:,::-1]
        out_flip = out_flip[:,:,::-1,::-1]

    out = (out + out_hflip + out_vflip + out_flip) / 4.0

    out = np.squeeze(out)
    return out


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def post_process(probability, threshold, min_size, fill_up=False):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.uint8)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1

    # experiments
    if fill_up:
        contours, _ = cv2.findContours(predictions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_up_predictions = np.zeros((256, 1600), np.uint8)
        for c in contours:
            cv2.drawContours(filled_up_predictions, [c], 0, 1, -1)
        predictions = filled_up_predictions

    # dilation
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # RECT, ELLIPSE, CROSS
    # predictions = cv2.dilate(predictions, kernel, iterations=1)

    return predictions


def dice_coef(preds, labels):
    preds = preds >= 0.5

    smooth = 1e-6
    intersection = (preds.float() * labels.float()).sum(dim=(2, 3))
    union = preds.float().sum(dim=(2, 3)) + labels.float().sum(dim=(2, 3))
    # class 별로 찍게 하자 [N, C]
    dice = ((2. * intersection + smooth) / (union + smooth)).mean(dim=0)

    return dice


def accuracy(preds, labels):
    """Computes the accuracy for multiple binary predictions"""
    pred = preds >= 0.5
    truth = labels >= 0.5
    acc = pred.eq(truth).sum().float() / float(labels.numel())

    return acc


def main():
    import warnings
    warnings.filterwarnings("ignore")
    seed_everything()

    ####################


    cls = 1
    seg = 0


    ####################


    submission = pd.read_csv('data/folds/seg_10fold_0.csv', engine='python')
    submission = submission.loc[submission['split'] == 'val'].reset_index(drop=True)
    ImageIds = submission['ImageId'].values
    labels = submission['hasMask'].values
    labels = torch.tensor(labels, dtype=torch.float32, device='cpu')

    if seg:
        train_df = pd.read_csv('data/train_processed.csv')
        masks = np.zeros((len(ImageIds), 256, 1600, 4), dtype=np.uint8)

        for i in range(len(submission)):
            EncodedPixels = train_df.loc[train_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) == ImageIds[i]]['EncodedPixels'].values

            for j in range(4):
                if str(EncodedPixels[j]) != 'nan':
                    masks[i, :, :, j] = rle2mask(EncodedPixels[j])

        masks = torch.tensor(masks, device='cpu')


    ############################################
    # 여기랑 밑에 checkpoint만 딱 건드리자
    threshold = 0.5
    min_size = 500
    fill_up = False

    ###################### cls #########################
    if cls:
        config_cls_0 = Config(task='cls', model='efficientnet-b3',
                            checkpoint='_results/CLS_eff-b3_bce/checkpoints/epoch_0024_score0.9786_loss0.0996.pth')
        cls_0 = run(config_cls_0)

        # config_cls_1 = Config(task='cls', model='efficientnet-b4',
        #                       checkpoint='_results/CLS_eff-b4_bce/checkpoints/epoch_0029_score0.9770_loss0.1333.pth')
        # cls_1 = run(config_cls_1)

        config_cls_2 = Config(task='cls', model='efficientnet-b2',
                              checkpoint='_results/CLS_eff-b2_bce/checkpoints/epoch_0026_score0.9810_loss0.1004.pth')
        cls_2 = run(config_cls_2)


        cls_output = (cls_0 + cls_2) / 2.0
        # cls_output = (cls_0 + cls_1 + cls_2) / 3.0
        print(cls_output)

        cls_output = torch.tensor(cls_output, dtype=torch.float32, device='cpu')

        print('ensemble CLS validation score: %.4f' % accuracy(cls_output, labels))


    ###################### seg #########################
    if seg:
        config_0 = Config(task='seg', architecture='Unet', encoder='se_resnext50_32x4d',
                          checkpoint='_results/Unet_se_resnext50_shift/checkpoints/epoch_0049_score0.9452_loss0.1433.pth')
        fold_0 = run(config_0)

        config_1 = Config(task='seg', architecture='Unet', encoder='efficientnet-b3',
                          checkpoint='_results/Unet_eff-b3_shift/checkpoints/epoch_0049_score0.9471_loss0.1393.pth')
        fold_1 = run(config_1)

        final = (fold_0 + fold_1) / 2.0  # (N, 4, 256, 1600)

        final = torch.tensor(final, device='cpu')

        print('ensemble SEG validation score: %.4f' % dice_coef(final, masks))


class Config():
    def __init__(self, task, model=None, add_fc=None, architecture=None, encoder=None, checkpoint=None):
        self.TASK = task
        self.MODEL = model
        self.ADD_FC = add_fc
        self.ARCHITECTURE = architecture
        self.ENCODER = encoder
        self.CHECKPOINT = checkpoint

        self.IMG_H = 256
        self.IMG_W = 1600

        self.DATA_DIR = 'data/train_images'
        self.SAMPLE_SUBMISSION = 'data/folds/seg_10fold_0.csv'

        self.DEBUG = False

        self.BATCH_SIZE = 32
        self.NUM_WORKERS = 4


if __name__ == '__main__':
    start = time.time()
    main()
    ellapsed = time.time() - start
    print('Total inference time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))

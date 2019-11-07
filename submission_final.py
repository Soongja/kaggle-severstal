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

os.chdir('/kaggle/input')
from efficientnet.model import EfficientNet
from efficientnetgem.model import EfficientNet as EfficientNetGem
from segmentation4.unet.model import Unet
os.chdir('/kaggle/working/')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

########################################################################################################################


def get_model(config):
    if config.TASK == 'seg':
        model_architecture = config.ARCHITECTURE
        model_encoder = config.ENCODER

        # activation은 eval 모드일 때 적용해 주는 거라 train 때에는 직접 sigmoid 쳐야한다.
        model = globals().get(model_architecture)(model_encoder, encoder_weights=None, classes=4, activation='sigmoid')
        print('architecture:', model_architecture, 'encoder:', model_encoder)

    elif config.TASK == 'cls':
        model_name = config.MODEL

        if config.GEM:
            model = EfficientNetGem.from_name(model_name, override_params={'num_classes': 4})
        else:
            model = EfficientNet.from_name(model_name, override_params={'num_classes': 4})

        print('model name:', model_name)

    return model


########################################################################################################################

class SteelDataset(Dataset):
    def __init__(self, config, fnames=None, transform=None):
        self.config = config
        self.transform = transform

        self.sample_submission = pd.read_csv(self.config.SAMPLE_SUBMISSION)
        self.ImageIds = np.unique(self.sample_submission['ImageId_ClassId'].apply(lambda x: x.split('_')[0]).values)

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


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


########################################################################################################################


def inference(model, dataloader):
    model.eval()

    output = []
    with torch.no_grad():
        # start = time.time()
        for i, images in enumerate(dataloader):
            images = images.cuda()
            logits = model(images)
            logits = F.sigmoid(logits)

            preds = logits.detach()

            output.append(preds)

            del images, logits, preds
            torch.cuda.empty_cache()

            # end = time.time()
            # if i % 50 == 0:
            #     print('[%2d/%2d] time: %.2f' % (i, len(dataloader), end - start))

    output = torch.cat(output, dim=0)
    output = output.cpu().numpy()
    # print('inference finished. shape:', output.shape)
    return output


def run(config, fnames=None):
    # run_start = time.time()

    model = get_model(config).cuda()
    checkpoint = torch.load(config.CHECKPOINT)

    # state_dict = checkpoint['state_dict']

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
    print('----- VFlip TTA -----')
    test_loader = get_dataloader(config, fnames=fnames, transform=VFlip())
    out_vflip = inference(model, test_loader)
    if config.TASK == 'seg':
        out_vflip = np.flip(out_vflip, axis=2)
    out += out_vflip
    del out_vflip
    ####################################################################################################
    print('----- HFlip TTA -----')
    test_loader = get_dataloader(config, fnames=fnames, transform=HFlip())
    out_hflip = inference(model, test_loader)
    if config.TASK == 'seg':
        out_hflip = np.flip(out_hflip, axis=3)
    out += out_hflip
    del out_hflip
    ####################################################################################################
    # print('----- VFlip + HFlip TTA -----')
    # test_loader = get_dataloader(config, fnames=fnames, transform=transforms.Compose([VFlip(),
    #                                                                                   HFlip()]))
    # out_vhflip = inference(model, test_loader)
    # if config.TASK == 'seg':
    #     out_vhflip = np.flip(out_vhflip, axis=(2,3))
    # out += out_vhflip
    # del out_vhflip
    ####################################################################################################

    out = out / 3.0

    # print('!!!!!single model + tta time: %.2f' % (time.time() - run_start))
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


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.uint8)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1

    return predictions


def main():
    main_start = time.time()
    import warnings
    warnings.filterwarnings("ignore")
    seed_everything()

    submission = pd.read_csv('/kaggle/input/severstal-steel-defect-detection/sample_submission.csv', engine='python')
    submission['EncodedPixels'] = ''
    ImageIds = np.unique(submission['ImageId_ClassId'].apply(lambda x: x.split('_')[0]).values)

    ############################################

    cls_thresh = [0.6, 0.5, 0.5, 0.5]
    threshold = [0.4, 0.4, 0.4, 0.4]
    min_size = 200

    ###################### cls #########################

    config_cls_5 = Config(task='cls', model='efficientnet-b5',
                        checkpoint='/kaggle/input/effb5-fold0-19/epoch_0019_score0.9830_loss0.0729.pth')
    cls = run(config_cls_5)
    print('[*] ellapsed: %d minutes %d seconds' % ((time.time() - main_start) // 60, (time.time() - main_start) % 60))
    print()

    config_cls_1 = Config(task='cls', model='efficientnet-b1',
                          checkpoint='/kaggle/input/effb1-fold0-29/epoch_0029_score0.9925_loss0.0225.pth')
    cls_1 = run(config_cls_1)
    cls += cls_1
    del cls_1
    print('[*] ellapsed: %d minutes %d seconds' % ((time.time() - main_start) // 60, (time.time() - main_start) % 60))
    print()

    #########
    # pseudo
    #########

    config_cls_pseudo_0 = Config(task='cls', model='efficientnet-b0', gem=True,
                          checkpoint='/kaggle/input/sejunb0/epoch_0021_score0.9845_loss0.0868.pth')
    cls_pseudo_0 = run(config_cls_pseudo_0)
    cls += cls_pseudo_0
    del cls_pseudo_0
    print('[*] ellapsed: %d minutes %d seconds' % ((time.time() - main_start) // 60, (time.time() - main_start) % 60))
    print()

    config_cls_pseudo_1 = Config(task='cls', model='efficientnet-b1',
                          checkpoint='/kaggle/input/effb1-fold0-19-pseudo-191022b/epoch_0019_score0.9831_loss0.0702.pth')
    cls_pseudo_1 = run(config_cls_pseudo_1)
    cls += cls_pseudo_1
    del cls_pseudo_1
    print('[*] ellapsed: %d minutes %d seconds' % ((time.time() - main_start) // 60, (time.time() - main_start) % 60))
    print()

    # config_cls_pseudo_2 = Config(task='cls', model='efficientnet-b1',
    #                              checkpoint='/kaggle/input/effb1-fold0-25-pseudo-091763/epoch_0025_score0.9822_loss0.1032.pth')
    # cls_pseudo_2 = run(config_cls_pseudo_2)
    # cls += cls_pseudo_2
    # del cls_pseudo_2
    # print('[*] ellapsed: %d minutes %d seconds' % ((time.time() - main_start) // 60, (time.time() - main_start) % 60))
    # print()

    config_cls_pseudo_3 = Config(task='cls', model='efficientnet-b0',
                                 checkpoint='/kaggle/input/effb0-fold5-29-pseudo-091763/epoch_0029_score0.9875_loss0.0714.pth')
    cls_pseudo_3 = run(config_cls_pseudo_3)
    cls += cls_pseudo_3
    del cls_pseudo_3
    print('[*] ellapsed: %d minutes %d seconds' % ((time.time() - main_start) // 60, (time.time() - main_start) % 60))
    print()

    cls = cls / 5.0

    for i in range(4):
        cls[:,i] = (cls[:,i] >= cls_thresh[i]).astype(np.uint8)

    nonmissing_idxs = [i for i in range(cls.shape[0]) if np.sum(cls[i]) > 0]
    nonmissing_cls_output = cls[nonmissing_idxs]
    nonmissing_fnames = [ImageIds[i] for i in range(cls.shape[0]) if np.sum(cls[i]) > 0]
    print('!!!!! len(nonmissing_fnames):', len(nonmissing_fnames))
    del cls


    split_size = len(nonmissing_fnames) // 10
    loop_count = 1
    for i in range(0, len(nonmissing_fnames), split_size):
        print()
        print('[Loop count: %d]' % loop_count)
        ###################### seg #########################

        config_5 = Config(task='seg', architecture='Unet', encoder='efficientnet-b5',
                          checkpoint='/kaggle/input/unet-effb5-fold0-86/epoch_0086_score0.9476_loss0.1389.pth')
        fold = run(config_5, nonmissing_fnames[i:i+split_size])
        print('[*] ellapsed: %d minutes %d seconds' % ((time.time() - main_start) // 60, (time.time() - main_start) % 60))
        print()

        config_1 = Config(task='seg', architecture='Unet', encoder='efficientnet-b1',
                          checkpoint='/kaggle/input/unet-effb1-fold0-78/epoch_0078_score0.9481_loss0.1404.pth')
        fold_1 = run(config_1, nonmissing_fnames[i:i+split_size])
        fold += fold_1
        del fold_1
        print('[*] ellapsed: %d minutes %d seconds' % ((time.time() - main_start) // 60, (time.time() - main_start) % 60))
        print()

        config_1_ = Config(task='seg', architecture='Unet', encoder='efficientnet-b1',
                          # checkpoint='/kaggle/input/unet-effb1-fold7-76/epoch_0076_score0.9533_loss0.1270.pth')
                          checkpoint='/kaggle/input/unet-effb1-fold7-84/epoch_0084_score0.9540_loss0.1258.pth')
        fold_1_ = run(config_1_, nonmissing_fnames[i:i + split_size])
        fold += fold_1_
        del fold_1_
        print('[*] ellapsed: %d minutes %d seconds' % ((time.time() - main_start) // 60, (time.time() - main_start) % 60))
        print()


        fold = fold / 3.0

        ####################################################

        for idx in range(fold.shape[0]):

            preds = []
            for ch in range(4):
                if nonmissing_cls_output[i:i+split_size][idx][ch] == 0:
                    preds.append('')
                else:
                    preds.append(mask2rle(post_process(fold[idx][ch], threshold=threshold[ch], min_size=min_size)))

            submission.loc[submission['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) == nonmissing_fnames[i:i+split_size][idx], 'EncodedPixels'] = preds

            del preds

        del fold
        loop_count += 1

    submission.to_csv('submission.csv', index=False)
    print('success!')


class Config():
    def __init__(self, task, model=None, gem=False, architecture=None, encoder=None, checkpoint=None):
        self.TASK = task
        self.MODEL = model
        self.GEM = gem
        self.ARCHITECTURE = architecture
        self.ENCODER = encoder
        self.CHECKPOINT = checkpoint

        self.IMG_H = 256
        self.IMG_W = 1600

        self.DATA_DIR = '/kaggle/input/severstal-steel-defect-detection/test_images'
        self.SAMPLE_SUBMISSION = '/kaggle/input/severstal-steel-defect-detection/sample_submission.csv'

        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 4


if __name__ == '__main__':
    start = time.time()
    main()
    ellapsed = time.time() - start
    print('Total inference time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))

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
from efficientnet_gem.model import EfficientNet as EfficientNetGem
from segmentation_models_pytorch import Unet, FPN


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
            model.fc = nn.Linear(in_features, 4)

        elif model_name.startswith('efficient'):
            if config.GEM:
                model = EfficientNetGem.from_name(model_name, override_params={'num_classes': 4})
            else:
                model = EfficientNet.from_name(model_name, override_params={'num_classes': 4})
            # model = EfficientNet.from_name(model_name, override_params={'num_classes': 4})

            if config.ADD_FC:
                in_features = model._fc.in_features
                new_fc = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.BatchNorm1d(256, eps=0.001, momentum=0.010000000000000009, affine=True,
                                   track_running_stats=True),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 4))

                model._fc = new_fc
                print('new fc added')

        else:
            model = f(num_classes=1000, pretrained=None)
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = model.last_linear.in_features
            model.last_linear = nn.Linear(in_features, 4)

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

        if self.config.DEBUG:
            self.ImageIds = self.ImageIds[:40]

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
        start = time.time()
        for i, images in enumerate(dataloader):
            images = images.cuda()
            logits = model(images)
            logits = F.sigmoid(logits)

            # preds = logits.detach().cpu().numpy()
            preds = logits.detach()

            # for idx in range(len(ids)):
            #     output[ids[idx]] = preds[idx]
            output.append(preds)

            del images, logits, preds
            torch.cuda.empty_cache()

            end = time.time()
            if i % 1 == 0:
                print('[%2d/%2d] time: %.2f' % (i, len(dataloader), end - start))

    # output = np.concatenate(output, axis=0)
    # output = np.squeeze(output)
    output = torch.cat(output, dim=0)
    output = output.cpu().numpy()
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
    test_loader = get_dataloader(config, fnames=fnames, transform=HFlip())
    out_hflip = inference(model, test_loader)
    if config.TASK == 'seg':
        out_hflip = np.flip(out_hflip, axis=3)
    out += out_hflip
    del out_hflip
    ####################################################################################################
    print('----- VFlip TTA -----')
    test_loader = get_dataloader(config, fnames=fnames, transform=VFlip())
    out_vflip = inference(model, test_loader)
    if config.TASK == 'seg':
        out_vflip = np.flip(out_vflip, axis=2)
    out += out_vflip
    del out_vflip
    ####################################################################################################
    # print('----- HFlip + VFlip TTA -----')
    # test_loader = get_dataloader(config, fnames=fnames, transform=transforms.Compose([HFlip(),
    #                                                                                    VFlip()]))
    # out_hvflip = inference(model, test_loader)
    # if config.TASK == 'seg':
    #     out_hvflip = np.flip(out_hvflip, axis=(2,3))
    #     out_hvflip = np.flip(out_hvflip, axis=2)
    # out += out_hvflip
    # del out_hvflip
    ####################################################################################################

    # if config.TASK == 'seg':
    #     out_hflip = out_hflip[:,:,:,::-1]
    #     out_vflip = out_vflip[:,:,::-1]
    #     out_flip = out_flip[:,:,::-1,::-1]

    # out = (out + out_hflip + out_vflip + out_flip) / 4.0
    out = out / 3.0

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


def sharpen(p, t=0.5):
    if t != 0:
        return p ** t
    else:
        return p


def main():
    import warnings
    warnings.filterwarnings("ignore")
    seed_everything()

    submission = pd.read_csv('data/sample_submission.csv', engine='python')
    ImageIds = np.unique(submission['ImageId_ClassId'].apply(lambda x: x.split('_')[0]).values)

    ############################################
    # 여기랑 밑에 checkpoint만 딱 건드리자

    cls_thresh = [0.5,0.5,0.5,0.5]

    threshold = [0.5,0.5,0.5,0.5]
    min_size = 200
    fill_up = False

    postfix = '191024e'

    ###################### cls #########################
    '''
    config_cls_0 = Config(task='cls', model='efficientnet-b3',
                        checkpoint='_results/CLS_eff-b3_bce/checkpoints/epoch_0024_score0.9786_loss0.0996.pth')
    cls_0 = run(config_cls_0)

    config_cls_1 = Config(task='cls', model='efficientnet-b2',
                          checkpoint='_results/CLS_eff-b2_bce/checkpoints/epoch_0026_score0.9810_loss0.1004.pth')
    cls_1 = run(config_cls_1)

    config_cls_2 = Config(task='cls', model='efficientnet-b4',
                          checkpoint='_results/CLS_eff-b4_bce/checkpoints/epoch_0029_score0.9770_loss0.1333.pth')
    cls_2 = run(config_cls_2)

    config_cls_3 = Config(task='cls', model='efficientnet-b1',
                          checkpoint='_results/CLS_eff-b1_bce/checkpoints/epoch_0027_score0.9762_loss0.1420.pth')
    cls_3 = run(config_cls_3)
    '''

    config_cls_5 = Config(task='cls', model='efficientnet-b5',
                          checkpoint='_results/CLS_eff-b5_new4/checkpoints/epoch_0019_score0.9830_loss0.0729.pth')
                          # checkpoint='_results/CLS_eff-b5_train_all/checkpoints/epoch_0024_score0.9897_loss0.0352.pth')
    cls_5 = run(config_cls_5)


    # config_cls_4 = Config(task='cls', model='efficientnet-b4',
    #                       checkpoint='_results/CLS_eff-b4_new4/checkpoints/epoch_0017_score0.9839_loss0.0677.pth')
                          # checkpoint='_results/CLS_eff-b4_train_all/checkpoints/epoch_0022_score0.9915_loss0.0264.pth')
                          # checkpoint='_results/CLS_eff-b4_fold1/checkpoints/epoch_0012_score0.9851_loss0.0524.pth')
    # cls_4 = run(config_cls_4)

    # config_cls_2 = Config(task='cls', model='efficientnet-b2',
                          # checkpoint='_results/CLS_eff-b2_new4/checkpoints/epoch_0026_score0.9839_loss0.0848.pth')
                          # checkpoint='_results/CLS_eff-b2_train_all/checkpoints/epoch_0031_score0.9925_loss0.0210.pth')
                          # checkpoint='_results/CLS_eff-b2_fold2/checkpoints/epoch_0018_score0.9889_loss0.0437.pth')
    # cls_2 = run(config_cls_2)

    config_cls_1 = Config(task='cls', model='efficientnet-b1',
                          # checkpoint='_results/CLS_eff-b1_new4/checkpoints/epoch_0024_score0.9839_loss0.0798.pth')
                          checkpoint='_results/CLS_eff-b1_train_all/checkpoints/epoch_0029_score0.9925_loss0.0225.pth')
                          # checkpoint='_results/CLS_eff-b1_fold3/checkpoints/epoch_0025_score0.9847_loss0.0624.pth')
    cls_1 = run(config_cls_1)

    # config_cls_1_ = Config(task='cls', model='efficientnet-b0', gem=True,
    #                       checkpoint='_results/sejun_pseudo_eff-b0_fold0/epoch_0021_score0.9845_loss0.0868.pth')
    # cls_1_ = run(config_cls_1_)

    config_cls_1_ = Config(task='cls', model='efficientnet-b1',
                           checkpoint='_results/CLS_eff-b1_fold0_pseudo_191022b/checkpoints/epoch_0019_score0.9831_loss0.0702.pth')
    cls_1_ = run(config_cls_1_)

    cls_output = (cls_5 + cls_1 + cls_1_) / 3.0
    # cls_output = (cls_5 + cls_1) / 2.0
    np.save('cls_output_' + postfix + '.npy', cls_output)
    # cls_output = (cls_0 + cls_1) / 2.0

    # nonmissing_fnames = [ImageIds[i] for i in range(cls_output.shape[0]) if cls_output[i] >= cls_thresh]
    for i in range(4):
        cls_output[:,i] = (cls_output[:,i] >= cls_thresh[i]).astype(np.uint8)

    nonmissing_idxs = [i for i in range(cls_output.shape[0]) if np.sum(cls_output[i]) > 0]
    nonmissing_cls_output = cls_output[nonmissing_idxs]
    np.save('nonmissing_cls_output_' + postfix + '.npy', nonmissing_cls_output)
    nonmissing_fnames = [ImageIds[i] for i in range(cls_output.shape[0]) if np.sum(cls_output[i]) > 0]
    # nonmissing_fnames = ImageIds[nonmissing_idxs]

    print(len(nonmissing_fnames))
    print(nonmissing_fnames)
    with open('nonmissing_fnames_' + postfix + '.txt', 'w') as f:
        f.write(str(nonmissing_fnames))
    del cls_5, cls_1


    # nonmissing_fnames = ['006f39c41.jpg', '00bbcd9af.jpg', '0109b68ec.jpg', '010ec96b4.jpg', '01b47d973.jpg', '01d49cd47.jpg', '0280c72a9.jpg', '037e7564c.jpg', '0384b28ff.jpg', '04922a23f.jpg', '04e78fd86.jpg', '059750041.jpg', '05cb27a3f.jpg', '06708f73c.jpg', '0693ae6fd.jpg', '0783111db.jpg', '0793dde4e.jpg', '080a44c75.jpg', '08696122b.jpg', '096e7338a.jpg', '096fa2f7c.jpg', '09a71e9c9.jpg', '09b7a2ac8.jpg', '0a2c9f2e5.jpg', '0a2dbbb6f.jpg', '0a3962685.jpg', '0a63f7765.jpg', '0a9cbb927.jpg', '0ad9a817d.jpg', '0b2f0a191.jpg', '0b2fcc8a7.jpg', '0b67c64ef.jpg', '0bd004c2d.jpg', '0bdfd29ba.jpg', '0c124b96b.jpg', '0c8d438af.jpg', '0caaaab31.jpg', '0cca51b4a.jpg', '0ccde8827.jpg', '0d0491e30.jpg', '0da22916c.jpg', '0e2adcd02.jpg', '0e39b3fcc.jpg', '0ef839b31.jpg', '0f3308225.jpg', '0f3a8491e.jpg', '0fa2f1fa2.jpg', '0fbc1597c.jpg', '100608ebe.jpg', '1014ae8b8.jpg', '10d64f676.jpg', '10e3e1ae2.jpg', '110caee87.jpg', '1134a8e6b.jpg', '115a301e7.jpg', '1160c4155.jpg', '1179d7767.jpg', '1197190fe.jpg', '1200683b1.jpg', '12b38e51c.jpg', '12e67ba9a.jpg', '133fe40b6.jpg', '1374fecc2.jpg', '13b049f6e.jpg', '1445d5ab6.jpg', '147a1b7ac.jpg', '159bb4951.jpg', '15c5d6e68.jpg', '15e92a657.jpg', '15f36c028.jpg', '16070626a.jpg', '162959249.jpg', '165be877c.jpg', '16b4b381f.jpg', '16e5cabec.jpg', '179a5ea0b.jpg', '17f6716dd.jpg', '1800f04df.jpg', '183a346ed.jpg', '1859bb7f6.jpg', '19439013c.jpg', '195bf8a80.jpg', '19c0e7a52.jpg', '1a4836303.jpg', '1a8003e79.jpg', '1a9b84f4a.jpg', '1af7ac008.jpg', '1b246801d.jpg', '1b2f07f66.jpg', '1b5984acb.jpg', '1ba63ec01.jpg', '1bbf4b4c0.jpg', '1bdc31ef8.jpg', '1cb3a9f21.jpg', '1cdc0358f.jpg', '1d52eb20c.jpg', '1d617527f.jpg', '1ddca477a.jpg', '1e2a28cae.jpg', '1e446e941.jpg', '1e8f59c7e.jpg', '1ed7c09f2.jpg', '1edcc97b3.jpg', '1ef19628e.jpg', '1f6d91f6c.jpg', '1f9011c8f.jpg', '1fee22ffe.jpg', '205522c82.jpg', '2080d4f8d.jpg', '211af4a0f.jpg', '21585a8d1.jpg', '216f0c026.jpg', '21875726a.jpg', '21e28f2e5.jpg', '2233caaaf.jpg', '2235e045d.jpg', '225028fc0.jpg', '2262f1399.jpg', '228137fed.jpg', '23365fa94.jpg', '2338c3671.jpg', '236244295.jpg', '2363573c7.jpg', '23949cda8.jpg', '23e0d228d.jpg', '23e3f9870.jpg', '243bdc490.jpg', '24810ee2c.jpg', '2511baed5.jpg', '2523fef1b.jpg', '256aeb46a.jpg', '25a1d91d4.jpg', '26306db43.jpg', '269431f85.jpg', '26a2f9809.jpg', '26d4c1c22.jpg', '272e97e63.jpg', '273d5bd8b.jpg', '274004b27.jpg', '278b20cb1.jpg', '280e93616.jpg', '28138ebf2.jpg', '2819eaa18.jpg', '288d92035.jpg', '28ae995da.jpg', '2a17f8fed.jpg', '2a5cc5f58.jpg', '2a830069f.jpg', '2a9352386.jpg', '2b4ea2115.jpg', '2b75eb7ae.jpg', '2beb4de91.jpg', '2c0eea9dc.jpg', '2c29f8e2d.jpg', '2c6df432d.jpg', '2ce89fa13.jpg', '2d2474be8.jpg', '2d8fa3ef6.jpg', '2d905add8.jpg', '2dba99f31.jpg', '2e1014fb7.jpg', '2eb1e6465.jpg', '2eb811b0c.jpg', '2f502ea2a.jpg', '2f97adb98.jpg', '2fed1cf68.jpg', '3034c8ebf.jpg', '304b3bdb8.jpg', '30a55717f.jpg', '30b549e48.jpg', '311a6b255.jpg', '312357e32.jpg', '31402958a.jpg', '318ce3987.jpg', '31f8b702a.jpg', '3228d8997.jpg', '328e3c209.jpg', '32c27e65f.jpg', '334b5fd37.jpg', '335d78292.jpg', '33605817c.jpg', '338b4c181.jpg', '33d296fef.jpg', '33fd0c742.jpg', '349fee110.jpg', '351c55d38.jpg', '352f722e4.jpg', '3550242ec.jpg', '35544c8c1.jpg', '356cd7089.jpg', '358c4e6ce.jpg', '35f50a445.jpg', '363f4106f.jpg', '3661ccb2d.jpg', '369947a71.jpg', '369fd8e23.jpg', '36d563e6b.jpg', '372e173f7.jpg', '3730c15d7.jpg', '37319d9cc.jpg', '37368759e.jpg', '3736d042c.jpg', '3752425f7.jpg', '378272756.jpg', '38b9631df.jpg', '38e89262f.jpg', '392be1c76.jpg', '3963aac50.jpg', '39dcf53f9.jpg', '3a1ee3897.jpg', '3a75cf4ec.jpg', '3ac931a18.jpg', '3acf04349.jpg', '3b3687032.jpg', '3b6c36185.jpg', '3b6f6868f.jpg', '3b7b7ceaf.jpg', '3b7e44715.jpg', '3c6854422.jpg', '3c93807f4.jpg', '3c9691d0c.jpg', '3cc137693.jpg', '3cee96935.jpg', '3cfa60f2d.jpg', '3cfda2739.jpg', '3d060ac2d.jpg', '3da41c9f0.jpg', '3dac7e059.jpg', '3de4313ab.jpg', '3df222f6a.jpg', '3e1b9d319.jpg', '3e430ace5.jpg', '3e49a04a2.jpg', '3e6c0f7a4.jpg', '3ecd7753b.jpg', '3f0e0a371.jpg', '3f17e4acc.jpg', '3f6145c7d.jpg', '3f62cb71d.jpg', '3f9314e1d.jpg', '3fcbe5470.jpg', '418b7376c.jpg', '418bf2729.jpg', '41f81f578.jpg', '4223116c6.jpg', '424b57fbe.jpg', '42636127e.jpg', '42d4d28f6.jpg', '42f269efe.jpg', '43a10ca4f.jpg', '43be155b2.jpg', '43c1a4a70.jpg', '43cdf7d8d.jpg', '43fe37865.jpg', '443d3ddd0.jpg', '445a4f77f.jpg', '44859b1b9.jpg', '448a73204.jpg', '4490e43c1.jpg', '44aca2961.jpg', '44bbde24d.jpg', '44ca85273.jpg', '454d8aab4.jpg', '456241aa5.jpg', '457e96b39.jpg', '45b112be2.jpg', '467e97477.jpg', '47608300f.jpg', '4778df12c.jpg', '47c1a0f4a.jpg', '47c9178a3.jpg', '48786fbac.jpg', '48994af29.jpg', '48c23125e.jpg', '49289e055.jpg', '499a9893b.jpg', '499e41bff.jpg', '49b1c08c6.jpg', '49d7ee139.jpg', '4a84c317a.jpg', '4ad8ea918.jpg', '4aea81288.jpg', '4b6d44881.jpg', '4c3feb803.jpg', '4c5671c92.jpg', '4c63aa5c5.jpg', '4c6a757c4.jpg', '4c7d0c617.jpg', '4c805698a.jpg', '4ce6b0ff9.jpg', '4d33bf863.jpg', '4d429c29f.jpg', '4dc0acaf8.jpg', '4dde69bdb.jpg', '4e0891122.jpg', '4e5c4ab98.jpg', '4e6339bd8.jpg', '4e8309017.jpg', '4efaced22.jpg', '4f84396bf.jpg', '4fd6d8734.jpg', '4fdd51899.jpg', '50377feca.jpg', '512652b4f.jpg', '5144b32b4.jpg', '517fffea8.jpg', '52a92951f.jpg', '52bef8669.jpg', '53592ec18.jpg', '53660ffd5.jpg', '53ffd9993.jpg', '5406c01d7.jpg', '541e5bfec.jpg', '544f5beba.jpg', '54a757589.jpg', '54d3df55b.jpg', '5500d78ed.jpg', '555917097.jpg', '559759f28.jpg', '559aceb9e.jpg', '56bdb9e3c.jpg', '56d3153f9.jpg', '56d707102.jpg', '573d1f811.jpg', '574f43979.jpg', '57cabd349.jpg', '585774b65.jpg', '587a9b3b2.jpg', '5895de1c3.jpg', '589dd7157.jpg', '58f22628e.jpg', '5900ac269.jpg', '595426f59.jpg', '59c0a3475.jpg', '5a7c94126.jpg', '5a9149cad.jpg', '5aa035361.jpg', '5aa3780c6.jpg', '5ae4f1314.jpg', '5b3a4b5ad.jpg', '5b4743e28.jpg', '5bbf5aa3f.jpg', '5c0dade54.jpg', '5c6d48548.jpg', '5c6fdc38e.jpg', '5c9df14bd.jpg', '5d304901c.jpg', '5d80961c1.jpg', '5d990dafc.jpg', '5da555152.jpg', '5e049970d.jpg', '5e23a5dd3.jpg', '5e2e9fbed.jpg', '5f1308e91.jpg', '5f2f8397d.jpg', '5f627b6b6.jpg', '5f9e63b94.jpg', '5ff740a19.jpg', '60b2a4e95.jpg', '60dc13c8f.jpg', '611f72ff8.jpg', '61243b268.jpg', '61fc6d361.jpg', '61ffc10bd.jpg', '627b75778.jpg', '63061a1ca.jpg', '632d55c7c.jpg', '63e8fd4d7.jpg', '64273b706.jpg', '6463cc692.jpg', '64a5153e0.jpg', '64d1f29e0.jpg', '64e741b4d.jpg', '656b2014f.jpg', '657bc3dab.jpg', '6583c7755.jpg', '65bb87731.jpg', '65de19b8e.jpg', '666567e46.jpg', '6686b7448.jpg', '668fed093.jpg', '66d62df71.jpg', '66e24b327.jpg', '670dda82e.jpg', '674544933.jpg', '67a345551.jpg', '67f3733e1.jpg', '67fd6ce4d.jpg', '681775be8.jpg', '681d2c161.jpg', '68281b5f0.jpg', '68afa072c.jpg', '68d1c6f63.jpg', '690fbb138.jpg', '69d2fe769.jpg', '69d9e3b5b.jpg', '6a38180d6.jpg', '6a75462f9.jpg', '6ad0e93f3.jpg', '6b414548c.jpg', '6b8b34d8a.jpg', '6b9d0b108.jpg', '6c4b43ee0.jpg', '6c9db53a5.jpg', '6d0102129.jpg', '6d0bcc474.jpg', '6d1b0740c.jpg', '6d422b814.jpg', '6d422d496.jpg', '6d5731630.jpg', '6d841409e.jpg', '6dc0f0256.jpg', '6dee56da7.jpg', '6e0a0050c.jpg', '6e1715d95.jpg', '6e33e7465.jpg', '6eb8caa4a.jpg', '6ec0efcf3.jpg', '6eec49a0b.jpg', '6f12a26e0.jpg', '6f18c587b.jpg', '6f88c6d64.jpg', '6fa8c306d.jpg', '6fc70467c.jpg', '6fd029d4d.jpg', '6fdb86bdf.jpg', '7081b8507.jpg', '7146e52ed.jpg', '714f9d00b.jpg', '719090053.jpg', '71f8c87bd.jpg', '72341723f.jpg', '7286d03ef.jpg', '72ed9c3d8.jpg', '732f9ff8b.jpg', '735aed3ac.jpg', '73779d0e5.jpg', '73850514e.jpg', '7392cb034.jpg', '73aef6a73.jpg', '73ff8efef.jpg', '74a57a20a.jpg', '74f951ca4.jpg', '7521c2550.jpg', '758437094.jpg', '75b24ab86.jpg', '75e4cf240.jpg', '75efd5e42.jpg', '75f0c1e00.jpg', '76288fc9c.jpg', '768e90b47.jpg', '76ee7f36f.jpg', '77005f2f4.jpg', '7758e7e93.jpg', '781b78af7.jpg', '782c9966e.jpg', '78fd79c06.jpg', '792038159.jpg', '7963573a1.jpg', '7979e93cd.jpg', '79d9ff869.jpg', '79e309e7b.jpg', '79e6f2b42.jpg', '79ea45aa1.jpg', '7a1986428.jpg', '7a1c520b9.jpg', '7a4e42374.jpg', '7a512ca2c.jpg', '7a5288cb8.jpg', '7add4314d.jpg', '7ae21e551.jpg', '7b0435807.jpg', '7b1ed2ad7.jpg', '7b3bb39d1.jpg', '7b455e37c.jpg', '7bed4da05.jpg', '7bfa422df.jpg', '7c1543f80.jpg', '7c46d8f79.jpg', '7c5534db1.jpg', '7cc1fb6ee.jpg', '7cccf944b.jpg', '7cd146e5e.jpg', '7cd86fd14.jpg', '7cda61a73.jpg', '7d09276e5.jpg', '7d62939af.jpg', '7e104caad.jpg', '7e1d18b03.jpg', '7e8904c22.jpg', '7eb199869.jpg', '7f7901508.jpg', '7f9ea512e.jpg', '80111d56e.jpg', '80657f706.jpg', '808359bda.jpg', '80cbe5d35.jpg', '80d489d3f.jpg', '80d4f7a25.jpg', '817d399e2.jpg', '8187f1b45.jpg', '81a5138eb.jpg', '81da020a0.jpg', '830ebda8b.jpg', '83325a620.jpg', '837f2703c.jpg', '83926cb04.jpg', '83a0dbb43.jpg', '83a187036.jpg', '846fff7a8.jpg', '8475899d7.jpg', '84e348664.jpg', '86d73c437.jpg', '86fe3cf8c.jpg', '874f5fe53.jpg', '87c5bbcba.jpg', '87f4645ee.jpg', '8827f75d2.jpg', '88afdeeb8.jpg', '88bfc04e8.jpg', '88cacaf33.jpg', '8973e4e91.jpg', '89977988b.jpg', '899a2ad2f.jpg', '89a629daa.jpg', '89e69dbf2.jpg', '8a5484d4d.jpg', '8b08c23f0.jpg', '8bc3d26cb.jpg', '8bd6ef3fb.jpg', '8c50370fe.jpg', '8d40206b1.jpg', '8d5f997f2.jpg', '8d8bd106b.jpg', '8dbbdcb59.jpg', '8e1cf2257.jpg', '8e4dbe30f.jpg', '8e703053f.jpg', '8f26dbe33.jpg', '8f480a549.jpg', '8f502dacc.jpg', '8ff2bd9f3.jpg', '8ff37df24.jpg', '9056783d7.jpg', '90928cdb6.jpg', '911e86fb7.jpg', '9132d6f86.jpg', '91376eb04.jpg', '91604e4f4.jpg', '9177da79f.jpg', '91a32baec.jpg', '91ef23069.jpg', '92f506757.jpg', '932101fec.jpg', '9354f7429.jpg', '9414a468c.jpg', '943a908b4.jpg', '944afa4e4.jpg', '9484781db.jpg', '948f0cad1.jpg', '94a687dea.jpg', '94b1d0288.jpg', '94c65ec76.jpg', '950487205.jpg', '9520225ec.jpg', '9553248f2.jpg', '95587f05e.jpg', '9568a10f7.jpg', '958feb050.jpg', '95aa55d7d.jpg', '95dca6ffa.jpg', '95eb51548.jpg', '9676c628e.jpg', '967ecac66.jpg', '96a864d9e.jpg', '96fefb723.jpg', '971cb01d4.jpg', '97880f512.jpg', '980d73519.jpg', '98861ce3b.jpg', '98bae0369.jpg', '98da1cdfd.jpg', '990852160.jpg', '9930cc99a.jpg', '9971b06fb.jpg', '99ced6e4a.jpg', '9aa6cf465.jpg', '9abf4e920.jpg', '9add1b7db.jpg', '9aec02d72.jpg', '9b06a292b.jpg', '9b16a82de.jpg', '9b5147492.jpg', '9b66b8330.jpg', '9c1af32c2.jpg', '9c595e23e.jpg', '9cbc700db.jpg', '9cc10ade1.jpg', '9cd461813.jpg', '9cebc2686.jpg', '9d143b20f.jpg', '9d2adc69b.jpg', '9d2b4edf3.jpg', '9e24388bd.jpg', '9ebdc1f1f.jpg', '9f7498252.jpg', '9f7a5b38b.jpg', '9fcc459b7.jpg', '9fe56a5a6.jpg', '9fef81473.jpg', 'a02e9f525.jpg', 'a07c95dec.jpg', 'a0838ee79.jpg', 'a0b280324.jpg', 'a0d37c784.jpg', 'a0fd8f9d6.jpg', 'a138d1604.jpg', 'a20204e10.jpg', 'a2423ec13.jpg', 'a269d58e3.jpg', 'a29360bc1.jpg', 'a301b86f5.jpg', 'a3849d778.jpg', 'a40c26912.jpg', 'a437f34dc.jpg', 'a44298fe1.jpg', 'a46396cca.jpg', 'a4a2f54f4.jpg', 'a4c9fb612.jpg', 'a4fe1adda.jpg', 'a521ed387.jpg', 'a55578b4c.jpg', 'a5a94ede7.jpg', 'a5e151ad8.jpg', 'a631d53aa.jpg', 'a69f95c48.jpg', 'a7377c95e.jpg', 'a79b53847.jpg', 'a7e803b17.jpg', 'a821909ee.jpg', 'a83fc096c.jpg', 'a8675325e.jpg', 'a873b638e.jpg', 'a8ab0eae9.jpg', 'a94261525.jpg', 'aa1d1e861.jpg', 'aa604ec06.jpg', 'aaebf8aa6.jpg', 'ab3a9a988.jpg', 'ab730950d.jpg', 'ab7ad60ea.jpg', 'ab9d7104b.jpg', 'abb5bcc54.jpg', 'ac4eb0b7f.jpg', 'aca4c0bf2.jpg', 'acc96f9ec.jpg', 'ad9471f70.jpg', 'ad9c71145.jpg', 'adab9d36c.jpg', 'add9aa6e6.jpg', 'ae15916ee.jpg', 'ae32bf450.jpg', 'ae3cf9bab.jpg', 'ae429576b.jpg', 'ae831607d.jpg', 'ae89187b3.jpg', 'aea7d0dcf.jpg', 'aefeb52b4.jpg', 'afaafa0e2.jpg', 'afc3e3e90.jpg', 'afec9857d.jpg', 'b0050b626.jpg', 'b03b804f4.jpg', 'b069e70a9.jpg', 'b0b5a3357.jpg', 'b1166a68f.jpg', 'b1648ba07.jpg', 'b1b83ad86.jpg', 'b1c324193.jpg', 'b214b85d0.jpg', 'b2491c424.jpg', 'b2aea7a20.jpg', 'b2c7e1161.jpg', 'b2fae3494.jpg', 'b371745dc.jpg', 'b388c6de3.jpg', 'b3912fc50.jpg', 'b399d14c2.jpg', 'b3a766a95.jpg', 'b3f594ba3.jpg', 'b461a25f2.jpg', 'b4a113603.jpg', 'b4d365c88.jpg', 'b4dfd853c.jpg', 'b506c5877.jpg', 'b51cdf84f.jpg', 'b6427c1f5.jpg', 'b64844bed.jpg', 'b64f40c00.jpg', 'b6d18d011.jpg', 'b6d316ee0.jpg', 'b6d419268.jpg', 'b6db2946b.jpg', 'b6edd61b7.jpg', 'b7018c2a5.jpg', 'b742bde4b.jpg', 'b760debb1.jpg', 'b85a0fa3b.jpg', 'b98946867.jpg', 'b991340c8.jpg', 'ba1196257.jpg', 'ba4168e4d.jpg', 'ba8ebb757.jpg', 'bb131c78e.jpg', 'bb13a714f.jpg', 'bb3ed626f.jpg', 'bb5322eae.jpg', 'bbc5bbff4.jpg', 'bbe2bd39d.jpg', 'bc34d04f1.jpg', 'bc6d00672.jpg', 'bc844c51a.jpg', 'bcc1fd0e4.jpg', 'bcf7d4b9a.jpg', 'bcf8c8739.jpg', 'bd96c3707.jpg', 'bd9de1aa3.jpg', 'bdb1f5a68.jpg', 'bddb8e7c7.jpg', 'be6f60f41.jpg', 'be76e9ca7.jpg', 'bee40f5e8.jpg', 'bf275fe13.jpg', 'bf537d326.jpg', 'c0473efc0.jpg', 'c090c3c13.jpg', 'c0bd7fecd.jpg', 'c1c3f9072.jpg', 'c1e1e7b0c.jpg', 'c224552e0.jpg', 'c234bf0a0.jpg', 'c2b52f420.jpg', 'c2cd72225.jpg', 'c37e5676c.jpg', 'c3db04098.jpg', 'c3e859718.jpg', 'c3f9989d5.jpg', 'c43fd3530.jpg', 'c4405bef9.jpg', 'c4617b332.jpg', 'c4987c901.jpg', 'c50de1f31.jpg', 'c521ad2a6.jpg', 'c55581dff.jpg', 'c57bbee34.jpg', 'c5bd13a82.jpg', 'c66f8a343.jpg', 'c6bc7ca4b.jpg', 'c6f648543.jpg', 'c716d86d5.jpg', 'c723f1ada.jpg', 'c725bafda.jpg', 'c7335eee0.jpg', 'c7667ac1b.jpg', 'c7fb01bad.jpg', 'c8144de93.jpg', 'c82561e06.jpg', 'c8311ec86.jpg', 'c832f2a91.jpg', 'c86f91dbf.jpg', 'c8f2d68aa.jpg', 'c91c10abe.jpg', 'c93217423.jpg', 'c96562a3f.jpg', 'c9737bd79.jpg', 'c9ca9769f.jpg', 'c9e2ce466.jpg', 'ca3f36941.jpg', 'ca57b6361.jpg', 'caf4103e1.jpg', 'cb17da791.jpg', 'cb241fa5d.jpg', 'cb28e4857.jpg', 'cba3fdc7a.jpg', 'cbb1d7ff5.jpg', 'cc069dc99.jpg', 'cc6e5e804.jpg', 'cc9ddbc17.jpg', 'ccb799c73.jpg', 'cd17f385a.jpg', 'cd75780c0.jpg', 'cde486dbf.jpg', 'cdfb634a5.jpg', 'ce1c08011.jpg', 'ce6170a7f.jpg', 'cf0483836.jpg', 'cf06b295a.jpg', 'cf075fd69.jpg', 'd01da361f.jpg', 'd04a7fb6c.jpg', 'd0793e6b2.jpg', 'd0cee7515.jpg', 'd18e48954.jpg', 'd1ae5708a.jpg', 'd1b3aa6e7.jpg', 'd1d054d6f.jpg', 'd22364fb6.jpg', 'd249e6983.jpg', 'd25120582.jpg', 'd3801de15.jpg', 'd42853714.jpg', 'd42d5725e.jpg', 'd49ed89d0.jpg', 'd4a0cb9af.jpg', 'd4ac8538b.jpg', 'd5137b6f8.jpg', 'd52fefa0f.jpg', 'd599b529f.jpg', 'd5ae74330.jpg', 'd5bbdc8a8.jpg', 'd5e8bbe9a.jpg', 'd5fab51f7.jpg', 'd6128fbfc.jpg', 'd6377dc32.jpg', 'd6677e1ff.jpg', 'd670249ed.jpg', 'd67e52625.jpg', 'd6cba7ed9.jpg', 'd704ab756.jpg', 'd7741a3ba.jpg', 'd8129a060.jpg', 'd838ab96a.jpg', 'd866b5d5c.jpg', 'd8697da18.jpg', 'd888063ea.jpg', 'd8f8075c8.jpg', 'da891d3af.jpg', 'dab41427c.jpg', 'dae47b14e.jpg', 'db021d031.jpg', 'db6a9172c.jpg', 'db758d243.jpg', 'dbfc43562.jpg', 'dc2205ab2.jpg', 'dc89301ff.jpg', 'dc8d040ab.jpg', 'dd4531cef.jpg', 'dd460a71f.jpg', 'dd9364137.jpg', 'ddca8eddd.jpg', 'de2df260c.jpg', 'de5881c2b.jpg', 'de7556be9.jpg', 'dec5aa06a.jpg', 'dedf68f1e.jpg', 'def7d77bd.jpg', 'defe590d5.jpg', 'df0facbbd.jpg', 'df6767e85.jpg', 'dfb1ab260.jpg', 'dfc265237.jpg', 'e01b0d22c.jpg', 'e07986242.jpg', 'e0b422958.jpg', 'e0d3ed478.jpg', 'e0f059688.jpg', 'e1090545d.jpg', 'e16b5476a.jpg', 'e237a3954.jpg', 'e30317c29.jpg', 'e30e861c8.jpg', 'e31cb8391.jpg', 'e380b37ba.jpg', 'e391fbf7c.jpg', 'e397fcb9c.jpg', 'e4168f460.jpg', 'e460fa399.jpg', 'e4612d7c4.jpg', 'e49ca4ccc.jpg', 'e55cdcaaf.jpg', 'e57a18b8f.jpg', 'e5803c399.jpg', 'e5c133bf8.jpg', 'e61b83be0.jpg', 'e62ab86cd.jpg', 'e7c0e2635.jpg', 'e7d90a9e5.jpg', 'e7e953217.jpg', 'e80bc63e0.jpg', 'e83b28579.jpg', 'e89125653.jpg', 'e89a4ca99.jpg', 'e8a1e41ef.jpg', 'e904c0dc2.jpg', 'e968f8497.jpg', 'e9843e450.jpg', 'e9d339446.jpg', 'ea8c1f361.jpg', 'eab87e269.jpg', 'eacf21af0.jpg', 'ec0e63edf.jpg', 'ec1e848f0.jpg', 'ec44edc94.jpg', 'ec6132d16.jpg', 'ec6434bac.jpg', 'ec7ab4de4.jpg', 'ec96e3f18.jpg', 'ecdcdfc6d.jpg', 'edfba47f3.jpg', 'ee047d59d.jpg', 'ee6666c5c.jpg', 'ef4b7c6be.jpg', 'ef5bd0f19.jpg', 'ef84d8825.jpg', 'ef8e8852a.jpg', 'ef963aadc.jpg', 'efb9face0.jpg', 'efc0006fa.jpg', 'efdf91da2.jpg', 'f04c7c5ac.jpg', 'f0842d221.jpg', 'f0fa65d89.jpg', 'f1338af36.jpg', 'f14a2776e.jpg', 'f14eb3bba.jpg', 'f1d90c01a.jpg', 'f1d98dffe.jpg', 'f1ff7d363.jpg', 'f2313afbd.jpg', 'f246790fc.jpg', 'f25152812.jpg', 'f289fa572.jpg', 'f3e31ebb4.jpg', 'f3e43b28e.jpg', 'f4ac9794b.jpg', 'f50a536c6.jpg', 'f550b3e1b.jpg', 'f605a77df.jpg', 'f625f93a1.jpg', 'f779cc81a.jpg', 'f7c7e853c.jpg', 'f7e741823.jpg', 'f7f2c2817.jpg', 'f80ed5101.jpg', 'f82d83496.jpg', 'f83e863db.jpg', 'f89ea01f9.jpg', 'f8e1d1d25.jpg', 'f9399b2df.jpg', 'f947a5bf6.jpg', 'f948cddb2.jpg', 'f98450a9d.jpg', 'f99381ed5.jpg', 'f9a5b9bea.jpg', 'f9c2ebdcf.jpg', 'f9ccc4b77.jpg', 'f9d438665.jpg', 'fa2a43fe9.jpg', 'fa32469ab.jpg', 'fa52724c1.jpg', 'fa5900edc.jpg', 'fb47d1546.jpg', 'fb878c641.jpg', 'fb8870b74.jpg', 'fbca0c4fb.jpg', 'fbf73dccc.jpg', 'fc1a8fe78.jpg', 'fc1dae347.jpg', 'fc214bcbc.jpg', 'fc2500486.jpg', 'fc28f7fe0.jpg', 'fc3c8279e.jpg', 'fc3ec57fb.jpg', 'fc6ec9b13.jpg', 'fc9258351.jpg', 'fc9b9483f.jpg', 'fca5873cd.jpg', 'fcbff7cb3.jpg', 'fcf678d47.jpg', 'fd0e64e40.jpg', 'fd61e999d.jpg', 'fd63c350a.jpg', 'fd92267ca.jpg', 'fdacd4806.jpg', 'fe03f768e.jpg', 'fe3116441.jpg', 'fe7315df8.jpg', 'ff18d8aa8.jpg', 'ff4d29e0d.jpg', 'ffa10f6f3.jpg', 'ffaf13d96.jpg', 'ffbf79783.jpg', 'ffdb60677.jpg']

    ###################### seg #########################
    # config_0 = Config(task='seg', architecture='Unet', encoder='se_resnext50_32x4d',
    #                   checkpoint='_results/Unet_se_resnext50_shift/checkpoints/epoch_0049_score0.9452_loss0.1433.pth')
    # fold_0 = run(config_0, nonmissing_fnames)

    # config_1 = Config(task='seg', architecture='Unet', encoder='efficientnet-b3',
    #                   checkpoint='_results/Unet_eff-b3_shift/checkpoints/epoch_0049_score0.9471_loss0.1393.pth')
    # fold_1 = run(config_1, nonmissing_fnames)
    # fold_0 += fold_1
    # del fold_1

    # config_2 = Config(task='seg', architecture='Unet', encoder='efficientnet-b4',
    #                   checkpoint='_results/Unet_eff-b4_shift/checkpoints/epoch_0039_score0.9455_loss0.1430.pth')
    # fold_2 = run(config_2, nonmissing_fnames)
    # fold_0 += fold_2
    # del fold_2

    # config_3 = Config(task='seg', architecture='Unet', encoder='efficientnet-b2',
    #                   checkpoint='_results/Unet_eff-b2_shift/checkpoints/epoch_0041_score0.9478_loss0.1377.pth')
    # fold_3 = run(config_3, nonmissing_fnames)
    # fold_0 += fold_3
    # del fold_3

    config_5 = Config(task='seg', architecture='Unet', encoder='efficientnet-b5',
                      # checkpoint='_results/Unet_eff-b5_shift/checkpoints/epoch_0071_score0.9471_loss0.1403.pth')
                      checkpoint='_results/Unet_eff-b5_shift/checkpoints/epoch_0086_score0.9476_loss0.1389.pth')
                      # checkpoint='_results/Unet_eff-b5_train_all/checkpoints/epoch_0088_score0.9476_loss0.1378.pth')
    fold_5 = run(config_5, nonmissing_fnames)

    # config_4 = Config(task='seg', architecture='Unet', encoder='efficientnet-b4',
    #                   checkpoint='_results/Unet_eff-b4_shift/checkpoints/epoch_0071_score0.9469_loss0.1401.pth')
                      # checkpoint='_results/Unet_eff-b4_train_all/checkpoints/epoch_0076_score0.9471_loss0.1402.pth')
    # fold_4 = run(config_4, nonmissing_fnames)

    # config_2 = Config(task='seg', architecture='Unet', encoder='efficientnet-b2',
    #                   checkpoint='_results/Unet_eff-b2_shift/checkpoints/epoch_0041_score0.9478_loss0.1377.pth')
    #                   checkpoint='_results/Unet_eff-b2_shift/checkpoints/epoch_0100_score0.9489_loss0.1351.pth')
                      # checkpoint='_results/Unet_eff-b2_train_all/checkpoints/epoch_0105_score0.9501_loss0.1332.pth')
    # fold_2 = run(config_2, nonmissing_fnames)

    config_1 = Config(task='seg', architecture='Unet', encoder='efficientnet-b1',
    #                   checkpoint='_results/Unet_eff-b1_shift/checkpoints/epoch_0047_score0.9467_loss0.1407.pth')
                      checkpoint='_results/Unet_eff-b1_shift/checkpoints/epoch_0078_score0.9481_loss0.1404.pth')
                      # checkpoint='_results/Unet_eff-b1_train_all/checkpoints/epoch_0083_score0.9488_loss0.1363.pth')
    fold_1 = run(config_1, nonmissing_fnames)

    config_1_ = Config(task='seg', architecture='Unet', encoder='efficientnet-b1',
                      checkpoint='_results/Unet_eff-b1_fold7/checkpoints/epoch_0076_score0.9533_loss0.1270.pth')
    fold_1_ = run(config_1_, nonmissing_fnames)


    # final = fold_0
    # final = (fold_0 + fold_1 + fold_2) / 3.0
    # final = (fold_0 + fold_1 + fold_2 + fold_3) / 4.0  # (N, 4, 256, 1600)
    final = (fold_5 + fold_1 + fold_1_) / 3.0  # (N, 4, 256, 1600)
    np.save('final_' + postfix + '.npy', final)

    ####################################################
    # os.makedirs('submissions', exist_ok=True)
    #
    # submission['EncodedPixels'] = ''  # 7204
    #
    # for idx in range(final.shape[0]):  # 887
    #     # preds = [mask2rle(post_process(final[idx][i], threshold=threshold[i], min_size=min_size, fill_up=fill_up)) for i in range(4)]
    #
    #     preds = []
    #     for ch in range(4):
    #         if nonmissing_cls_output[idx][ch] == 0:
    #             preds.append('')
    #         else:
    #             preds.append(mask2rle(post_process(final[idx][ch], threshold=threshold[ch], min_size=min_size, fill_up=fill_up)))
    #
    #     submission.loc[submission['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) == nonmissing_fnames[idx], 'EncodedPixels'] = preds
    #
    #
    # submission.to_csv(os.path.join('submissions', 'submission_' + postfix + '.csv'), index=False)
    print('success!')

    ####################################################
    '''
    # save pseudo-labels
    os.makedirs('pseudo-labels', exist_ok=True)

    # for cls(fold)
    cls_pseudo = pd.DataFrame(columns=['ImageId', 'hasMask', 'ClassIds', 'split'])
    cls_pseudo['ImageId'] = ImageIds
    # cls_output = np.where(cls_output >= 0.5, 1, 0).astype(np.int32)
    # cls_pseudo['hasMask'] = cls_output

    for k in range(cls_output.shape[0]):
        where = np.where(cls_output[k] == 1)[0]
        where += 1
        where = [str(p) for p in list(where)]

        classid = ''.join(where)

        cls_pseudo['ClassIds'][k] = classid

    cls_pseudo['split'] = 'train'
    cls_pseudo.to_csv(os.path.join('pseudo-labels', 'pseudo_' + postfix + '.csv'), index=False)
    
    
    # for seg(train)
    submission.loc[submission['EncodedPixels'] == '', 'EncodedPixels'] = np.nan
    submission['ImageId'] = submission['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    submission['ClassId'] = submission['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
    submission['hasMask'] = ~ submission['EncodedPixels'].isna()
    submission.to_csv(os.path.join('pseudo-labels', 'train_pseudo_' + postfix + '.csv'), index=False)

    # for seg(fold)
    seg_pseudo = submission.groupby('ImageId').agg(np.sum).reset_index()

    class_ids = []
    for id in seg_pseudo['ImageId'].values:
        class_ids.append(
            "".join((submission.loc[(submission['ImageId'] == id) & (submission['hasMask'] == True), 'ClassId'].values)))
    seg_pseudo['ClassIds'] = class_ids

    seg_pseudo['hasMask'] = (seg_pseudo['hasMask'] > 0).astype(int)
    seg_pseudo['split'] = 'train'

    seg_pseudo.to_csv(os.path.join('pseudo-labels', 'seg_pseudo_' + postfix + '.csv'), index=False)
    '''


class Config():
    def __init__(self, task, model=None, gem=False, add_fc=None, architecture=None, encoder=None, checkpoint=None):
        self.TASK = task
        self.MODEL = model
        self.GEM = gem
        self.ADD_FC = add_fc
        self.ARCHITECTURE = architecture
        self.ENCODER = encoder
        self.CHECKPOINT = checkpoint

        self.IMG_H = 256
        self.IMG_W = 1600

        self.DATA_DIR = 'data/test_images'
        self.SAMPLE_SUBMISSION = 'data/sample_submission.csv'

        self.DEBUG = False

        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 4


if __name__ == '__main__':
    start = time.time()
    main()
    ellapsed = time.time() - start
    print('Total inference time: %d hours %d minutes %d seconds' % (ellapsed // 3600, (ellapsed % 3600) // 60, (ellapsed % 3600) % 60))

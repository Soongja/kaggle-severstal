import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from albumentations import (
    OneOf, Compose,
    Flip, ShiftScaleRotate, GridDistortion, ElasticTransform,
    RandomGamma, RandomContrast, RandomBrightness, RandomBrightnessContrast,
    Blur, MedianBlur, MotionBlur,
    CLAHE, IAASharpen, GaussNoise,
    HueSaturationValue, RGBShift)


def strong_aug(p=1.0):
    return Compose([
        Flip(p=0.75),
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75)
        # ShiftScaleRotate(p=1.0, shift_limit=0.5, scale_limit=0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT),
        # OneOf([
        #     GridDistortion(p=1.0),
        #     ElasticTransform(p=1.0)
        # ], p=0.2),
        # OneOf([
        #     Blur(p=1.0),
        #     MedianBlur(p=1.0),
        #     MotionBlur(p=1.0),
        # ], p=0.3),
        # OneOf([
        #     GaussNoise(p=1.0),
        #     HueSaturationValue(p=1.0),
        #     RGBShift(p=1.0)
        # ], p=0.4),
        # OneOf([
        #     CLAHE(p=1.0),
        #     IAASharpen(p=1.0),
        # ], p=0.2)
    ], p=p)


class RandomHorizontalShift():
    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, image, mask):

        if random.random() <= self.p:
            dx = random.uniform(-0.5, 0.5)

            height, width = image.shape[:2]
            center = (width / 2, height / 2)

            matrix = cv2.getRotationMatrix2D(center, angle=0.0, scale=1.0)
            matrix[0, 2] += dx * width

            image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return image, mask


class Albu_Seg():
    def __call__(self, image, mask):
        augmentation = strong_aug()

        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)

        image, mask = augmented["image"], augmented["mask"]

        # shift
        shift = RandomHorizontalShift(p=0.75)
        image, mask = shift(image, mask)

        return image, mask


class Albu_Cls():
    def __call__(self, image):
        augmentation = strong_aug()

        data = {"image": image}
        augmented = augmentation(**data)

        image = augmented["image"]

        return image


class Albu_test():
    def __call__(self, image, mask):
        augmentation = Compose([
                            # Blur(p=0.5),
                            # MedianBlur(p=0.5),
                            MotionBlur(p=0.5),
                            # OneOf([
                            #     Blur(p=1.0),
                            #     MedianBlur(p=1.0),
                            #     MotionBlur(p=1.0),
                            # ], p=0.3),
                            # OneOf([
                            #     CLAHE(p=1.0),
                            #     IAASharpen(p=1.0),
                            # ], p=0.2)
                            # CLAHE(p=0.5),
                            # IAASharpen(p=0.5)

                        ], p=1.0)

        data = {"image": image, "mask": mask}
        augmented = augmentation(**data)

        image, mask = augmented["image"], augmented["mask"]

        return image, mask


if __name__ == "__main__":
    # aug = Albu_Seg()
    # aug = RandomHorizontalShift()
    aug = Albu_test()

    img = cv2.imread('000f6bf48.jpg', 0)
    mask = cv2.imread('000f6bf48_4.jpg', 0)

    for i in range(100):
        out_img, out_mask = aug(img, mask)

        cv2.imshow('img', out_img)
        cv2.imshow('mask', out_mask)
        cv2.waitKey()

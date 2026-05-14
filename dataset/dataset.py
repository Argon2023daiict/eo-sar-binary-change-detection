
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import warnings
warnings.filterwarnings("ignore")

import cv2
cv2.setNumThreads(0)

import glob
import torch
import numpy as np

from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChangeDetectionDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        image_size=256,
    ):

        self.root_dir = root_dir

        self.split = split

        self.image_size = image_size

        self.pre_dir = os.path.join(
            root_dir,
            "pre-event",
        )

        self.post_dir = os.path.join(
            root_dir,
            "post-event",
        )

        self.target_dir = os.path.join(
            root_dir,
            "target",
        )

        self.pre_images = sorted(
            glob.glob(
                os.path.join(
                    self.pre_dir,
                    "*.tif",
                )
            )
        )

        self.post_images = sorted(
            glob.glob(
                os.path.join(
                    self.post_dir,
                    "*.tif",
                )
            )
        )

        self.targets = sorted(
            glob.glob(
                os.path.join(
                    self.target_dir,
                    "*.tif",
                )
            )
        )

        print(
            f"[{split}] Found {len(self.pre_images)} samples"
        )

        if split == "train":

            self.transforms = A.Compose([

                A.RandomResizedCrop(
                    size=(
                        image_size,
                        image_size,
                    ),
                    scale=(0.7, 1.0),
                    p=1.0,
                ),

                A.HorizontalFlip(p=0.5),

                A.VerticalFlip(p=0.5),

                A.RandomRotate90(p=0.5),

                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.05, 0.05),
                    rotate=(-15, 15),
                    shear=(-10, 10),
                    p=0.5,
                ),

                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),

                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.3,
                ),

                A.GaussianBlur(
                    blur_limit=(3, 5),
                    p=0.2,
                ),

                A.MotionBlur(
                    blur_limit=3,
                    p=0.2,
                ),

                A.GaussNoise(
                    std_range=(0.02, 0.08),
                    p=0.3,
                ),

                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(16, 32),
                    hole_width_range=(16, 32),
                    fill=0,
                    p=0.3,
                ),

                ToTensorV2(),
            ])

        else:

            self.transforms = A.Compose([

                A.Resize(
                    image_size,
                    image_size,
                ),

                ToTensorV2(),
            ])

    def __len__(self):

        return len(self.pre_images)

    def load_pre(self, path):

        img = cv2.imread(
            path,
            cv2.IMREAD_COLOR,
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )

        img = img.astype(np.float32) / 255.0

        return img

    def load_post(self, path):

        img = cv2.imread(
            path,
            cv2.IMREAD_UNCHANGED,
        )

        if len(img.shape) == 3:

            img = img[:, :, 0]

        img = img.astype(np.float32)

        img = (
            img - img.min()
        ) / (
            img.max() - img.min() + 1e-6
        )

        return img

    def load_target(self, path):

        mask = cv2.imread(
            path,
            cv2.IMREAD_UNCHANGED,
        )

        if len(mask.shape) == 3:

            mask = mask[:, :, 0]

        mask = (
            mask > 0
        ).astype(np.float32)

        return mask

    def __getitem__(self, idx):

        pre = self.load_pre(
            self.pre_images[idx]
        )

        post = self.load_post(
            self.post_images[idx]
        )

        target = self.load_target(
            self.targets[idx]
        )

        transformed = self.transforms(

            image=pre,

            masks=[
                post,
                target,
            ],
        )

        pre = transformed["image"]

        post = transformed["masks"][0]

        target = transformed["masks"][1]

        if len(post.shape) == 2:

            post = post.unsqueeze(0)

        if len(target.shape) == 2:

            target = target.unsqueeze(0)

        post = post.float()

        target = target.float()

        return {

            "pre": pre,

            "post": post,

            "target": target,
        }

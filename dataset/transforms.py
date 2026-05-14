
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentation(image_size=256):

    return A.Compose([

        # Spatial diversity
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            p=1.0,
        ),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),

        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=20,
            border_mode=0,
            p=0.5,
        ),

        # EO appearance robustness
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
            ),

            A.RandomGamma(
                gamma_limit=(80, 120),
                p=1.0,
            ),

        ], p=0.5),

        # Blur robustness
        A.OneOf([

            A.GaussianBlur(
                blur_limit=(3, 5),
                p=1.0,
            ),

            A.MotionBlur(
                blur_limit=(3, 5),
                p=1.0,
            ),

        ], p=0.3),

        # Sensor variability
        A.GaussNoise(
            std_range=(0.02, 0.05),
            p=0.3,
        ),

        # Strong regularization
        A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0,
            p=0.4,
        ),

        ToTensorV2(),
    ])


def get_val_augmentation(image_size=256):

    return A.Compose([

        A.Resize(
            image_size,
            image_size,
        ),

        ToTensorV2(),
    ])

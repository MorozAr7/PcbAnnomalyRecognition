import albumentations as A
from albumentations.pytorch import ToTensorV2
#import torchvision.transforms as A
import cv2
augmentation_training = A.Compose([
		A.RandomBrightness(limit=(-0.15, 0.15), p=1),
		A.HorizontalFlip(p=0.5),
		A.VerticalFlip(p=0.5),
		A.Rotate(p=1, limit=(-15, 15), border_mode=cv2.BORDER_REPLICATE),
		A.Rotate(p=0.5, limit=(180, 180), border_mode=cv2.BORDER_REPLICATE),
])

augmentation_noise = A.Compose([
    	A.OneOf([
			A.GaussNoise(var_limit=(25.0, 100.0), mean=0, per_channel=True, always_apply=True, p=1),
			A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=True, always_apply=True, p=1),
		], p=0.25)])

transform_to_tensor = A.Compose([A.Normalize(0, 1), ToTensorV2()])


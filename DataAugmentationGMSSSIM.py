import albumentations as A
from albumentations.pytorch import ToTensorV2
#import torchvision.transforms as A
import cv2
augmentation_training = A.Compose([
		#A.ColorJitter(brightness=(0.7, 1.5), hue=(-0.5, 0.5), saturation=(0.7, 1.5), contrast=(0.7, 1.5), p=1),
		##A.RGBShift(r_shift_limit=(-50, 50), g_shift_limit=(-50, 50), b_shift_limit=(-50, 50), p=1),
		A.RandomBrightness(limit=(-0.15, 0.15), p=1),
		A.HorizontalFlip(p=0.5),
		A.VerticalFlip(p=0.5),
		#A.MultiplicativeNoise()
		A.Rotate(p=1, limit=(-15, 15), border_mode=cv2.BORDER_REPLICATE)
])

transform_to_tensor = A.Compose([A.Normalize(0, 1), ToTensorV2()])


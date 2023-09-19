import torch

LEARNING_RATE = 0.002
BATCH_SIZE = 16
APPLY_AUGMENTATION = True

DEVICE = "mps" if getattr(torch, 'has_mps', False) else 3 if torch.cuda.is_available() else "cpu"

NUM_MASK_SCALES = 3
IMAGE_SIZE = 256
MODEL_NAME = "DiscsQualityCheckModel3"
LossCoefficients = {"L1": 4, "SSIM": 4, "GradL1": 1, "GradSSIM": 1}


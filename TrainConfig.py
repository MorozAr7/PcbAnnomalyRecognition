import torch

LEARNING_RATE = 0.002
BATCH_SIZE = 20
APPLY_AUGMENTATION = True

DEVICE = "mps" if getattr(torch, 'has_mps', False) else 6 if torch.cuda.is_available() else "cpu"

NUM_MASK_SCALES = 3
IMAGE_SIZE = 256
MODEL_NAME = "Model4"
LossCoefficients = {"L1": 5, "SSIM": 1, "GradL1": 1, "GradSSIM": 1}


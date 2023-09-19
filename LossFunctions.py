from torch.nn.functional import conv2d
from TrainConfig import *
import torch.nn as nn


class L2RestorationLoss(nn.Module):
	def __init__(self):
		super(L2RestorationLoss, self).__init__()

	def forward(self, prediction, ground_truth, normalize=False):
		if normalize:
			normalization_constant = prediction.shape[1] #* prediction.shape[2] * prediction.shape[3]
			return torch.sum(torch.square(prediction - ground_truth)) / (2 * normalization_constant)
		else:
			return torch.sum(torch.square(prediction - ground_truth))


class SSIMLoss(nn.Module):
	def __init__(self):
		super(SSIMLoss, self).__init__()

		self.K1 = 0.01
		self.K2 = 0.03
		self.C1 = (self.K1 * 1)**2
		self.C2 = (self.K2 * 1)**2

	def compute_patches_mean(self, image):
		convolved = conv2d(input=image, weight=self.ones_filter/(self.window_size ** 2), stride=1)
		return convolved

	def compute_patches_std(self, image, mean_map):
		convolved = conv2d(image ** 2, weight=self.ones_filter/(self.window_size ** 2 - 1), stride=1) - mean_map ** 2
		return convolved

	def compute_joint_std(self, prediction, ground_truth, prediction_mean_map, ground_truth_mean_map):
		convolved = conv2d(input=prediction * ground_truth, weight=self.ones_filter/(self.window_size ** 2 - 1), stride=1) - prediction_mean_map * ground_truth_mean_map

		return convolved

	def forward(self, prediction, ground_truth, window_size=3, use_brightness=True, use_contrast=True, use_content=False, use_all=True):
		self.window_size = window_size
		self.ones_filter = torch.ones(size=(1, 1, window_size, window_size), dtype=torch.float32).to(DEVICE)

		prediction_mean_map = self.compute_patches_mean(prediction)
		ground_truth_mean_map = self.compute_patches_mean(ground_truth)

		prediction_std_map = self.compute_patches_std(prediction, prediction_mean_map)
		ground_truth_std_map = self.compute_patches_std(ground_truth, ground_truth_mean_map)

		joint_std_map = self.compute_joint_std(prediction, ground_truth, prediction_mean_map, ground_truth_mean_map)

		if not use_all:
			brightness_map = (2 * prediction_mean_map * ground_truth_mean_map + self.C1) / (prediction_mean_map ** 2 + ground_truth_mean_map ** 2 + self.C1)
			contrast_map = (2 * prediction_std_map * ground_truth_std_map + self.C2) / (prediction_std_map ** 2 + ground_truth_std_map ** 2 + self.C2)
			content_map = (joint_std_map ** 2 + self.C2) / (prediction_std_map * ground_truth_std_map + self.C2)
			SSIM_map = torch.ones(size=brightness_map.shape).to(DEVICE)
			if use_brightness:
				SSIM_map *= brightness_map
			if use_contrast:
				SSIM_map *= contrast_map
			if use_content:
				SSIM_map *= content_map
		else:
			#print("USE ALL")
			SSIM_map = ((2 * prediction_mean_map * ground_truth_mean_map + self.C1) * (2 * joint_std_map + self.C2)) / ((prediction_mean_map ** 2 + ground_truth_mean_map ** 2 + self.C1) * (prediction_std_map + ground_truth_std_map + self.C2))

		return torch.mean(1 - SSIM_map), 1 - SSIM_map

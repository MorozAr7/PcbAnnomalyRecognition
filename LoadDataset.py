import pandas as pd
import random
import numpy as np
from DataAugmentationGMSSSIM import transform_to_tensor, augmentation_noise
import cv2
from TrainConfig import *


class DiskAnomalyDataset(torch.utils.data.Dataset):
	def __init__(self, data_augmentation=None, use_multiscale=False):
		super(DiskAnomalyDataset, self).__init__()
		self.num_images = 97
		self.data_augmentation = data_augmentation
		self.use_multiscale = use_multiscale
		self.im_size = IMAGE_SIZE
		self.num_chunks = 4
		self.square_sizes = [2 ** i for i in range(4, 4 + NUM_MASK_SCALES)]
		self.grid_sizes = [self.im_size // i for i in self.square_sizes]
		self.grid_coords = list()

		for index in range(len(self.square_sizes)):
			self.grid_coords.append(self.get_grid_coords_multiscale(self.square_sizes[index], self.grid_sizes[index]))

	@staticmethod
	def get_grid_coords_multiscale(square_size, grid_size):
		coords_list = []

		for row in range(grid_size):
			for col in range(grid_size):
				coords_list.append([col * square_size, row * square_size])
		return coords_list

	def create_masks(self):
		masks_different_scales = np.array([])
		for scale in range(NUM_MASK_SCALES):
			if not self.use_multiscale:
				scale = random.randint(0, len(self.square_sizes)-1)

			grid_coords = self.grid_coords[scale]
			square_size = self.square_sizes[scale]
			random.shuffle(grid_coords)

			num_cells_per_chunk = len(grid_coords)//self.num_chunks

			chunks = [grid_coords[num_cells_per_chunk * index: num_cells_per_chunk * (index + 1)] for index in range(self.num_chunks)]

			masks_one_scale = np.array([])
			for index in range(self.num_chunks):
				mask = np.ones(shape=(1, self.im_size, self.im_size))
				chunk = chunks[index]
				for square_coord in chunk:
					x_coord = square_coord[0]
					y_coord = square_coord[1]
					mask[:, y_coord:y_coord + square_size, x_coord:x_coord + square_size] = 0
				if index > 0:
					masks_one_scale = np.concatenate([masks_one_scale, mask], axis=0)
				else:
					masks_one_scale = mask
			if scale == 0 or not self.use_multiscale:
				masks_different_scales = masks_one_scale
				if not self.use_multiscale:
					break
			else:
				masks_different_scales = np.concatenate([masks_different_scales, masks_one_scale], axis=0)
		return masks_different_scales

	@staticmethod
	def random_crop(image):
		size = random.randint(450, 500)
		x_coord = random.randint(0, 500 - size - 1)
		y_coord = random.randint(0, 500 - size - 1)

		return image[y_coord:y_coord + size, x_coord:x_coord + size]

	def __len__(self):
		return self.num_images

	def __getitem__(self, index):

		image = cv2.imread("./ImagesNew/" + f"pcb_{index}.png", 0)
		image = self.random_crop(image)
		if self.data_augmentation:
			transformed = self.data_augmentation(image=image)

			image = transformed["image"]
			image = self.random_crop(image)
			image_input = augmentation_noise(image=image)["image"]
			image_gt = image
		masks = self.create_masks()
  
		image_gt = cv2.resize(image_gt, (self.im_size, self.im_size))
		image_input = cv2.resize(image_input, (self.im_size, self.im_size))

		image_gt = transform_to_tensor(image=image_gt)["image"]
		image_input = transform_to_tensor(image=image_input)["image"]

		return image_input, image_gt, masks

import os
import random
import time
from torch.nn.functional import conv2d
import cv2
import numpy as np
import torch
from ModelsCNN import EdgeRestoreModel
import matplotlib
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision


class DiscQualityCheckApi:
	def __init__(self, DEVICE):
		self.model = EdgeRestoreModel()
		self.model_weights = "/Users/artemmoroz/Desktop/CIIRC_projects/EdgeRestorationInfill/DiscsQualityCheckModel1.pt"
		self.DEVICE = DEVICE
		#self.initialize_model()
		self.image_size = 256
		self.num_chunks = 4
		self.square_sizes = [2 ** i for i in range(4, 7)]
		self.grid_sizes = [self.image_size // i for i in self.square_sizes]
		self.num_grid_scales = len(self.grid_sizes)
		self.num_scales = 3
		self.image_size_multiscale = [256, 128, 64]
		self.grid_coords = list()
		self.K1 = 0.01
		self.K2 = 0.03
		self.C1 = (self.K1 * 1) ** 2
		self.C2 = (self.K2 * 1) ** 2
		self.dataset_mean = 0.5958
		self.num_images = 4
		self.masks = self.create_masks_()
		self.defect_score_threshold = 0.6

	@staticmethod
	def increase_contrast(image):
		transformation = A.Compose([A.RandomContrast(limit=(1, 1), p=1)])

		return transformation(image=image)["image"]

	def compute_patches_mean(self, image_tensor):
		convolved = conv2d(input=image_tensor, weight=self.ones_filter / (self.window_size ** 2), stride=1)
		return convolved

	def compute_patches_std(self, image_tensor, mean_map):
		convolved = conv2d(image_tensor ** 2, weight=self.ones_filter / (self.window_size ** 2 - 1), stride=1) - mean_map ** 2
		return convolved

	def compute_joint_std(self, in_painted_image_tensor, real_image_tensor, prediction_mean_map, ground_truth_mean_map):
		convolved = conv2d(input=in_painted_image_tensor * real_image_tensor, weight=self.ones_filter / (self.window_size ** 2 - 1), stride=1) - prediction_mean_map * ground_truth_mean_map
		return convolved

	def get_contrast_similarity_map(self, in_painted_image_tensor, real_image_tensor, window_size=5):
		self.window_size = window_size
		self.ones_filter = torch.ones(size=(1, 1, self.window_size, self.window_size), dtype=torch.float32).to(self.DEVICE)
		prediction_mean_map = self.compute_patches_mean(in_painted_image_tensor)
		ground_truth_mean_map = self.compute_patches_mean(real_image_tensor)

		prediction_std_map = self.compute_patches_std(in_painted_image_tensor, prediction_mean_map)
		ground_truth_std_map = self.compute_patches_std(real_image_tensor, ground_truth_mean_map)

		contrast_sim_map = (2 * prediction_std_map * ground_truth_std_map + self.C2) / (prediction_std_map ** 2 + ground_truth_std_map ** 2 + self.C2)
		contrast_sim_map = self.resize_tensor(contrast_sim_map, in_painted_image_tensor.shape[2])

		return 1 - contrast_sim_map, 1 - torch.mean(contrast_sim_map)

	def get_ssim_maps(self, in_painted_image_tensor, real_image_tensor, window_size=5):
		self.window_size = window_size
		self.ones_filter = torch.ones(size=(1, 1, self.window_size, self.window_size), dtype=torch.float32).to(self.DEVICE)
		prediction_mean_map = self.compute_patches_mean(in_painted_image_tensor)
		ground_truth_mean_map = self.compute_patches_mean(real_image_tensor)

		prediction_std_map = self.compute_patches_std(in_painted_image_tensor, prediction_mean_map)
		ground_truth_std_map = self.compute_patches_std(real_image_tensor, ground_truth_mean_map)

		joint_std_map = self.compute_joint_std(in_painted_image_tensor, real_image_tensor, prediction_mean_map, ground_truth_mean_map)

		brightness_sim = (2 * prediction_mean_map * ground_truth_mean_map + self.C2) / (prediction_mean_map ** 2 + ground_truth_mean_map ** 2 + self.C2)
		contrast_sim = (2 * torch.sqrt(prediction_std_map + 1e-4) * torch.sqrt(ground_truth_std_map + 1e-4) + self.C1) / (prediction_std_map + ground_truth_std_map + self.C1)
		content_sim = (joint_std_map + self.C1) / (torch.sqrt(prediction_std_map + 1e-4) * torch.sqrt(ground_truth_std_map + 1e-4) + self.C1)

		ssim_map = content_sim ** 4 * brightness_sim ** 1 * contrast_sim ** 1
		# ssim_map = content_sim * brightness_sim * contrast_sim
		ssim_map = self.resize_tensor(ssim_map, in_painted_image_tensor.shape[2])

		return 1 - ssim_map, torch.mean(1 - ssim_map)

	def create_masks_(self):
		masks_multiscale = np.array([])
		for (index, square_size) in enumerate(self.square_sizes):
			masks_one_scale = np.ones(shape=(1, 4, self.image_size, self.image_size))
			for index_width in range(0, self.grid_sizes[index], 2):
				for index_height in range(0, self.grid_sizes[index], 2):
					masks_one_scale[:, 0:1, index_width*square_size:(index_width + 1)*square_size, index_height*square_size:(index_height + 1)*square_size] = 0
					masks_one_scale[:, 1:2, (index_width + 1) * square_size:(index_width + 2) * square_size, index_height * square_size:(index_height + 1) * square_size] = 0
					masks_one_scale[:, 2:3, index_width * square_size:(index_width + 1) * square_size, (index_height + 1) * square_size:(index_height + 2) * square_size] = 0
					masks_one_scale[:, 3:4, (index_width + 1) * square_size:(index_width + 2) * square_size, (index_height + 1) * square_size:(index_height + 2) * square_size] = 0

			if index == 0:
				masks_multiscale = masks_one_scale
			else:
				masks_multiscale = np.concatenate([masks_multiscale, masks_one_scale], axis=0)

		return torch.tensor(masks_multiscale, dtype=torch.float32).to(self.DEVICE).reshape(-1, 1, self.image_size, self.image_size)#.unsqueeze(0)

	@staticmethod
	def resize_tensor(image_tensor, image_size):
		resize_transform = torch.nn.Sequential(torchvision.transforms.Resize((image_size, image_size)))
		return resize_transform(image_tensor)

	@staticmethod
	def convolve(image, kernel):
		convolved = torch.nn.functional.conv2d(image, weight=kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=1)
		return convolved

	def compute_gradient_map(self, image):
		filter_1 = torch.tensor([[0, 0, 0, 0, 0], [1, 3, 8, 3, 1], [0, 0, 0, 0, 0], [-1, -3, -8, -3, -1], [0, 0, 0, 0, 0]]).to(self.DEVICE) / 1
		filter_2 = torch.tensor([[0, 1, 0, -1, 0], [0, 3, 0, -3, 0], [0, 8, 0, -8, 0], [0, 3, 0, -3, 0], [0, 1, 0, -1, 0]]).to(self.DEVICE) / 1
		filter_3 = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 3, 8, 0], [-1, -3, 0, 3, 1], [0, -8, -3, 0, 0], [0, 0, -1, 0, 0]]).to(self.DEVICE) / 1
		filter_4 = torch.tensor([[0, 0, 1, 0, 0], [0, 8, 3, 0, 0], [1, 3, 0, -3, -1], [0, 0, -3, -8, 0], [0, 0, -1, 0, 0]]).to(self.DEVICE) / 1

		image_1_gradient = torch.abs(self.convolve(image, filter_1).unsqueeze(0))
		image_2_gradient = torch.abs(self.convolve(image, filter_2).unsqueeze(0))
		image_3_gradient = torch.abs(self.convolve(image, filter_3).unsqueeze(0))
		image_4_gradient = torch.abs(self.convolve(image, filter_4).unsqueeze(0))
		concatenated = torch.cat([image_1_gradient, image_2_gradient, image_3_gradient, image_4_gradient], dim=0)
		image_grad_map = torch.max(concatenated, dim=0, keepdim=False)[0]

		return image_grad_map

	def get_multiscale_representation(self, images_tensor, is_real=True):
		multiscale_images = []
		for scale in range(self.num_scales):
			if is_real:
				multiscale_images.append(images_tensor)
				images_tensor = torch.nn.functional.avg_pool2d(images_tensor, kernel_size=2, stride=2)
			else:
				multiscale_images.append(images_tensor.reshape(self.num_images * self.num_grid_scales, 1, self.image_size_multiscale[scale], self.image_size_multiscale[scale]))
				images_tensor = torch.nn.functional.avg_pool2d(images_tensor.reshape(self.num_images * self.num_grid_scales, 1, self.image_size_multiscale[scale], self.image_size_multiscale[scale]),
				                                               kernel_size=2,
				                                               stride=2)
		return multiscale_images

	def compute_multiscale_gradient_maps(self, multiscale_representation_list):
		multiscale_grad_maps = list()
		for scale in range(len(multiscale_representation_list)):
			grad_maps = self.compute_gradient_map(multiscale_representation_list[scale])
			multiscale_grad_maps.append(grad_maps)

		return multiscale_grad_maps

	def combine_in_painted_chunks(self, in_painted_images, masks):
		in_painted_images = in_painted_images.reshape(self.num_images, self.num_grid_scales, self.num_chunks, 1, self.image_size, self.image_size)
		masks = masks.reshape(self.num_images, self.num_grid_scales, self.num_chunks, 1, self.image_size, self.image_size)
		combined_chunks = torch.sum(in_painted_images * (1 - masks), dim=2, keepdim=False)

		return combined_chunks

	@staticmethod
	def scale_min_max(image_tensor):
		max_values = torch.max(torch.max(image_tensor, dim=2)[0], dim=2)[0].reshape(image_tensor.shape[0], image_tensor.shape[1], 1, 1)
		min_values = torch.min(torch.min(image_tensor, dim=2)[0], dim=2)[0].reshape(image_tensor.shape[0], image_tensor.shape[1], 1, 1)
		return (image_tensor - min_values) / (max_values - min_values)

	@staticmethod
	def get_augmented_residual_map(in_painted_image_tensor, real_image_tensor):
		return (torch.square(in_painted_image_tensor - real_image_tensor) + 0.001) / (torch.square(in_painted_image_tensor) + torch.square(real_image_tensor) + 0.001)

	@staticmethod
	def visualize_torch_tensor(tensor):
		tensor = tensor.permute(0, 2, 3, 1)
		for index in range(tensor.shape[0]):
			cv2.imshow("window", tensor[index, ...].detach().cpu().numpy())
			cv2.waitKey(0)

	@staticmethod
	def transform_to_tensor(image):
		transformation = A.Compose([A.Normalize(0, 1), ToTensorV2()])
		return transformation(image=image)["image"]

	@staticmethod
	def get_threshold_segmentation(real_image_tensor, in_painted_image_tensor):
		tensor_to_threshold = torch.cat([real_image_tensor, in_painted_image_tensor], dim=0) < 0.55
		tensor_to_threshold = torch.tensor(tensor_to_threshold, dtype=torch.float32)
		return torch.sum(tensor_to_threshold, dim=0, keepdim=True).clamp(0, 1)

	def initialize_model(self):
		self.model.load_state_dict(torch.load(self.model_weights, map_location="cpu"))
		self.model = self.model.eval().to(self.DEVICE)

	@staticmethod
	def get_pixel_wise_max(maps_tensor):
		return torch.max(maps_tensor, dim=0, keepdim=True)[0]

	def get_multiscale_similarity(self, real_image_multiscale_maps, in_painted_image_multiscale_maps, similarity_function):
		multi_window_maps = list()
		for window_size in [5, 7, 9]:
			multiscale_sim_maps = list()
			for index in range(len(real_image_multiscale_maps)):
				similarity_maps, _ = similarity_function(in_painted_image_multiscale_maps[index], real_image_multiscale_maps[index].repeat(1, 3, 1, 1).
				                                         reshape(self.num_images * self.num_grid_scales, 1,
				                                                 real_image_multiscale_maps[index].shape[-1],
				                                                 real_image_multiscale_maps[index].shape[-1]),
				                                         window_size)
				similarity_maps = similarity_maps.reshape(self.num_images, self.num_grid_scales, 1, similarity_maps.shape[-1], similarity_maps.shape[-1])
				similarity_maps = (similarity_maps[:, 0, ...] * similarity_maps[:, 1, ...] + \
				                  similarity_maps[:, 1, ...] * similarity_maps[:, 2, ...])
				#print(similarity_maps.shape)
				#self.visualize_torch_tensor(similarity_maps[0, ...])
				#similarity_maps = (similarity_maps[:, 0, ...] + similarity_maps[:, 1, ...] + similarity_maps[:, 2, ...]) / 3
				multiscale_sim_maps.append(similarity_maps)
			multi_window_maps.append(multiscale_sim_maps)
		multiscale_sim_maps = list()
		for scale_num in range(self.num_scales):
			multiscale_sim_maps.append((multi_window_maps[0][scale_num] + multi_window_maps[1][scale_num] + multi_window_maps[2][scale_num]) / 3)# + multi_window_maps[1][scale_num] *
		# multi_window_maps[2][scale_num])/2) #*
		# multi_window_maps[2][scale_num])

		return multiscale_sim_maps

	@staticmethod
	def get_multiscale_l1_loss(real_image_multiscale_maps, in_painted_image_multiscale_maps):
		l1_loss = torch.nn.L1Loss()
		multiscale_l1_loss = 0
		for index in range(len(real_image_multiscale_maps)):
			multiscale_l1_loss += l1_loss(in_painted_image_multiscale_maps[index], real_image_multiscale_maps[index])
		return multiscale_l1_loss

	@staticmethod
	def get_multiscale_structural_sim_loss(real_image_multiscale_maps, in_painted_image_multiscale_maps, sim_function, window_size=5):
		multiscale_sim_loss = 0
		biggest_scale_loss = 0
		for index in range(len(real_image_multiscale_maps)):
			multiscale_sim_loss += sim_function(in_painted_image_multiscale_maps[index], real_image_multiscale_maps[index], window_size)[1]
			if index == 0:
				biggest_scale_loss = multiscale_sim_loss.item()
		return multiscale_sim_loss, biggest_scale_loss

	def combine_multiscale_maps(self, maps):
		return self.resize_tensor(maps[0], self.image_size) * self.resize_tensor(maps[1], self.image_size) + self.resize_tensor(maps[1], self.image_size) * self.resize_tensor(maps[2],
		                                                                                                                                                                        self.image_size)

	def combine_maps_sum(self, maps):
		return (self.resize_tensor(maps[0], self.image_size) + self.resize_tensor(maps[1], self.image_size) + self.resize_tensor(maps[2], self.image_size))

	def get_rgb_heatmap(self, defect_map, in_painted_images_combined, real_image_tensor):
		batch_heatmap = torch.tensor([]).to(self.DEVICE)
		defect_scores = []
		for image_index in range(self.num_images):
			numpy_defect_map = defect_map[image_index][0].detach().cpu().numpy()
			segmentation = self.get_threshold_segmentation(real_image_tensor[image_index:image_index + 1],
			                                                                      in_painted_images_combined[image_index])[0][0].detach().cpu().numpy()
			numpy_defect_map = numpy_defect_map * segmentation
			numpy_defect_map = numpy_defect_map#(numpy_defect_map - np.mean(numpy_defect_map[np.array(segmentation, dtype=bool)])) / np.std(numpy_defect_map[np.array(segmentation, dtype=bool)])
			mean_value_defect_map = np.max(numpy_defect_map)
			defect_scores.append(mean_value_defect_map)
			colormap = matplotlib.cm.get_cmap('coolwarm')
			data = colormap(numpy_defect_map)[..., 0:3]
			heatmap = torch.tensor(cv2.cvtColor(np.array(data * 255, dtype=np.uint8), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).to(self.DEVICE)
			if image_index > 0:
				batch_heatmap = torch.cat([batch_heatmap, heatmap], dim=0)
			else:
				batch_heatmap = heatmap
		return batch_heatmap / 255, defect_scores

	def visualize_result_batch(self, images_batch, heatmap_batch):
		concat = torch.cat([images_batch.repeat(1, 3, 1, 1).reshape(1, 3, self.image_size, self.image_size * self.num_images),
		                            heatmap_batch.reshape(1, 3, self.image_size, self.image_size * self.num_images)], dim=3).permute(0, 2, 3, 1).detach().cpu().numpy()[0]
		#Qcv2.imshow("result", concat)
		#cv2.waitKey(0)

	@staticmethod
	def save_result_image(heatmaps, images):
		images = torch.cat([images, images, images], dim=1)
		num_images = heatmaps.shape[0]
		image_size = heatmaps.shape[-1]
		empty_tensor = torch.zeros(size=(3, image_size * num_images // 2, image_size * num_images // 2))
		for i in range(images.shape[0]):
			index_x = i % 4
			if i < 4:
				index_y = 0
			else:
				index_y = 1
			empty_tensor[:, index_y*image_size:(index_y + 1) * image_size, index_x*image_size:(index_x + 1) * image_size] = images[i, ...]
			empty_tensor[:, (index_y + 2) * image_size:(index_y + 3) * image_size, index_x * image_size:(index_x + 1) * image_size] = heatmaps[i, ...]
		numpy_image = empty_tensor.permute(1, 2, 0).detach().cpu().numpy()
		cv2.imwrite("DiscQualityCheckImage.png", numpy_image * 255)
		cv2.imshow("results", numpy_image)
		cv2.waitKey(0)

	def get_quality_check_results(self, images):
		self.num_images = len(images)
		heatmap_batch, images_batch, defect_scores = self.process_images(images)
		self.save_result_image(heatmap_batch, images_batch)
		predictions = []
		for i in range(len(defect_scores)):
			if defect_scores[i] > self.defect_score_threshold or np.isnan(defect_scores[i]):
				predictions.append(False)
			else:
				predictions.append(True)
		return defect_scores

	def process_images(self, images):
		self.masks = self.create_masks_().repeat(self.num_images, 1, 1, 1)
		since = time.time()
		images_batch = torch.tensor([]).to(self.DEVICE)
		images_batch_repeated = torch.tensor([]).to(self.DEVICE)
		for image_index in range(len(images)):
			image = cv2.resize(images[image_index], (self.image_size, self.image_size))
			real_image_tensor = self.transform_to_tensor(image).to(self.DEVICE).unsqueeze(0)#.repeat(12, 1, 1, 1)
			if image_index > 0:
				images_batch_repeated = torch.cat([images_batch_repeated, real_image_tensor.repeat(self.num_chunks * self.num_grid_scales, 1, 1, 1)], dim=0)
				images_batch = torch.cat([images_batch, real_image_tensor], dim=0)
			else:
				images_batch_repeated = real_image_tensor.repeat(self.num_chunks * self.num_grid_scales, 1, 1, 1)
				images_batch = real_image_tensor

		with torch.no_grad():

			in_painted_images = self.model(images_batch_repeated * self.masks + (1 - self.masks) * self.dataset_mean, self.masks)

			in_painted_images_combined = self.combine_in_painted_chunks(in_painted_images, self.masks)

			real_image_multiscale = self.get_multiscale_representation(images_batch)
			in_painted_image_multiscale = self.get_multiscale_representation(in_painted_images_combined, is_real=False)

			real_image_multiscale_gradients = self.compute_multiscale_gradient_maps(real_image_multiscale)
			in_painted_image_multiscale_gradients = self.compute_multiscale_gradient_maps(in_painted_image_multiscale)

			multiscale_grad_sim_maps = self.get_multiscale_similarity(real_image_multiscale_gradients, in_painted_image_multiscale_gradients, self.get_contrast_similarity_map)
			multiscale_image_sim_maps = self.get_multiscale_similarity(real_image_multiscale, in_painted_image_multiscale, self.get_ssim_maps)

			gradient_similarity = self.combine_multiscale_maps(multiscale_grad_sim_maps)
			image_similarity = self.combine_multiscale_maps(multiscale_image_sim_maps)

			total_defect_map = gradient_similarity + image_similarity

			heatmap_batch, defect_scores = self.get_rgb_heatmap(total_defect_map, in_painted_images_combined, images_batch)
			print(defect_scores)
			#print("TIME", time.time() - since)

			"""self.visualize_torch_tensor(torch.concatenate([images_batch.repeat(1, 3, 1, 1), heatmap_batch, in_painted_images_combined[:, 0, ...].repeat(1, 3, 1, 1), in_painted_images_combined[:, 1,
			                                                                                                                                                         ...].repeat(1, 3, 1, 1),
			                                               in_painted_images_combined[:, 2, ...].repeat(1, 3, 1, 1)], dim=3))"""
			return heatmap_batch, images_batch, defect_scores


if __name__ == "__main__":
	api = DiscQualityCheckApi("mps")

	CORRECT = True

	path_correct = "/Users/artemmoroz/Desktop/CIIRC_projects/EdgeRestorationInfill/NewDiscsValidLowResolution/"
	path_incorrect = "/Users/artemmoroz/Desktop/CIIRC_projects/EdgeRestorationInfill/DefectedHighResolution/"

	images_correct = os.listdir(path_correct)
	images_incorrect = os.listdir(path_incorrect)

	images_list = images_correct if CORRECT else images_incorrect
	images_path = path_correct if CORRECT else path_incorrect
	print(len(images_list))
	for image_name in images_list:
		image1 = cv2.imread(path_correct + image_name, 0)
		image2 = cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/EdgeRestorationInfill/ServerData/disc_qc_1.png", 0)
		image3 = cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/EdgeRestorationInfill/ServerData/disc_qc_2.png", 0)
		image4 = cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/EdgeRestorationInfill/ServerData/disc_qc_3.png", 0)
		image5 = cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/EdgeRestorationInfill/ServerData/disc_qc_4.png", 0)
		image6 = cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/EdgeRestorationInfill/ServerData/disc_qc_5.png", 0)
		image7 = cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/EdgeRestorationInfill/ServerData/disc_qc_6.png", 0)
		image8 = cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/EdgeRestorationInfill/ServerData/disc_qc_7.png", 0)
		images = [image1, image2, image3, image4, image5, image6, image7, image8]

		qc_scores = api.get_quality_check_results(images)
		#if qc_scores[0] > 0.25:
			#print(qc_scores[0], image_name)

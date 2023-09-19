from ModelsCNN import EdgeRestoreModel
from LoadDataset import DiskAnomalyDataset
from DataAugmentationGMSSSIM import augmentation_training
from torch.utils.data import DataLoader
from LossFunctions import SSIMLoss
import time
import torch.nn as nn
import random
import cv2
from TrainConfig import *
import numpy as np

from DiscQualityCheckAPI import DiscQualityCheckApi
import os
api = DiscQualityCheckApi(DEVICE)


def init_weights(m):
	if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
		torch.nn.init.xavier_uniform_(m.weight)
	elif type(m) in [nn.BatchNorm2d]:
		torch.nn.init.normal_(m.weight.data, 1.0, 2.0)
		torch.nn.init.constant_(m.bias.data, 0)


def change_learning_rate(optim, epoch):
	epochs_to_change = list(range(250, 5000, 250))
	if epoch in epochs_to_change:
		optim.param_groups[0]["lr"] /= 1.5


def one_epoch(models, optimizers, dataloader, is_training=True):
	models.train() if is_training else models.eval()
	global api
	epoch_loss_l1 = 0
	epoch_loss_ssim = 0
	epoch_loss_l1_grad = 0
	epoch_loss_ssim_grad = 0

	if is_training:
		for index, (gt_image, masks) in enumerate(dataloader):
			print("BATCH TRAINING: ", index)
			optimizers.zero_grad()
			gt_image = gt_image.to(DEVICE)

			masks = torch.tensor(masks, dtype=torch.float32).to(DEVICE)
			restored_images_per_masks = torch.tensor([]).to(DEVICE)

			for mask_idx in range(masks.shape[1]):
				mask = masks[:, mask_idx:mask_idx + 1, ...]
				restored_image = models(gt_image * mask + (1 - mask) * torch.mean(gt_image), mask)
				restored_images_per_masks = torch.cat([restored_images_per_masks, restored_image], dim=1)

			restored_image = torch.sum(restored_images_per_masks * (1 - masks), dim=1, keepdim=True)
			"""visualize = restored_image.permute(0, 2, 3, 1).detach().cpu().numpy()
			visualize_gt = gt_image.permute(0, 2, 3, 1).detach().cpu().numpy()
			for index in range(gt_image.shape[0]):
				cv2.imshow("img", np.vstack([visualize[index, ...], visualize_gt[index, ...]]))
				cv2.waitKey(0)"""
			real_image_multiscale = api.get_multiscale_representation(gt_image)
			in_painted_image_multiscale = api.get_multiscale_representation(restored_image)

			l1_loss = api.get_multiscale_l1_loss(real_image_multiscale, in_painted_image_multiscale)
			window_size = [5, 7, 9][random.randint(0, 2)]

			ssim_loss, biggest_scale_image = api.get_multiscale_structural_sim_loss(real_image_multiscale, in_painted_image_multiscale, api.get_ssim_maps, window_size=window_size)

			pred_grads = api.compute_multiscale_gradient_maps(real_image_multiscale)
			gt_grads = api.compute_multiscale_gradient_maps(in_painted_image_multiscale)
			grad_ssim_loss, biggest_scale_grad = api.get_multiscale_structural_sim_loss(pred_grads, gt_grads, api.get_contrast_similarity_map, window_size=window_size)
			grad_l1_loss = api.get_multiscale_l1_loss(pred_grads, gt_grads)

			total_loss = LossCoefficients['L1'] * l1_loss + LossCoefficients['SSIM'] * ssim_loss + LossCoefficients["GradL1"] * grad_l1_loss + LossCoefficients["GradSSIM"] * grad_ssim_loss

			total_loss.backward()
			optimizers.step()

			torch.cuda.empty_cache()

			epoch_loss_l1 += l1_loss.item() * gt_image.shape[0]
			epoch_loss_ssim += biggest_scale_image * gt_image.shape[0]
			epoch_loss_l1_grad += grad_l1_loss.item() * gt_image.shape[0]
			epoch_loss_ssim_grad += biggest_scale_grad * gt_image.shape[0]

		return epoch_loss_l1 / (len(train_dataset)), epoch_loss_ssim / len(train_dataset), epoch_loss_l1_grad / len(train_dataset), epoch_loss_ssim_grad / len(train_dataset)
	else:
		with torch.no_grad():
			for index, (gt_image, masks) in enumerate(dataloader):
				print("BATCH VALIDATION: ", index)
				epoch_loss_l1 += 0#loss_res.item()
				torch.cuda.empty_cache()
			return epoch_loss_l1 / (len(validation_dataset)), epoch_loss_ssim / len(validation_dataset), epoch_loss_l1_grad / len(validation_dataset), epoch_loss_ssim_grad / len(validation_dataset)


def main(model, optimizer,  training_dataloader, validation_dataloader):
	for epoch in range(1, 5001):
		since = time.time()
		change_learning_rate(optimizer, epoch)

		training_loss_l1, train_loss_ssim, train_loss_l1_grad, train_loss_ssim_grad = one_epoch(model,
										                                    optimizer,
										                                    training_dataloader,
										                                    is_training=True,
										                                    )


		print("Epoch: {}, Train l1: {}, Train SSIM: {}, Train l1 grad: {} Train SSIM grad: {}".format(epoch, training_loss_l1, train_loss_ssim, train_loss_l1_grad, train_loss_ssim_grad))
		print("EPOCH RUNTIME", time.time() - since)

		if epoch % 100 == 0:
			print("SAVING MODEL...")
			torch.save(model.state_dict(), "{}.pt".format("{}".format(MODEL_NAME)))
			print("MODEL WAS SUCCESSFULLY SAVED!")


if __name__ == "__main__":
	model = EdgeRestoreModel().to(DEVICE).apply(init_weights)
	#model.load_state_dict(torch.load("/Users/artemmoroz/Desktop/CIIRC_projects/PcbAnnomalyRecognition/Model1.pt", map_location="cpu"))
	optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())

	train_dataset = DiskAnomalyDataset(data_augmentation=augmentation_training if APPLY_AUGMENTATION else None, use_multiscale=False)
	validation_dataset = DiskAnomalyDataset(data_augmentation=augmentation_training if APPLY_AUGMENTATION else None, use_multiscale=False)

	training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

	main(model, optimizer, training_dataloader, validation_dataloader)


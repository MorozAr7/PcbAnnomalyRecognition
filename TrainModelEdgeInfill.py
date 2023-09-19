from DiscQualityControlModelsCNN import EdgeRestoreModel, EdgeFillModel
from LoadDataset import DiskAnomalyDataset
from DataAugmentationGMSSSIM import augmentation_training
from torch.utils.data import DataLoader
from LossFunctions import L2RestorationLoss, SSIMLoss
import time
from utils import *
from TrainConfig import *
import cv2


def one_epoch(models, optimizers, losses, dataloader, is_training=True):
	models[0].train() if is_training else models[0].eval()
	models[1].train() if is_training else models[1].eval()

	epoch_loss_l1_value = 0
	epoch_loss_ssim_value = 0
	epoch_loss_l1_grad = 0
	epoch_loss_ssim_grad = 0
	if is_training:
		for index, (edges, reference_image, masks) in enumerate(dataloader):
			print("BATCH TRAINING: ", index)

			optimizers[0].zero_grad()
			edges = edges.to(DEVICE)
			reference_image = reference_image.to(DEVICE)
			masks = torch.tensor(masks, dtype=torch.float32).to(DEVICE)

			restored_edges_all_scales = torch.tensor([]).to(DEVICE)
			with torch.no_grad():
				for scale in range(0, 3):
					restored_edges_per_scale = torch.tensor([]).to(DEVICE)
					for mask_idx in range(scale * 4, (scale + 1) * 4):
						mask = masks[:, mask_idx:mask_idx + 1, ...]
						restored_edges = models[0](edges * mask + (1 - mask) * torch.mean(edges), mask)
						restored_edges_per_scale = torch.cat([restored_edges_per_scale, restored_edges], dim=1)

					restored_edges = torch.sum(restored_edges_per_scale * (1 - masks[:, scale * 4:(scale + 1) * 4, ...]), dim=1, keepdim=True)
					restored_edges_all_scales = torch.cat([restored_edges_all_scales, restored_edges], dim=1)

			merged_edges = merge_edges_scales(restored_edges_all_scales)
			inpainted_image = models[1](merged_edges)

			l1_loss = losses[0](inpainted_image, reference_image)
			ssim_loss, _ = losses[1](inpainted_image, reference_image, window_size=9, use_all=True)

			prediction_multiscale_grads = get_sobel_grad_map(inpainted_image)
			ground_truth_multiscale_grads = get_sobel_grad_map(reference_image)
			grad_ssim_loss, _ = losses[1](prediction_multiscale_grads, ground_truth_multiscale_grads, window_size=9, use_all=True)
			grad_l1_loss = losses[0](prediction_multiscale_grads, ground_truth_multiscale_grads)
			#grad_l1_loss = get_multiscale_grad_l1_loss(prediction_multiscale_grads, ground_truth_multiscale_grads, losses[0])
			#grad_ssim_loss, _ = get_multiscale_grad_ssim_loss(prediction_multiscale_grads, ground_truth_multiscale_grads, losses[1], window_size=5)

			total_loss = (LossCoefficients["L1"] * l1_loss + LossCoefficients["SSIM"] * ssim_loss + LossCoefficients["GradL1"] * grad_l1_loss + LossCoefficients["GradSSIM"] * grad_ssim_loss)

			total_loss.backward()
			optimizers[0].step()

			"""features_numpy = torch.cat([edges, restored_edges_all_scales[:, 0:1, ...], restored_edges_all_scales[:, 1:2, ...], restored_edges_all_scales[:, 2:3, ...],
			                           merged_edges,
			                            reference_image, inpainted_image],
			                           dim=3).permute(0,
			                                                                                                                                                                                 2, 3, 1).detach().cpu().numpy()
			for i in range(4):
				cv2.imshow("featuires", features_numpy[i, ...])
				cv2.waitKey(0)"""
			torch.cuda.empty_cache()

			epoch_loss_l1_value += l1_loss.item() * edges.shape[0]
			epoch_loss_ssim_value += ssim_loss.item() * edges.shape[0]
			epoch_loss_l1_grad += grad_l1_loss.item() * edges.shape[0]
			epoch_loss_ssim_grad += grad_ssim_loss.item() * edges.shape[0]
		return epoch_loss_l1_value / (len(train_dataset)), epoch_loss_ssim_value / len(train_dataset), epoch_loss_l1_grad / len(train_dataset), epoch_loss_ssim_grad / len(train_dataset)
	else:
		with torch.no_grad():
			for index, (edges, _, masks) in enumerate(dataloader):
				print("BATCH VALIDATION: ", index)
				#epoch_loss_l1_value += 0#loss_res.item()
				#torch.cuda.empty_cache()
			return epoch_loss_l1_value / (len(validation_dataset)), epoch_loss_ssim_value / len(validation_dataset), epoch_loss_l1_grad / len(validation_dataset), epoch_loss_ssim_grad / len(train_dataset)


def main(models, optimizers, losses,  training_dataloader, validation_dataloader):
	for epoch in range(1, 5001):
		since = time.time()
		change_learning_rate(optimizers, epoch)

		training_loss_l1, train_loss_ssim, train_loss_l1_grad, train_loss_ssim_grad = one_epoch(models,
		                                    optimizers,
		                                    losses,
		                                    training_dataloader,
		                                    is_training=True)
		_, _, _, _ = one_epoch(models,
		                                      optimizers,
		                                      losses,
		                                      validation_dataloader,
		                                      is_training=False)

		print("Epoch: {}, Train L1: {}, Train SSIM: {}, Train L1 grad: {} Train SSIM grad: {}".format(epoch, training_loss_l1, train_loss_ssim, train_loss_l1_grad, train_loss_ssim_grad))
		print("EPOCH RUNTIME", time.time() - since)

		if epoch % 100 == 0:
			print("SAVING MODEL...")
			torch.save(models[1].state_dict(), "{}.pt".format("{}".format(MODEL_NAME)))
			print("MODEL WAS SUCCESSFULLY SAVED!")


if __name__ == "__main__":
	model_edge_restore = EdgeRestoreModel().to(DEVICE)
	model_edge_infill = EdgeFillModel().to(DEVICE).apply(init_weights)

	model_edge_restore.load_state_dict(torch.load("./LowerResLightModel2.pt", map_location="cpu"))
	#model_edge_infill.load_state_dict(torch.load("./ModelEdgeFill3.pt", map_location="cpu"))
	optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model_edge_infill.parameters())

	loss_ssim = SSIMLoss()
	loss_l1 = torch.nn.L1Loss(reduction="mean")

	train_dataset = DiskAnomalyDataset(csv_file="training_dataset.csv", data_augmentation=augmentation_training if APPLY_AUGMENTATION else None, use_multiscale=True)
	validation_dataset = DiskAnomalyDataset(csv_file="validation_dataset.csv", data_augmentation=augmentation_training if APPLY_AUGMENTATION else None, use_multiscale=True)

	training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

	models = [model_edge_restore, model_edge_infill]
	optimizers = [optimizer]
	losses = [loss_l1, loss_ssim]

	main(models, optimizers, losses, training_dataloader, validation_dataloader)
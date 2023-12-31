import cv2
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch
from TrainConfig import DEVICE


class TransposeConvBnReLU(nn.Module):
	def __init__(self, in_channels, out_channels, kernel=3, stride=2, in_padding=(1, 1), out_padding=(1, 1), apply_bn=True, apply_relu=True):
		super(TransposeConvBnReLU, self).__init__()
		self.TransposeConv = nn.ConvTranspose2d(in_channels=in_channels,
		                                        out_channels=out_channels,
		                                        kernel_size=kernel,
		                                        stride=stride,
		                                        padding=in_padding,
		                                        output_padding=out_padding,
		                                        bias=not apply_bn)
		self.BN = nn.BatchNorm2d(num_features=out_channels)
		self.ReLU = nn.SiLU(inplace=True)

		self.apply_bn = apply_bn
		self.apply_relu = apply_relu

	def forward(self, x):

		if self.apply_bn and self.apply_relu:
			return self.ReLU(self.BN(self.TransposeConv(x)))
		elif self.apply_relu and not self.apply_bn:
			return self.ReLU(self.TransposeConv(x))
		elif self.apply_bn and not self.apply_relu:
			return self.BN(self.TransposeConv(x))
		else:
			return self.TransposeConv(x)


class ConvBnReLU(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation_rate=1, apply_bn=True, apply_activation=True, apply_bias=False):
		super(ConvBnReLU, self).__init__()

		self.apply_relu = apply_activation
		self.apply_bn = apply_bn
		self.Conv = nn.Conv2d(in_channels=in_channels,
		                      out_channels=out_channels,
		                      kernel_size=kernel_size,
		                      stride=stride,
		                      padding=dilation_rate if dilation_rate > 1 else padding,
		                      groups=groups,
		                      bias=apply_bias,
		                      dilation=dilation_rate)
		if apply_bn:
			self.BN = nn.BatchNorm2d(num_features=out_channels)
		if apply_activation:
			self.ReLU = nn.SiLU(inplace=True)

	def forward(self, x):

		if self.apply_bn and self.apply_relu:
			return self.ReLU(self.BN(self.Conv(x)))
		elif not self.apply_bn and self.apply_relu:
			return self.ReLU(self.Conv(x))
		elif self.apply_bn and not self.apply_relu:
			return self.BN(self.Conv(x))
		elif not self.apply_bn and not self.apply_relu:
			return self.Conv(x)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, apply_activ=True):
		super(ResidualBlock, self).__init__()
		self.apply_activ = apply_activ
  
		self.Conv1 = ConvBnReLU(in_channels=in_channels, out_channels=out_channels)
		self.Conv2 = ConvBnReLU(in_channels=in_channels, out_channels=out_channels, apply_activation=False)
  
		if self.apply_activ:
			self.Activation = nn.SiLU(inplace=True)

	def forward(self, x):
		residual = x
		x = self.Conv1(x)
		x = self.Conv2(x)
		if self.apply_activ:
			return self.Activation(x + residual)
		else:
			return x + residual


class EdgeRestoreModel(nn.Module):
	def __init__(self):
		super(EdgeRestoreModel, self).__init__()

		channels = [1, 64, 128, 256, 512, 512]
		self.ConvLayer0 = ConvBnReLU(in_channels=channels[0], out_channels=channels[1])

		self.Downscale1 = ConvBnReLU(in_channels=channels[1], out_channels=channels[2], stride=2)

		self.ConvLayer1 = ConvBnReLU(in_channels=channels[2], out_channels=channels[2])

		self.Downscale2 = ConvBnReLU(in_channels=channels[2], out_channels=channels[3], stride=2)

		self.ConvLayer2 = ConvBnReLU(in_channels=channels[3], out_channels=channels[3])

		self.Downscale3 = ConvBnReLU(in_channels=channels[3], out_channels=channels[4], stride=2)

		self.ConvLayer3 = ConvBnReLU(in_channels=channels[4], out_channels=channels[4])

		self.Downscale4 = ConvBnReLU(in_channels=channels[4], out_channels=channels[5], stride=2)

		self.ConvLayer4 = ConvBnReLU(in_channels=channels[5], out_channels=channels[5], dilation_rate=2)

		self.ConvLayer5 = ConvBnReLU(in_channels=channels[5], out_channels=channels[5], dilation_rate=4)

		self.Upscale1 = TransposeConvBnReLU(in_channels=channels[5], out_channels=channels[4])

		#self.ConvLayer6 = ConvBnReLU(in_channels=channels[4] * 2, out_channels=channels[4])
		self.ResLayer1 = ResidualBlock(in_channels=channels[4], out_channels=channels[4])
		self.ResLayer2 = ResidualBlock(in_channels=channels[4], out_channels=channels[4])
  
		self.Upscale2 = TransposeConvBnReLU(in_channels=channels[4], out_channels=channels[3])

		#self.ConvLayer7 = ConvBnReLU(in_channels=channels[3] * 2, out_channels=channels[3])

		self.ResLayer3 = ResidualBlock(in_channels=channels[3], out_channels=channels[3])
		self.ResLayer4 = ResidualBlock(in_channels=channels[3], out_channels=channels[3])
  
		self.Upscale3 = TransposeConvBnReLU(in_channels=channels[3], out_channels=channels[2])

		#self.ConvLayer8 = ConvBnReLU(in_channels=channels[2] * 2, out_channels=channels[2])

		self.ResLayer5 = ResidualBlock(in_channels=channels[2], out_channels=channels[2])
		self.ResLayer6 = ResidualBlock(in_channels=channels[2], out_channels=channels[2])
  
		self.Upscale4 = TransposeConvBnReLU(in_channels=channels[2], out_channels=channels[1])

		self.OutConv1 = ConvBnReLU(in_channels=channels[1], out_channels=channels[1])

		self.OutConv2 = ConvBnReLU(in_channels=channels[1], out_channels=channels[0], apply_bn=False, apply_activation=False, apply_bias=False)

		self.Sigmoid = nn.Sigmoid()

	@staticmethod
	def update_partial_conv_binary_mask(mask, kernel=3, stride=1, dilation=1):
		mask = torch.tensor(mask, dtype=torch.float32)
		filter = torch.ones(size=(1, 1, kernel, kernel), dtype=torch.float32).to(DEVICE)

		convolved = f.conv2d(input=mask, weight=filter, stride=stride, dilation=dilation, padding=dilation, bias=None)
		new_mask = torch.tensor((convolved > 0), dtype=torch.uint8)

		return new_mask, (torch.tensor(new_mask.shape[-2] * new_mask.shape[-1]) / torch.sum(new_mask, dim=[2, 3])).reshape(-1, 1, 1, 1)

	def forward(self, x, mask):

		skip0 = self.ConvLayer0(x)

		mask, rescale_factor = self.update_partial_conv_binary_mask(mask)
		skip0 = skip0 * mask * rescale_factor
		x = self.Downscale1(skip0)

		mask, rescale_factor = self.update_partial_conv_binary_mask(mask, stride=2)

		skip1 = self.ConvLayer1(x * mask * rescale_factor)

		mask, rescale_factor = self.update_partial_conv_binary_mask(mask)
		skip1 = skip1 * mask * rescale_factor
		x = self.Downscale2(skip1)

		mask, rescale_factor = self.update_partial_conv_binary_mask(mask, stride=2)

		skip2 = self.ConvLayer2(x * mask * rescale_factor)

		mask, rescale_factor = self.update_partial_conv_binary_mask(mask)
		skip2 = skip2 * mask * rescale_factor

		x = self.Downscale3(skip2)

		mask, rescale_factor = self.update_partial_conv_binary_mask(mask, stride=2)

		skip3 = self.ConvLayer3(x * mask * rescale_factor)

		mask, rescale_factor = self.update_partial_conv_binary_mask(mask)
		skip3 = skip3 * mask * rescale_factor

		x = self.Downscale4(skip3)

		mask, rescale_factor = self.update_partial_conv_binary_mask(mask, stride=2)

		x = self.ConvLayer4(x * mask * rescale_factor)

		mask, rescale_factor = self.update_partial_conv_binary_mask(mask, dilation=2)

		x = self.ConvLayer5(x * mask * rescale_factor)

		mask, rescale_factor = self.update_partial_conv_binary_mask(mask, dilation=4)

		x = self.Upscale1(x * mask * rescale_factor)
		#x = self.ConvLayer6(torch.cat([x, skip3], dim=1))
		x = self.ResLayer1(x)
		x = self.ResLayer2(x)
  
		x = self.Upscale2(x + skip3)
		#x = self.ConvLayer7(torch.cat([x, skip2], dim=1))
		x = self.ResLayer3(x)
		x = self.ResLayer4(x)
  
		x = self.Upscale3(x + skip2)
		#x = self.ConvLayer8(torch.cat([x, skip1], dim=1))
		x = self.ResLayer5(x)
		x = self.ResLayer6(x)

		x = self.Upscale4(x + skip1)

		x = self.OutConv1(x + skip0)#self.ConvLayer9(torch.cat([x, skip0], dim=1))

		x = self.OutConv2(x)

		x = self.Sigmoid(x)

		return x






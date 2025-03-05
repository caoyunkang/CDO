import torch
import torch.nn as nn
try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.resnet import Bottleneck

from model import get_model
from model import MODEL
from ._base_model import BaseModel


################ For CDO, HRNet #############
import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'seg_hrnet'
_C.MODEL.PRETRAINED = ''
_C.MODEL.ALIGN_CORNERS = True
_C.MODEL.NUM_OUTPUTS = 1
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.MODEL.OCR = CN()
_C.MODEL.OCR.MID_CHANNELS = 512
_C.MODEL.OCR.KEY_CHANNELS = 256
_C.MODEL.OCR.DROPOUT = 0.05
_C.MODEL.OCR.SCALE = 1

_C.LOSS = CN()
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = False
_C.LOSS.BALANCE_WEIGHTS = [1]

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = 'list/cityscapes/train.lst'
_C.DATASET.EXTRA_TRAIN_SET = ''
_C.DATASET.TEST_SET = 'list/cityscapes/val.lst'

# training
_C.TRAIN = CN()

_C.TRAIN.FREEZE_LAYERS = ''
_C.TRAIN.FREEZE_EPOCHS = -1
_C.TRAIN.NONBACKBONE_KEYWORDS = []
_C.TRAIN.NONBACKBONE_MULT = 10

_C.TRAIN.IMAGE_SIZE = [1024, 512]  # width * height
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16

_C.TRAIN.RANDOM_BRIGHTNESS = False
_C.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE = 10

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
# only using some training samples
_C.TRAIN.NUM_SAMPLES = 0

# testing
_C.TEST = CN()

_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048

_C.TEST.BATCH_SIZE_PER_GPU = 32
# only testing some samples
_C.TEST.NUM_SAMPLES = 0

_C.TEST.MODEL_FILE = ''
_C.TEST.FLIP_TEST = False
_C.TEST.MULTI_SCALE = False
_C.TEST.SCALE_LIST = [1]

_C.TEST.OUTPUT_INDEX = -1

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
	cfg.defrost()

	cfg.merge_from_file(args.cfg)
	cfg.merge_from_list(args.opts)

	cfg.freeze()


from yacs.config import CfgNode as CN


# configs for HRNet48
HRNET_48 = CN()
HRNET_48.FINAL_CONV_KERNEL = 1

HRNET_48.STAGE1 = CN()
HRNET_48.STAGE1.NUM_MODULES = 1
HRNET_48.STAGE1.NUM_BRANCHES = 1
HRNET_48.STAGE1.NUM_BLOCKS = [4]
HRNET_48.STAGE1.NUM_CHANNELS = [64]
HRNET_48.STAGE1.BLOCK = 'BOTTLENECK'
HRNET_48.STAGE1.FUSE_METHOD = 'SUM'

HRNET_48.STAGE2 = CN()
HRNET_48.STAGE2.NUM_MODULES = 1
HRNET_48.STAGE2.NUM_BRANCHES = 2
HRNET_48.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_48.STAGE2.NUM_CHANNELS = [48, 96]
HRNET_48.STAGE2.BLOCK = 'BASIC'
HRNET_48.STAGE2.FUSE_METHOD = 'SUM'

HRNET_48.STAGE3 = CN()
HRNET_48.STAGE3.NUM_MODULES = 4
HRNET_48.STAGE3.NUM_BRANCHES = 3
HRNET_48.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_48.STAGE3.NUM_CHANNELS = [48, 96, 192]
HRNET_48.STAGE3.BLOCK = 'BASIC'
HRNET_48.STAGE3.FUSE_METHOD = 'SUM'

HRNET_48.STAGE4 = CN()
HRNET_48.STAGE4.NUM_MODULES = 3
HRNET_48.STAGE4.NUM_BRANCHES = 4
HRNET_48.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_48.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
HRNET_48.STAGE4.BLOCK = 'BASIC'
HRNET_48.STAGE4.FUSE_METHOD = 'SUM'


# configs for HRNet32
HRNET_32 = CN()
HRNET_32.FINAL_CONV_KERNEL = 1

HRNET_32.STAGE1 = CN()
HRNET_32.STAGE1.NUM_MODULES = 1
HRNET_32.STAGE1.NUM_BRANCHES = 1
HRNET_32.STAGE1.NUM_BLOCKS = [4]
HRNET_32.STAGE1.NUM_CHANNELS = [64]
HRNET_32.STAGE1.BLOCK = 'BOTTLENECK'
HRNET_32.STAGE1.FUSE_METHOD = 'SUM'

HRNET_32.STAGE2 = CN()
HRNET_32.STAGE2.NUM_MODULES = 1
HRNET_32.STAGE2.NUM_BRANCHES = 2
HRNET_32.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_32.STAGE2.NUM_CHANNELS = [32, 64]
HRNET_32.STAGE2.BLOCK = 'BASIC'
HRNET_32.STAGE2.FUSE_METHOD = 'SUM'

HRNET_32.STAGE3 = CN()
HRNET_32.STAGE3.NUM_MODULES = 4
HRNET_32.STAGE3.NUM_BRANCHES = 3
HRNET_32.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_32.STAGE3.NUM_CHANNELS = [32, 64, 128]
HRNET_32.STAGE3.BLOCK = 'BASIC'
HRNET_32.STAGE3.FUSE_METHOD = 'SUM'

HRNET_32.STAGE4 = CN()
HRNET_32.STAGE4.NUM_MODULES = 3
HRNET_32.STAGE4.NUM_BRANCHES = 4
HRNET_32.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_32.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
HRNET_32.STAGE4.BLOCK = 'BASIC'
HRNET_32.STAGE4.FUSE_METHOD = 'SUM'


# configs for HRNet18
HRNET_18 = CN()
HRNET_18.FINAL_CONV_KERNEL = 1

HRNET_18.STAGE1 = CN()
HRNET_18.STAGE1.NUM_MODULES = 1
HRNET_18.STAGE1.NUM_BRANCHES = 1
HRNET_18.STAGE1.NUM_BLOCKS = [4]
HRNET_18.STAGE1.NUM_CHANNELS = [64]
HRNET_18.STAGE1.BLOCK = 'BOTTLENECK'
HRNET_18.STAGE1.FUSE_METHOD = 'SUM'

HRNET_18.STAGE2 = CN()
HRNET_18.STAGE2.NUM_MODULES = 1
HRNET_18.STAGE2.NUM_BRANCHES = 2
HRNET_18.STAGE2.NUM_BLOCKS = [4, 4]
HRNET_18.STAGE2.NUM_CHANNELS = [18, 36]
HRNET_18.STAGE2.BLOCK = 'BASIC'
HRNET_18.STAGE2.FUSE_METHOD = 'SUM'

HRNET_18.STAGE3 = CN()
HRNET_18.STAGE3.NUM_MODULES = 4
HRNET_18.STAGE3.NUM_BRANCHES = 3
HRNET_18.STAGE3.NUM_BLOCKS = [4, 4, 4]
HRNET_18.STAGE3.NUM_CHANNELS = [18, 36, 72]
HRNET_18.STAGE3.BLOCK = 'BASIC'
HRNET_18.STAGE3.FUSE_METHOD = 'SUM'

HRNET_18.STAGE4 = CN()
HRNET_18.STAGE4.NUM_MODULES = 3
HRNET_18.STAGE4.NUM_BRANCHES = 4
HRNET_18.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HRNET_18.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
HRNET_18.STAGE4.BLOCK = 'BASIC'
HRNET_18.STAGE4.FUSE_METHOD = 'SUM'


HRNET_MODEL_CONFIGS = {
    'hrnet18': HRNET_18,
    'hrnet32': HRNET_32,
    'hrnet48': HRNET_48,
}


# high_resoluton_net related params for segmentation
HIGH_RESOLUTION_NET = CN()
HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
HIGH_RESOLUTION_NET.STEM_INPLANES = 64
HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
HIGH_RESOLUTION_NET.WITH_HEAD = True

HIGH_RESOLUTION_NET.STAGE2 = CN()
HIGH_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
HIGH_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
HIGH_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [32, 64]
HIGH_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'

HIGH_RESOLUTION_NET.STAGE3 = CN()
HIGH_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
HIGH_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
HIGH_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [32, 64, 128]
HIGH_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'

HIGH_RESOLUTION_NET.STAGE4 = CN()
HIGH_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
HIGH_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HIGH_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
HIGH_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'

MODEL_EXTRAS = {
    'seg_hrnet': HIGH_RESOLUTION_NET,
}

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by RainbowSecret (yhyuan@pku.edu.cn)
# ------------------------------------------------------------------------------

import os
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.hub import load_state_dict_from_url
import numpy as np

logger = logging.getLogger('hrnet_backbone')

__all__ = ['hrnet18', 'hrnet32', 'hrnet48']  # 表示最后输出的通道数

model_urls = {
	'hrnet18_imagenet': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
	'hrnet32_imagenet': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
	'hrnet48_imagenet': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
	'hrnet48_cityscapes': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
	'hrnet48_ocr_cityscapes': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class HighResolutionModule(nn.Module):
	def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
				 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
		super(HighResolutionModule, self).__init__()
		self._check_branches(
			num_branches, blocks, num_blocks, num_inchannels, num_channels)

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self.norm_layer = norm_layer

		self.num_inchannels = num_inchannels
		self.fuse_method = fuse_method
		self.num_branches = num_branches

		self.multi_scale_output = multi_scale_output

		self.branches = self._make_branches(
			num_branches, blocks, num_blocks, num_channels)
		self.fuse_layers = self._make_fuse_layers()
		self.relu = nn.ReLU(inplace=True)

	def _check_branches(self, num_branches, blocks, num_blocks,
						num_inchannels, num_channels):
		if num_branches != len(num_blocks):
			error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
				num_branches, len(num_blocks))
			logger.error(error_msg)
			raise ValueError(error_msg)

		if num_branches != len(num_channels):
			error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
				num_branches, len(num_channels))
			logger.error(error_msg)
			raise ValueError(error_msg)

		if num_branches != len(num_inchannels):
			error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
				num_branches, len(num_inchannels))
			logger.error(error_msg)
			raise ValueError(error_msg)

	def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
						 stride=1):
		downsample = None
		if stride != 1 or \
				self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.num_inchannels[branch_index],
						  num_channels[branch_index] * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				self.norm_layer(num_channels[branch_index] * block.expansion),
			)

		layers = []
		layers.append(block(self.num_inchannels[branch_index],
							num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
		self.num_inchannels[branch_index] = \
			num_channels[branch_index] * block.expansion
		for i in range(1, num_blocks[branch_index]):
			layers.append(block(self.num_inchannels[branch_index],
								num_channels[branch_index], norm_layer=self.norm_layer))

		return nn.Sequential(*layers)

	def _make_branches(self, num_branches, block, num_blocks, num_channels):
		branches = []

		for i in range(num_branches):
			branches.append(
				self._make_one_branch(i, block, num_blocks, num_channels))

		return nn.ModuleList(branches)

	def _make_fuse_layers(self):
		if self.num_branches == 1:
			return None

		num_branches = self.num_branches
		num_inchannels = self.num_inchannels
		fuse_layers = []
		for i in range(num_branches if self.multi_scale_output else 1):
			fuse_layer = []
			for j in range(num_branches):
				if j > i:
					fuse_layer.append(nn.Sequential(
						nn.Conv2d(num_inchannels[j],
								  num_inchannels[i],
								  1,
								  1,
								  0,
								  bias=False),
						self.norm_layer(num_inchannels[i])))
				elif j == i:
					fuse_layer.append(None)
				else:
					conv3x3s = []
					for k in range(i - j):
						if k == i - j - 1:
							num_outchannels_conv3x3 = num_inchannels[i]
							conv3x3s.append(nn.Sequential(
								nn.Conv2d(num_inchannels[j],
										  num_outchannels_conv3x3,
										  3, 2, 1, bias=False),
								self.norm_layer(num_outchannels_conv3x3)))
						else:
							num_outchannels_conv3x3 = num_inchannels[j]
							conv3x3s.append(nn.Sequential(
								nn.Conv2d(num_inchannels[j],
										  num_outchannels_conv3x3,
										  3, 2, 1, bias=False),
								self.norm_layer(num_outchannels_conv3x3),
								nn.ReLU(inplace=True)))
					fuse_layer.append(nn.Sequential(*conv3x3s))
			fuse_layers.append(nn.ModuleList(fuse_layer))

		return nn.ModuleList(fuse_layers)

	def get_num_inchannels(self):
		return self.num_inchannels

	def forward(self, x):
		if self.num_branches == 1:
			return [self.branches[0](x[0])]

		for i in range(self.num_branches):
			x[i] = self.branches[i](x[i])

		x_fuse = []
		for i in range(len(self.fuse_layers)):
			y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
			for j in range(1, self.num_branches):
				if i == j:
					y = y + x[j]
				elif j > i:
					width_output = x[i].shape[-1]
					height_output = x[i].shape[-2]
					y = y + F.interpolate(
						self.fuse_layers[i][j](x[j]),
						size=[height_output, width_output],
						mode='bilinear',
						align_corners=True
					)
				else:
					y = y + self.fuse_layers[i][j](x[j])
			x_fuse.append(self.relu(y))

		return x_fuse


blocks_dict = {
	'BASIC': BasicBlock,
	'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

	def __init__(self,
				 cfg,
				 norm_layer=None):
		super(HighResolutionNet, self).__init__()

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self.norm_layer = norm_layer
		# stem network
		# stem net
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
							   bias=False)
		self.bn1 = self.norm_layer(64)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
							   bias=False)
		self.bn2 = self.norm_layer(64)
		self.relu = nn.ReLU(inplace=True)

		# stage 1
		self.stage1_cfg = cfg['STAGE1']
		num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
		block = blocks_dict[self.stage1_cfg['BLOCK']]
		num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
		self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
		stage1_out_channel = block.expansion * num_channels

		# stage 2
		self.stage2_cfg = cfg['STAGE2']
		num_channels = self.stage2_cfg['NUM_CHANNELS']
		block = blocks_dict[self.stage2_cfg['BLOCK']]
		num_channels = [
			num_channels[i] * block.expansion for i in range(len(num_channels))]
		self.transition1 = self._make_transition_layer(
			[stage1_out_channel], num_channels)
		self.stage2, pre_stage_channels = self._make_stage(
			self.stage2_cfg, num_channels)

		# stage 3
		self.stage3_cfg = cfg['STAGE3']
		num_channels = self.stage3_cfg['NUM_CHANNELS']
		block = blocks_dict[self.stage3_cfg['BLOCK']]
		num_channels = [
			num_channels[i] * block.expansion for i in range(len(num_channels))]
		self.transition2 = self._make_transition_layer(
			pre_stage_channels, num_channels)
		self.stage3, pre_stage_channels = self._make_stage(
			self.stage3_cfg, num_channels)

		# stage 4
		self.stage4_cfg = cfg['STAGE4']
		num_channels = self.stage4_cfg['NUM_CHANNELS']
		block = blocks_dict[self.stage4_cfg['BLOCK']]
		num_channels = [
			num_channels[i] * block.expansion for i in range(len(num_channels))]
		self.transition3 = self._make_transition_layer(
			pre_stage_channels, num_channels)
		self.stage4, pre_stage_channels = self._make_stage(
			self.stage4_cfg, num_channels, multi_scale_output=True)

		last_inp_channels = np.int32(np.sum(pre_stage_channels))

		self.last_layer = nn.Sequential(
			nn.Conv2d(
				in_channels=last_inp_channels,
				out_channels=last_inp_channels,
				kernel_size=1,
				stride=1,
				padding=0),
			self.norm_layer(last_inp_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(
				in_channels=last_inp_channels,
				out_channels=19,
				kernel_size=1,
				stride=1,
				padding=0)
		)

	def _make_transition_layer(
			self, num_channels_pre_layer, num_channels_cur_layer):
		num_branches_cur = len(num_channels_cur_layer)
		num_branches_pre = len(num_channels_pre_layer)

		transition_layers = []
		for i in range(num_branches_cur):
			if i < num_branches_pre:
				if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
					transition_layers.append(nn.Sequential(
						nn.Conv2d(num_channels_pre_layer[i],
								  num_channels_cur_layer[i],
								  3,
								  1,
								  1,
								  bias=False),
						self.norm_layer(num_channels_cur_layer[i]),
						nn.ReLU(inplace=True)))
				else:
					transition_layers.append(None)
			else:
				conv3x3s = []
				for j in range(i + 1 - num_branches_pre):
					inchannels = num_channels_pre_layer[-1]
					outchannels = num_channels_cur_layer[i] \
						if j == i - num_branches_pre else inchannels
					conv3x3s.append(nn.Sequential(
						nn.Conv2d(
							inchannels, outchannels, 3, 2, 1, bias=False),
						self.norm_layer(outchannels),
						nn.ReLU(inplace=True)))
				transition_layers.append(nn.Sequential(*conv3x3s))

		return nn.ModuleList(transition_layers)

	def _make_layer(self, block, inplanes, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				self.norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
		inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

		return nn.Sequential(*layers)

	def _make_stage(self, layer_config, num_inchannels,
					multi_scale_output=True):
		num_modules = layer_config['NUM_MODULES']
		num_branches = layer_config['NUM_BRANCHES']
		num_blocks = layer_config['NUM_BLOCKS']
		num_channels = layer_config['NUM_CHANNELS']
		block = blocks_dict[layer_config['BLOCK']]
		fuse_method = layer_config['FUSE_METHOD']

		modules = []
		for i in range(num_modules):
			# multi_scale_output is only used last module
			if not multi_scale_output and i == num_modules - 1:
				reset_multi_scale_output = False
			else:
				reset_multi_scale_output = True

			modules.append(
				HighResolutionModule(num_branches,
									 block,
									 num_blocks,
									 num_inchannels,
									 num_channels,
									 fuse_method,
									 reset_multi_scale_output,
									 norm_layer=self.norm_layer)
			)
			num_inchannels = modules[-1].get_num_inchannels()

		return nn.Sequential(*modules), num_inchannels

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.layer1(x)

		x_list = []
		for i in range(self.stage2_cfg['NUM_BRANCHES']):
			if self.transition1[i] is not None:
				x_list.append(self.transition1[i](x))
			else:
				x_list.append(x)
		y_list = self.stage2(x_list)

		x_list = []
		for i in range(self.stage3_cfg['NUM_BRANCHES']):
			if self.transition2[i] is not None:
				if i < self.stage2_cfg['NUM_BRANCHES']:
					x_list.append(self.transition2[i](y_list[i]))
				else:
					x_list.append(self.transition2[i](y_list[-1]))
			else:
				x_list.append(y_list[i])
		y_list = self.stage3(x_list)

		x_list = []
		for i in range(self.stage4_cfg['NUM_BRANCHES']):
			if self.transition3[i] is not None:
				if i < self.stage3_cfg['NUM_BRANCHES']:
					x_list.append(self.transition3[i](y_list[i]))
				else:
					x_list.append(self.transition3[i](y_list[-1]))
			else:
				x_list.append(y_list[i])
		x = self.stage4(x_list)

		# Upsampling
		x0_h, x0_w = x[0].size(2), x[0].size(3)
		x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
		x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
		x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

		x = torch.cat([x[0], x1, x2, x3], 1)

		x = self.last_layer(x)

		return x


def _hrnet(arch, pretrained, progress, **kwargs):

	model = HighResolutionNet(HRNET_MODEL_CONFIGS[arch], **kwargs)
	if pretrained:
		model_url = model_urls[f'{arch}_imagenet']
		state_dict = load_state_dict_from_url(model_url,
											  progress=progress, file_name=f'{arch}.pth')
		model.load_state_dict(state_dict, strict=False)
	return model


@MODEL.register_module
def hrnet18(pretrained=True, progress=True, **kwargs):
	r"""HRNet-18 model
    """
	return _hrnet('hrnet18', pretrained, progress,
				  **kwargs)

@MODEL.register_module
def hrnet32(pretrained=True, progress=True, **kwargs):
	r"""HRNet-32 model
    """
	return _hrnet('hrnet32', pretrained, progress,
				  **kwargs)

@MODEL.register_module
def hrnet48(pretrained=True, progress=True, **kwargs):
	r"""HRNet-48 model
    """
	return _hrnet('hrnet48', pretrained, progress,
				  **kwargs)


class MultistageHRNet(torch.nn.Module):
	def __init__(self, backbone, pretrained):
		super(MultistageHRNet, self).__init__()
		if backbone == 'hrnet18':
			self.model = hrnet18(pretrained=pretrained)
		elif backbone == 'hrnet32':
			self.model = hrnet32(pretrained=pretrained)
		elif backbone == 'hrnet48':
			self.model = hrnet48(pretrained=pretrained)
		else:
			raise NotImplementedError

	def forward(self, x):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.conv2(x)
		x = self.model.bn2(x)
		x = self.model.relu(x)
		x = self.model.layer1(x)

		x_list = []
		for i in range(self.model.stage2_cfg['NUM_BRANCHES']):
			if self.model.transition1[i] is not None:
				x_list.append(self.model.transition1[i](x))
			else:
				x_list.append(x)
		y_list = self.model.stage2(x_list)

		x_list = []
		for i in range(self.model.stage3_cfg['NUM_BRANCHES']):
			if self.model.transition2[i] is not None:
				if i < self.model.stage2_cfg['NUM_BRANCHES']:
					x_list.append(self.model.transition2[i](y_list[i]))
				else:
					x_list.append(self.model.transition2[i](y_list[-1]))
			else:
				x_list.append(y_list[i])
		y_list = self.model.stage3(x_list)

		x_list = []
		for i in range(self.model.stage4_cfg['NUM_BRANCHES']):
			if self.model.transition3[i] is not None:
				if i < self.model.stage3_cfg['NUM_BRANCHES']:
					x_list.append(self.model.transition3[i](y_list[i]))
				else:
					x_list.append(self.model.transition3[i](y_list[-1]))
			else:
				x_list.append(y_list[i])
		x = self.model.stage4(x_list)

		# Upsampling
		x0_h, x0_w = x[0].size(2), x[0].size(3)
		# x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
		# x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
		# x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
		x1 = x[1]
		x2 = x[2]
		x3 = x[3]

		x0 = x[0]

		x_list = []
		x_list.append(x0)
		x_list.append(x1)
		x_list.append(x2)

		return x_list

@MODEL.register_module
def mshrnet18(pretrained = False, progress = True, **kwargs):
	return MultistageHRNet(pretrained=pretrained, backbone='hrnet18')

@MODEL.register_module
def mshrnet32(pretrained = False, progress = True, **kwargs):
	return MultistageHRNet(pretrained=pretrained, backbone='hrnet32')

@MODEL.register_module
def mshrnet48(pretrained = False, progress = True, **kwargs):
	return MultistageHRNet(pretrained=pretrained, backbone='hrnet48')

from torch.cuda.amp import autocast

class CDO(BaseModel):
	def __init__(self, model_t, model_s):
		super(CDO, self).__init__()
		self.net_t = get_model(model_t)
		self.net_s = get_model(model_s)

		self.set_frozen_layers(['net_t'])


	def forward(self, imgs):
		with torch.no_grad():
			feats_t = self.net_t(imgs)
		feats_s = self.net_s(imgs)

		feats_t = [F.normalize(f, p=2, dim=1) for f in feats_t]
		feats_s = [F.normalize(f, p=2, dim=1) for f in feats_s]

		return feats_t, feats_s

@MODEL.register_module
def cdo(pretrained=False, **kwargs):
	model = CDO(**kwargs)
	return model

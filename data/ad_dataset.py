import copy
import glob
import json
import math
import os
import pickle
import random
import warnings

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.data import get_img_loader
from data.dataset_info import *

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

# from . import DATA
from data import DATA
from data.noise import Simplex_CLASS

# data
# ├── mvtec
#     ├── meta.json
#     ├── bottle
#         ├── train
#             └── good
#                 ├── 000.png
#         ├── test
#             ├── good
#                 ├── 000.png
#             ├── anomaly1
#                 ├── 000.png
#         └── ground_truth
#             ├── anomaly1
#                 ├── 000.png


@DATA.register_module
class DefaultAD(data.Dataset):
	def __init__(self, data_cfg, train=True, transform=None, target_transform=None):

		self.name = data_cfg.name
		assert self.name in DATA_SUBDIR.keys(), f"Only Support {DATA_SUBDIR.keys()}"

		self.root = os.path.join(DATA_ROOT, DATA_SUBDIR[self.name])

		self.train = train
		self.transform = transform
		self.target_transform = target_transform

		self.loader = get_img_loader(data_cfg.loader_type)
		self.loader_target = get_img_loader(data_cfg.loader_type_target)

		self.all_cls_names = CLASS_NAMES[self.name]
		self.data_all = []

		meta_path = EXPERIMENTAL_SETUP[data_cfg.mode]
		meta_info = json.load(open(f'{self.root}/{meta_path}', 'r'))
		meta_info = meta_info['train' if self.train else 'test']
		self.cls_names = data_cfg.cls_names
		if not isinstance(self.cls_names, list):
			self.cls_names = [self.cls_names]
		self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names

		for cls_name in self.cls_names:
			self.data_all.extend(meta_info[cls_name])
		random.shuffle(self.data_all) if self.train else None
		self.length = len(self.data_all)

		self.enable_anomaly_generation = data_cfg.anomaly_generator.enable
		if hasattr(data_cfg, 'anomaly_generator'):
			if data_cfg.anomaly_generator.name:
				self.anomaly_generator = ANOMALY_GENERATOR.get(data_cfg.anomaly_generator.name, None)
				self.anomaly_generator = self.anomaly_generator(**data_cfg.anomaly_generator.kwargs)

		self.normal_idx = []
		self.outlier_idx = []
		for indx, item in enumerate(self.data_all):
			if item['anomaly'] == 0:
				self.normal_idx.append(indx)
			else:
				self.outlier_idx.append(indx)

		#### verbose
		self.verbose_info = []
		for cls in self.cls_names:
			anomalous_samples = [item for item in meta_info[cls] if item['anomaly'] == 1]
			normal_samples = [item for item in meta_info[cls] if item['anomaly'] == 0]

			self.verbose_info.append(f'Training: {train}, Class: {cls}, #Normal: {len(normal_samples)}, #Abnormal: {len(anomalous_samples)}, '
				  f'#Total: {len(normal_samples) + len(anomalous_samples)}')

	def __len__(self):
		return self.length
		# return 1

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		img_path = f'{self.root}/{img_path}'
		img = self.loader(img_path)
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(self.loader_target(f'{self.root}/{mask_path}')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

		return_dict = {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'img_path': img_path}

		if hasattr(self, 'anomaly_generator'):
			if self.train: # only need to generate anomalies during the training stage
				if self.enable_anomaly_generation and anomaly != 1:	# we don't generate anomalies for real anomalies...
					augmented_image, augmented_mask, augmented_anomaly = self.anomaly_generator(img)
				else: # just use the original data
					augmented_image = img
					augmented_mask = img_mask
					augmented_anomaly = anomaly

				augmented_image = self.transform(augmented_image) if self.transform is not None else augmented_image
				augmented_mask = self.target_transform(
					augmented_mask) if self.target_transform is not None and augmented_mask is not None else augmented_mask
				augmented_mask = [] if augmented_mask is None else augmented_mask

				return_dict.update({'augmented_image': augmented_image,
									'augmented_mask': augmented_mask,
									'augmented_anomaly': augmented_anomaly,
									})

		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return_dict.update({'img': img, 'img_mask': img_mask})

		return return_dict


class ToTensor(object):
	def __call__(self, image):
		try:
			image = torch.from_numpy(image.transpose(2, 0, 1))
		except:
			print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
		if not isinstance(image, torch.FloatTensor):
			image = image.float()
		return image


class Normalize(object):
	def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
		self.mean = np.array(mean)
		self.std = np.array(std)

	def __call__(self, image):
		image = (image - self.mean) / self.std
		return image

def get_data_transforms(size, isize):
	data_transforms = transforms.Compose([Normalize(),ToTensor()])
	gt_transforms = transforms.Compose([
		transforms.Resize((size, size)),
		transforms.ToTensor()])
	return data_transforms, gt_transforms


if __name__ == '__main__':
	from argparse import Namespace as _Namespace

	cfg = _Namespace()
	data = _Namespace()
	data.sampler = 'naive'
	# ========== MVTec ==========
	# data.root = 'data/mvtec'
	# data.meta = 'meta.json'
	# # data.cls_names = ['bottle']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# data.root = 'data/mvtec3d'
	# data.meta = 'meta.json'
	# # data.cls_names = ['bagel']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# data.root = 'data/coco'
	# data.meta = 'meta_20_0.json'
	# data.cls_names = ['coco']
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# data.root = 'data/visa'
	# data.meta = 'meta.json'
	# # data.cls_names = ['candle']
	# data.cls_names = []
	# data.loader_type = 'pil'
	# data.loader_type_target = 'pil_L'
	# data_fun = DefaultAD

	# ========== Cifar ==========
	# data.type = 'DefaultAD'
	# data.root = 'data/cifar'
	# data.type_cifar = 'cifar10'
	# data.cls_names = ['cifar']
	# data.uni_setting = True
	# data.one_cls_train = True
	# data.split_idx = 0
	# data_fun = CifarAD

	# ========== Tiny ImageNet ==========
	# data.root = 'data/tiny-imagenet-200'
	# data.cls_names = ['tin']
	# data.loader_type = 'pil'
	# data.split_idx = 0
	# data_fun = TinyINAD

	# ========== Real-IAD ==========
	data.root = 'data/realiad/explicit_full'
	# data.cls_names = ['audiojack']
	data.cls_names = []
	data.loader_type = 'pil'
	data.loader_type_target = 'pil_L'
	data.views = ['C1', 'C2']
	# data.views = []
	data.use_sample = True
	data_fun = RealIAD


	cfg.data = data
	data_debug = data_fun(cfg, train=True)
	# data_debug = data_fun(cfg, train=False)
	for idx, data in enumerate(data_debug):
		break
		if idx > 1000:
			break
		print()


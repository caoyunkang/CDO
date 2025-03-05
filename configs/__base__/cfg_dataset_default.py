from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_dataset_default(Namespace):

	def __init__(self):
		Namespace.__init__(self)

		###### Train DATA
		self.train_data = Namespace()
		self.train_data.anomaly_generator = Namespace()
		self.train_data.sampler = Namespace()
		self.train_data.sampler.name = 'naive'
		self.train_data.sampler.kwargs = dict()
		self.train_data.loader_type = 'pil'
		self.train_data.loader_type_target = 'pil_L'
		self.train_data.type = 'DefaultAD'
		self.train_data.name = 'mvtec' # mvtec, visa
		self.train_data.mode = 'semi10'
		self.train_data.cls_names = ['carpet'] ###### Set to [] to utilize all classes
		self.train_data.anomaly_generator.name = 'white_noise'
		self.train_data.anomaly_generator.enable = False ### TODO
		self.train_data.anomaly_generator.kwargs = dict()


		###### Test DATA
		self.test_data = Namespace()
		self.test_data.anomaly_generator = Namespace()
		self.test_data.sampler = Namespace()
		self.test_data.sampler.name = 'naive'
		self.test_data.sampler.kwargs = dict()
		self.test_data.loader_type = 'pil'
		self.test_data.loader_type_target = 'pil_L'
		self.test_data.type = 'DefaultAD'
		self.test_data.name = 'mvtec' # mvtec, visa
		self.test_data.mode = 'semi10'
		self.test_data.cls_names = ['carpet'] ###### Set to [] to utilize all classes
		self.test_data.anomaly_generator.name = 'white_noise'
		self.test_data.anomaly_generator.enable = False ### TODO
		self.test_data.anomaly_generator.kwargs = dict()


		mvtec = [
			'carpet', 'grid', 'leather', 'tile', 'wood',
			'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
			'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
		]
		visa = [
			'pcb1', 'pcb2', 'pcb3', 'pcb4',
			'macaroni1', 'macaroni2', 'capsules', 'candle',
			'cashew', 'chewinggum', 'fryum', 'pipe_fryum',
		]
		mvtec3d = [
			'bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
			'foam', 'peach', 'potato', 'rope', 'tire',
		]
		medical = [
			'brain', 'liver', 'retinal',
		]

		# --> for RealIAD
		# self.data.type = 'RealIAD'
		# self.data.root = 'data/realiad'
		# self.data.use_sample = False
		# self.data.views = []  # ['C1', 'C2', 'C3', 'C4', 'C5']
		# self.data.cls_names = ['audiojack', 'bottle_cap', 'button_battery']
		realiad = [
			'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser',
			'fire_hood', 'mint', 'mounts', 'pcb', 'phone_battery',
			'plastic_nut', 'plastic_plug', 'porcelain_doll', 'regulator', 'rolled_strip_base',
			'sim_card_set', 'switch', 'tape', 'terminalblock', 'toothbrush',
			'toy', 'toy_brick', 'transistor1', 'u_block', 'usb',
			'usb_adaptor', 'vcpill', 'wooden_beads', 'woodstick', 'zipper',
		]

		self.train_data.train_transforms = [
			dict(type='Resize', size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(256, 256)),
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]
		self.train_data.test_transforms = self.train_data.train_transforms
		self.train_data.target_transforms = [
			dict(type='Resize', size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(256, 256)),
			dict(type='ToTensor'),
		]

		self.test_data.train_transforms = [
			dict(type='Resize', size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(256, 256)),
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]
		self.test_data.test_transforms = self.test_data.train_transforms
		self.test_data.target_transforms = [
			dict(type='Resize', size=(256, 256), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(256, 256)),
			dict(type='ToTensor'),
		]


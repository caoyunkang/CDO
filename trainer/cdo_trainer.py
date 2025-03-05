from util.compute_am import compute_discrepancy_map, maximum_as_anomaly_score
from ._base_trainer import BaseTrainer
from . import TRAINER


@TRAINER.register_module
class CDOTrainer(BaseTrainer):
	def __init__(self, cfg):
		super(CDOTrainer, self).__init__(cfg)

	def pre_train(self):
		pass

	def pre_test(self):
		pass

	def set_input(self, inputs, train=True):
		self.imgs = inputs['img'].cuda()
		self.imgs_mask = inputs['img_mask'].cuda()

		if train:
			self.augmented_image = inputs['augmented_image'].cuda()
			self.augmented_mask = inputs['augmented_mask'].cuda()

		self.bs = self.imgs.shape[0]

		### Note: necessray for evaluations and visualizations
		self.cls_name = inputs['cls_name']
		self.anomaly = inputs['anomaly']
		self.img_path = inputs['img_path']

	def forward(self, train=True):
		if train:
			self.feats_t, self.feats_s = self.net(self.augmented_image)
		else:
			self.feats_t, self.feats_s = self.net(self.imgs)

	def compute_loss(self, train=True):
		if train:
			loss = self.loss_terms['cdo_loss'](self.feats_t, self.feats_s, self.augmented_mask)
			loss_log = {'cdo_loss': loss}
		else:
			loss = self.loss_terms['cdo_loss'](self.feats_t, self.feats_s, self.imgs_mask)
			loss_log = {'cdo_loss': loss}
		return loss, loss_log

	def compute_anomaly_scores(self):
		anomaly_map, anomaly_map_list = compute_discrepancy_map(self.feats_t, self.feats_s,
														[self.imgs.shape[2], self.imgs.shape[3]],
														uni_am=False, amap_mode='add', gaussian_sigma=4)
		anomaly_score = maximum_as_anomaly_score(anomaly_map, 0.01)
		return anomaly_map, anomaly_score


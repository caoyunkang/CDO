import glob
import importlib

import torch
import torch.nn as nn
from timm.models._registry import is_model_in_modules
from timm.models._helpers import load_checkpoint
from timm.models.layers import set_layer_config
from timm.models._hub import load_model_config_from_hf
from timm.models._factory import parse_model_name
from util.registry import Registry
MODEL = Registry('Model')

def get_model(cfg_model):
	model_name = cfg_model.name
	kwargs = {k: v for k, v in cfg_model.kwargs.items()}
	model_fn = MODEL.get_module(model_name)
	pretrained = kwargs.pop('pretrained')
	checkpoint_path = kwargs.pop('checkpoint_path')
	strict = kwargs.pop('strict')

	if model_name.startswith('timm_'):
		if 'hf' in kwargs:
			model_name_hf = kwargs.pop('hf')
		else:
			model_name_hf = None
		if model_name_hf:
			pretrained_cfg, model_name_hf = load_model_config_from_hf(model_name_hf)
			pretrained_cfg['url'] = ''
		else:
			pretrained_cfg = None
		with set_layer_config(scriptable=None, exportable=None, no_jit=None):
			model = model_fn(pretrained=pretrained, pretrained_cfg=pretrained_cfg, **kwargs)
		if not pretrained and pretrained_cfg is None and checkpoint_path is not None:
			load_checkpoint(model, checkpoint_path, strict=strict)
	else:
		model = model_fn(pretrained=pretrained, **kwargs)
		if checkpoint_path:
			ckpt = torch.load(checkpoint_path, map_location='cpu')
			if 'net' in ckpt.keys():
				state_dict = ckpt['net']
			else:
				state_dict = ckpt
			if not strict and False:
				no_ft_keywords = model.no_ft_keywords()
				for no_ft_keyword in no_ft_keywords:
					del state_dict[no_ft_keyword]
				ft_head_keywords, num_classes = model.ft_head_keywords()
				for ft_head_keyword in ft_head_keywords:
					if state_dict[ft_head_keyword].shape[0] <= num_classes:
						del state_dict[ft_head_keyword]
					elif state_dict[ft_head_keyword].shape[0] == num_classes:
						continue
					else:
						state_dict[ft_head_keyword] = state_dict[ft_head_keyword][:num_classes]
			# load ckpt
			if isinstance(model, nn.Module):
				model.load_state_dict(state_dict, strict=strict)
			else:
				for sub_model_name, sub_state_dict in state_dict.items():
					sub_model = getattr(model, sub_model_name, None)
					sub_model.load_state_dict(sub_state_dict, strict=strict) if sub_model else None
	return model

files = glob.glob('model/[!_]*.py')
for file in files:
	model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))

import glob
import importlib

from util.registry import Registry
TRAINER = Registry('Trainer')


# def get_trainer(cfg):
# 	module_name = f"trainer.{cfg.trainer.name.lower()}"
# 	model_lib = importlib.import_module(module_name)
# 	return TRAINER.get_module(cfg.trainer.name)(cfg)

files = glob.glob('trainer/[!_]*.py')
for file in files:
	model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))


def get_trainer(cfg):
	return TRAINER.get_module(cfg.trainer.name)(cfg)
